"""Correctness and determinism tests for the fused DSA kpool top-k transform."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.kpool_topk_transform import (
    fast_kpool_topk_transform_fused,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
class TestKpoolTopKTransform(CustomTestCase):
    GROUP_TOPK = 128
    POOL_SIZE = 2
    TOPK = GROUP_TOPK * POOL_SIZE

    def _run_topk(self, score: torch.Tensor) -> torch.Tensor:
        lengths = torch.tensor(
            [score.shape[1]], dtype=torch.int32, device=score.device
        )
        return fast_kpool_topk_transform_fused(
            score=score,
            lengths=lengths,
            pool_size=self.POOL_SIZE,
            topk=self.TOPK,
        )

    def _expand_sorted_groups(self, group_ids: torch.Tensor) -> torch.Tensor:
        slots = torch.arange(
            self.POOL_SIZE, dtype=torch.int64, device=group_ids.device
        )
        return (
            (group_ids.to(torch.int64).unsqueeze(1) * self.POOL_SIZE + slots)
            .reshape(1, self.TOPK)
            .to(torch.int32)
        )

    def test_exact_boundary_ties_choose_lower_group_ids_deterministically(self):
        score = torch.zeros((1, 4096), dtype=torch.float32, device="cuda")
        expected = torch.arange(
            self.TOPK, dtype=torch.int32, device=score.device
        ).unsqueeze(0)

        for _ in range(20):
            torch.testing.assert_close(
                self._run_topk(score), expected, atol=0, rtol=0
            )

    def test_selected_groups_are_emitted_in_canonical_index_order(self):
        score = torch.full((1, 4096), -1000.0, dtype=torch.float32, device="cuda")
        selected_groups = (torch.arange(self.GROUP_TOPK, device="cuda") * 31) % 4096
        score[0, selected_groups] = torch.arange(
            1, self.GROUP_TOPK + 1, dtype=torch.float32, device="cuda"
        )
        expected = self._expand_sorted_groups(torch.sort(selected_groups).values)

        torch.testing.assert_close(self._run_topk(score), expected, atol=0, rtol=0)

    def test_coarse_bin_overflow_keeps_late_higher_scores(self):
        score = torch.full((1, 8192), 1.0, dtype=torch.float32, device="cuda")
        score[:, 4096:] = 1.001
        expected_groups = torch.arange(
            4096,
            4096 + self.GROUP_TOPK,
            dtype=torch.int64,
            device=score.device,
        )
        expected = self._expand_sorted_groups(expected_groups)

        torch.testing.assert_close(self._run_topk(score), expected, atol=0, rtol=0)

    def test_canonical_sort_is_correct_under_concurrent_stream_load(self):
        group_topk = 512
        pool_size = 2
        topk = group_topk * pool_size
        group_ids = torch.arange(4096, dtype=torch.int64, device="cuda")
        score_row = ((group_ids * 2053) % 4096).to(torch.float32)
        score = score_row.unsqueeze(0).expand(64, -1).contiguous()
        lengths = torch.full(
            (score.shape[0],), score.shape[1], dtype=torch.int32, device="cuda"
        )
        selected = torch.argsort(score_row, descending=True, stable=True)[:group_topk]
        selected = torch.sort(selected).values
        slots = torch.arange(pool_size, dtype=torch.int64, device="cuda")
        expected_row = (
            (selected.unsqueeze(1) * pool_size + slots).reshape(topk).to(torch.int32)
        )
        expected = expected_row.unsqueeze(0).expand(score.shape[0], -1)

        streams = [torch.cuda.Stream() for _ in range(8)]
        outputs = []
        for stream in streams:
            with torch.cuda.stream(stream):
                outputs.append(
                    fast_kpool_topk_transform_fused(
                        score=score,
                        lengths=lengths,
                        pool_size=pool_size,
                        topk=topk,
                    )
                )
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for output in outputs:
            torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_unfused_switch_preserves_page_mapping_and_tail(self):
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            topk_from_pooled_history_logits,
        )

        torch.manual_seed(0)
        rows = 2
        pool_size = 4
        topk = 2048
        group_topk = topk // pool_size
        pooled_len = 21_784 // pool_size
        seq_len = 21_785
        scores = torch.randn(
            rows, pooled_len, dtype=torch.float32, device="cuda"
        )
        group_lengths = torch.full(
            (rows,), pooled_len, dtype=torch.int32, device="cuda"
        )
        seq_lens = torch.full(
            (rows,), seq_len, dtype=torch.int32, device="cuda"
        )
        page_table = torch.arange(
            4 * seq_len, dtype=torch.int32, device="cuda"
        ).view(4, seq_len)
        page_table_row_index = torch.tensor(
            [3, 1], dtype=torch.int32, device="cuda"
        )

        selected_groups = torch.topk(
            scores, group_topk, dim=-1, sorted=False
        ).indices
        token_offsets = torch.arange(pool_size, device="cuda")
        selected_tokens = (
            selected_groups.unsqueeze(-1) * pool_size + token_offsets
        ).reshape(rows, topk)
        selected_page_table = page_table.index_select(
            0, page_table_row_index.to(torch.int64)
        )
        expected_history = torch.gather(selected_page_table, 1, selected_tokens)
        expected_tail = torch.full(
            (rows, pool_size - 1), -1, dtype=torch.int32, device="cuda"
        )
        expected_tail[:, 0] = selected_page_table[:, seq_len - 1]

        with (
            patch.object(envs.SGLANG_DSA_FUSE_TOPK, "get", return_value=False),
            patch(
                "sglang.kernels.ops.moe.kpool_topk_transform."
                "fast_kpool_topk_transform_fused",
                side_effect=AssertionError("unfused mode called the fused kernel"),
            ),
        ):
            result = topk_from_pooled_history_logits(
                logits=scores,
                group_lengths=group_lengths,
                pool_size=pool_size,
                topk=topk,
                page_table=page_table,
                seq_lens=seq_lens,
                out_rows=rows + 1,
                page_table_row_index=page_table_row_index,
            )

        self.assertEqual(result.shape, (rows + 1, topk + pool_size - 1))
        torch.testing.assert_close(
            torch.sort(result[:rows, :topk], dim=-1).values,
            torch.sort(expected_history, dim=-1).values,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            result[:rows, topk:], expected_tail, atol=0, rtol=0
        )
        self.assertTrue(torch.all(result[rows] == -1))


if __name__ == "__main__":
    unittest.main()
