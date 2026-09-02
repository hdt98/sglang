import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.layers.attention.dsa import kpool_plan
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    _topk_from_pooled_history_logits_unfused,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKPoolMqaBackend(CustomTestCase):
    def setUp(self):
        super().setUp()
        dsa_indexer_kpool.IndexerKPool._last_ragged_allocator_shape.clear()
        dsa_indexer_kpool._RAGGED_LOGITS_WORKSPACES.clear()
        dsa_indexer_kpool._SEGMENTED_RAGGED_LOGGED_DEVICES.clear()

    def test_ragged_logits_workspace_reuses_power_of_two_bucket(self):
        device = torch.device("cpu")
        first = dsa_indexer_kpool._get_ragged_logits_workspace(3, 257, device)
        smaller = dsa_indexer_kpool._get_ragged_logits_workspace(2, 128, device)

        self.assertEqual(first.shape, (4, 512))
        self.assertEqual(first.data_ptr(), smaller.data_ptr())

        grown = dsa_indexer_kpool._get_ragged_logits_workspace(3, 513, device)
        self.assertEqual(grown.shape, (4, 1024))
        self.assertNotEqual(first.data_ptr(), grown.data_ptr())

    def test_rocm_reuses_ragged_logits_workspace_when_enabled(self):
        marker = object()
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.reuse_ragged_logits = True
        indexer.hip_ragged_mqa_logits = MagicMock(return_value=marker)
        q_fp8 = torch.empty((3, 2, 4))
        k_fp8 = torch.empty((257, 4))
        k_scale = torch.empty(257)
        weights = torch.empty((3, 2))
        starts = torch.zeros(3, dtype=torch.int32)
        ends = torch.full((3,), 257, dtype=torch.int32)

        with patch.object(dsa_indexer_kpool, "is_hip", return_value=True):
            result = indexer._fp8_mqa_logits(
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                starts,
                ends,
                clean_logits=True,
            )

        self.assertIs(result, marker)
        kwargs = indexer.hip_ragged_mqa_logits.call_args.kwargs
        self.assertTrue(kwargs["clean_logits"])
        self.assertEqual(kwargs["out"].shape, (4, 512))

    def test_segmented_ragged_topk_uses_request_local_windows(self):
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer._fp8_mqa_logits = MagicMock(
            side_effect=lambda q, k, *_args, **_kwargs: torch.empty(
                (q.shape[0], k.shape[0]), dtype=torch.float32
            )
        )
        topk_call = 0

        def fake_topk(logits, *_args, **_kwargs):
            nonlocal topk_call
            topk_call += 1
            return torch.full((logits.shape[0], 3), topk_call, dtype=torch.int32)

        indexer._topk_from_kpool_logits = MagicMock(side_effect=fake_topk)
        q = torch.empty((5, 2, 4))
        k = torch.empty((12, 4))
        result = indexer._run_segmented_ragged_topk(
            q_fp8=q,
            k_fp8=k,
            k_scale=torch.empty(12),
            weights=torch.empty((5, 2)),
            ks_per_q=torch.tensor([0, 1, 1, 7, 8], dtype=torch.int32),
            ke_per_q=torch.tensor([4, 4, 4, 12, 12], dtype=torch.int32),
            pool_lens=torch.tensor([4, 3, 3, 5, 4], dtype=torch.int32),
            seq_lens=torch.tensor([16, 12, 12, 20, 16], dtype=torch.int32),
            total_q=6,
            segments=((0, 3, 0, 7), (3, 5, 7, 12)),
            page_table=None,
            topk_offsets=None,
            page_table_row_index=None,
        )

        self.assertEqual(indexer._fp8_mqa_logits.call_count, 2)
        first_call, second_call = indexer._fp8_mqa_logits.call_args_list
        self.assertEqual(first_call.args[0].shape[0], 3)
        self.assertEqual(first_call.args[1].shape[0], 7)
        self.assertEqual(second_call.args[0].shape[0], 2)
        self.assertEqual(second_call.args[1].shape[0], 5)
        torch.testing.assert_close(
            second_call.args[4], torch.tensor([0, 1], dtype=torch.int32)
        )
        torch.testing.assert_close(
            result,
            torch.tensor(
                [[1, 1, 1]] * 3 + [[2, 2, 2]] * 2 + [[-1, -1, -1]],
                dtype=torch.int32,
            ),
        )

    def test_kpool_write_plan_update_runs_on_rocm(self):
        metadata = MagicMock()
        metadata.kpool_write_plan.pool_schedule_metadata = None
        forward_mode = MagicMock()
        forward_mode.is_target_verify.return_value = True
        forward_mode.is_decode_or_idle.return_value = False
        forward_mode.is_draft_extend_v2.return_value = False

        with (
            patch.object(kpool_plan, "is_cuda", return_value=False),
            patch.object(kpool_plan, "update_kpool_write_plan_cuda_graph") as update,
        ):
            kpool_plan.update_kpool_write_plan(
                metadata,
                write_start=torch.tensor([0], dtype=torch.int32),
                req_pool_indices=torch.tensor([0], dtype=torch.int32),
                real_page_table=torch.tensor([[1]], dtype=torch.int32),
                pool_size=4,
                real_page_size=64,
                num_draft_tokens=6,
                forward_mode=forward_mode,
                slots_per_page=16,
            )

        update.assert_called_once()

    def test_cuda_tilelang_selector_reads_heads_from_unexpanded_query(self):
        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            self.assertFalse(
                dsa_indexer_kpool.IndexerKPool._should_use_tilelang_paged_mqa_logits(
                    torch.empty(1, 32, 128)
                )
            )
            self.assertTrue(
                dsa_indexer_kpool.IndexerKPool._should_use_tilelang_paged_mqa_logits(
                    torch.empty(1, 16, 128)
                )
            )

    def test_rocm_uses_aiter_mqa_logits(self):
        marker = object()
        aiter_impl = MagicMock(return_value=marker)
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.reuse_ragged_logits = False
        indexer.hip_ragged_mqa_logits = aiter_impl

        args = tuple(object() for _ in range(6))
        with patch.object(dsa_indexer_kpool, "is_hip", return_value=True):
            result = indexer._fp8_mqa_logits(*args, clean_logits=False)

        self.assertIs(result, marker)
        aiter_impl.assert_called_once_with(*args, clean_logits=False)

    def test_cuda_keeps_deep_gemm_mqa_logits(self):
        marker = object()
        deep_gemm = MagicMock()
        deep_gemm.fp8_mqa_logits.return_value = marker
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        q_fp8, k_fp8, k_scale, weights, starts, ends = (object() for _ in range(6))

        with (
            patch.object(dsa_indexer_kpool, "is_hip", return_value=False),
            patch.object(dsa_indexer_kpool, "deep_gemm", deep_gemm, create=True),
        ):
            result = indexer._fp8_mqa_logits(
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                starts,
                ends,
                clean_logits=True,
            )

        self.assertIs(result, marker)
        deep_gemm.fp8_mqa_logits.assert_called_once_with(
            q_fp8,
            (k_fp8, k_scale),
            weights,
            starts,
            ends,
            clean_logits=True,
        )

    def test_ragged_allocator_trim_runs_once_per_shape_at_high_watermark(self):
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.kpool_cache_trim_threshold = 0.85
        plan = MagicMock()
        plan.ragged_total_k_rows = 100
        plan.seq_lens_expanded = torch.empty(200)
        gib = 1 << 30

        with (
            patch("torch.cuda.get_device_properties") as properties,
            patch("torch.cuda.memory_reserved", return_value=9 * gib),
            patch("torch.cuda.memory_allocated", return_value=6 * gib),
            patch("torch.cuda.empty_cache") as empty_cache,
        ):
            properties.return_value.total_memory = 10 * gib
            self.assertTrue(
                indexer._maybe_trim_ragged_allocator(plan, torch.device("cuda:0"))
            )
            self.assertFalse(
                indexer._maybe_trim_ragged_allocator(plan, torch.device("cuda:0"))
            )

        empty_cache.assert_called_once_with()

    def test_ragged_allocator_trim_is_default_off(self):
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.kpool_cache_trim_threshold = 0.0

        with patch("torch.cuda.empty_cache") as empty_cache:
            self.assertFalse(
                indexer._maybe_trim_ragged_allocator(
                    MagicMock(), torch.device("cuda:0")
                )
            )

        empty_cache.assert_not_called()

    def test_portable_topk_masks_invalid_groups_and_expands(self):
        logits = torch.tensor([[0.1, 0.9, 0.8, 50.0]], dtype=torch.float32)
        result = _topk_from_pooled_history_logits_unfused(
            logits=logits,
            group_lengths=torch.tensor([3], dtype=torch.int32),
            pool_size=2,
            topk=4,
        )

        torch.testing.assert_close(
            result,
            torch.tensor([[2, 3, 4, 5]], dtype=torch.int32),
        )

    def test_portable_topk_reuses_clean_logits_mask(self):
        logits = torch.tensor(
            [[0.1, 0.9, 0.8, -float("inf")]], dtype=torch.float32
        )
        result = _topk_from_pooled_history_logits_unfused(
            logits=logits,
            group_lengths=torch.tensor([3], dtype=torch.int32),
            pool_size=2,
            topk=4,
            logits_are_clean=True,
        )

        torch.testing.assert_close(
            result,
            torch.tensor([[2, 3, 4, 5]], dtype=torch.int32),
        )


if __name__ == "__main__":
    unittest.main()
