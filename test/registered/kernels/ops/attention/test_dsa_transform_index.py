import unittest
from unittest.mock import patch

import torch

import sglang.kernels.ops.attention.dsa.transform_index as transform_index_module
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode_fast,
    transform_index_page_table_prefill_fast,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

TOPK = 2048


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDSATransformIndex(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def _make_page_table(self, rows: int, context_length: int) -> torch.Tensor:
        columns = torch.arange(context_length, dtype=torch.int32, device=self.device)
        row_bias = (
            torch.arange(rows, dtype=torch.int32, device=self.device).unsqueeze(1) * 17
        )
        return columns.unsqueeze(0) + row_bias

    def _make_topk(self, rows: int, context_length: int) -> torch.Tensor:
        topk = (
            torch.arange(TOPK, dtype=torch.int64, device=self.device)
            .remainder(context_length)
            .repeat(rows, 1)
        )
        if rows > 0:
            topk[:, 0] = 0
            topk[:, 1] = context_length - 1
            topk[:, 257::257] = -1
        return topk

    def _expected(
        self,
        page_table: torch.Tensor,
        topk_indices: torch.Tensor,
        extend_lens_cpu: list[int],
        output_num_tokens: int,
        page_table_is_expanded: bool,
    ) -> torch.Tensor:
        real_num_tokens = sum(extend_lens_cpu)
        expected = torch.full(
            (output_num_tokens, TOPK),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        if real_num_tokens == 0:
            return expected

        if page_table_is_expanded:
            source_rows = page_table[:real_num_tokens]
        else:
            request_ids = torch.repeat_interleave(
                torch.arange(
                    len(extend_lens_cpu), dtype=torch.int64, device=self.device
                ),
                torch.tensor(extend_lens_cpu, dtype=torch.int64, device=self.device),
            )
            source_rows = page_table[request_ids]

        real_topk = topk_indices[:real_num_tokens]
        torch.gather(
            source_rows,
            dim=1,
            index=real_topk.clamp(min=0),
            out=expected[:real_num_tokens],
        )
        expected[:real_num_tokens][real_topk < 0] = -1
        return expected

    def _check_decode_case(
        self,
        batch_size: int,
        context_length: int,
        *,
        zero_row_stride: bool = False,
        provide_result: bool = False,
    ) -> None:
        if zero_row_stride:
            page_table = self._make_page_table(1, context_length).expand(batch_size, -1)
        else:
            page_table = self._make_page_table(batch_size, context_length)
        topk_indices = self._make_topk(batch_size, context_length)
        expected = torch.empty(
            (batch_size, TOPK), dtype=torch.int32, device=self.device
        )
        torch.gather(
            page_table,
            dim=1,
            index=topk_indices.clamp(min=0),
            out=expected,
        )
        expected[topk_indices < 0] = -1
        result = torch.empty_like(expected) if provide_result else None

        actual = transform_index_page_table_decode_fast(
            page_table=page_table,
            topk_indices=topk_indices,
            result=result,
        )
        torch.cuda.synchronize()
        if result is not None:
            self.assertIs(actual, result)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def _check_case(
        self,
        extend_lens_cpu: list[int],
        context_length: int,
        *,
        page_table_is_expanded: bool,
        topk_padding: int = 0,
        output_padding: int = 0,
    ) -> None:
        real_num_tokens = sum(extend_lens_cpu)
        page_table_rows = (
            real_num_tokens if page_table_is_expanded else len(extend_lens_cpu)
        )
        topk_num_tokens = real_num_tokens + topk_padding
        output_num_tokens = topk_num_tokens + output_padding
        page_table = self._make_page_table(page_table_rows, context_length)
        topk_indices = self._make_topk(topk_num_tokens, context_length)
        expected = self._expected(
            page_table,
            topk_indices,
            extend_lens_cpu,
            output_num_tokens,
            page_table_is_expanded,
        )

        actual = transform_index_page_table_prefill_fast(
            page_table=page_table,
            topk_indices=topk_indices,
            extend_lens_cpu=extend_lens_cpu,
            output_num_tokens=output_num_tokens,
            page_table_is_expanded=page_table_is_expanded,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prefill_uses_dedicated_kernel(self):
        extend_lens_cpu = [2, 1]
        context_length = 4096
        page_table = self._make_page_table(len(extend_lens_cpu), context_length)
        topk_indices = self._make_topk(sum(extend_lens_cpu), context_length)

        with patch.object(
            transform_index_module,
            "transform_index_page_table_decode_fast",
            side_effect=AssertionError("prefill must not launch decode per request"),
        ):
            transform_index_page_table_prefill_fast(
                page_table=page_table,
                topk_indices=topk_indices,
                extend_lens_cpu=extend_lens_cpu,
            )

    def test_prefill_uses_device_cu_seqlens(self):
        extend_lens_cpu = [2, 1]
        context_length = 4096
        page_table = self._make_page_table(len(extend_lens_cpu), context_length)
        topk_indices = self._make_topk(sum(extend_lens_cpu), context_length)
        cu_seqlens_q = torch.tensor([0, 2, 3], dtype=torch.int32, device=self.device)

        with patch.object(
            transform_index_module.torch,
            "tensor",
            side_effect=AssertionError("must reuse device-side metadata"),
        ):
            transform_index_page_table_prefill_fast(
                page_table=page_table,
                topk_indices=topk_indices,
                extend_lens_cpu=extend_lens_cpu,
                cu_seqlens_q=cu_seqlens_q,
            )

    def test_mixed_lengths_padding_and_empty_batch(self):
        self._check_case(
            [0, 3, 1, 0, 4],
            8192,
            page_table_is_expanded=False,
            topk_padding=5,
            output_padding=7,
        )
        self._check_case(
            [0, 0],
            16,
            page_table_is_expanded=False,
            output_padding=8,
        )

    def test_large_batch_size(self):
        self._check_case(
            [1] * 8192,
            4096,
            page_table_is_expanded=False,
        )

    def test_large_context_lengths(self):
        for context_length, page_table_is_expanded in (
            (640_000, True),
            (1_000_000, False),
        ):
            with self.subTest(
                context_length=context_length,
                page_table_is_expanded=page_table_is_expanded,
            ):
                self._check_case(
                    [2, 1],
                    context_length,
                    page_table_is_expanded=page_table_is_expanded,
                )

    def test_decode_fast_correctness_and_strides(self):
        self._check_decode_case(17, 8192, provide_result=True)
        self._check_decode_case(17, 8192, zero_row_stride=True)

    def test_decode_fast_extreme_shapes(self):
        self._check_decode_case(8192, 4096)
        self._check_decode_case(2, 1_000_000)

    # ------------------------------------------------------------------
    # DCP correctness tests
    # ------------------------------------------------------------------

    def _expected_dcp(
        self,
        page_table: torch.Tensor,
        topk_indices: torch.Tensor,
        extend_lens_cpu: list[int],
        output_num_tokens: int,
        page_table_is_expanded: bool,
        dcp_size: int,
        dcp_rank: int,
    ) -> torch.Tensor:
        """Ref implementation with DCP owner-filter + global-to-local map."""
        expected = self._expected(
            page_table,
            topk_indices,
            extend_lens_cpu,
            output_num_tokens,
            page_table_is_expanded,
        )
        if dcp_size <= 1:
            return expected
        real_num_tokens = sum(extend_lens_cpu)
        if real_num_tokens == 0:
            return expected
        slab = expected[:real_num_tokens]
        valid = slab >= 0
        owned = valid & (slab % dcp_size == dcp_rank)
        slab[valid & ~owned] = -1
        slab[owned] //= dcp_size
        return expected

    def _check_prefill_dcp_case(
        self,
        extend_lens_cpu: list[int],
        context_length: int,
        *,
        page_table_is_expanded: bool,
        dcp_size: int,
        dcp_rank: int,
        topk_padding: int = 0,
        output_padding: int = 0,
    ) -> None:
        real_num_tokens = sum(extend_lens_cpu)
        page_table_rows = (
            real_num_tokens if page_table_is_expanded else len(extend_lens_cpu)
        )
        topk_num_tokens = real_num_tokens + topk_padding
        output_num_tokens = topk_num_tokens + output_padding
        page_table = self._make_page_table(page_table_rows, context_length)
        topk_indices = self._make_topk(topk_num_tokens, context_length)
        expected = self._expected_dcp(
            page_table,
            topk_indices,
            extend_lens_cpu,
            output_num_tokens,
            page_table_is_expanded,
            dcp_size,
            dcp_rank,
        )

        actual = transform_index_page_table_prefill_fast(
            page_table=page_table,
            topk_indices=topk_indices,
            extend_lens_cpu=extend_lens_cpu,
            output_num_tokens=output_num_tokens,
            page_table_is_expanded=page_table_is_expanded,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prefill_dcp_across_widths_and_lengths(self):
        """Fast-vs-ref correctness across page-table widths, extend lengths,
        and DCP configurations.

        Page-table widths 64/127/128/4096 verify that do_not_specialize on
        page_table_stride_0 does not break correctness for any width.
        Extend lengths 1/2/4/7 cover BLOCK_Q 1/2/4/4.
        DCP sizes 1 (baseline) and 2 (owner-filter) with both ranks.
        """
        for context_length in (64, 127, 128, 4096):
            for extend_lens in ([1], [2], [4], [7], [1, 2, 4, 7]):
                for dcp_size, dcp_rank in (
                    (1, 0),
                    (2, 0),
                    (2, 1),
                ):
                    with self.subTest(
                        context_length=context_length,
                        extend_lens=extend_lens,
                        dcp_size=dcp_size,
                        dcp_rank=dcp_rank,
                    ):
                        self._check_prefill_dcp_case(
                            extend_lens,
                            context_length,
                            page_table_is_expanded=False,
                            dcp_size=dcp_size,
                            dcp_rank=dcp_rank,
                        )

    def test_prefill_dcp_expanded_mode(self):
        """DCP correctness with page_table_is_expanded=True (target_verify path)."""
        for context_length in (128, 4096):
            for extend_lens in ([4], [7], [2, 4]):
                for dcp_size, dcp_rank in ((1, 0), (2, 0), (2, 1)):
                    with self.subTest(
                        context_length=context_length,
                        extend_lens=extend_lens,
                        dcp_size=dcp_size,
                        dcp_rank=dcp_rank,
                    ):
                        self._check_prefill_dcp_case(
                            extend_lens,
                            context_length,
                            page_table_is_expanded=True,
                            dcp_size=dcp_size,
                            dcp_rank=dcp_rank,
                        )

    def test_prefill_dcp_with_padding(self):
        """DCP correctness with topk and output padding."""
        self._check_prefill_dcp_case(
            [4, 7],
            128,
            page_table_is_expanded=False,
            dcp_size=2,
            dcp_rank=0,
            topk_padding=3,
            output_padding=5,
        )
        self._check_prefill_dcp_case(
            [1, 2],
            4096,
            page_table_is_expanded=False,
            dcp_size=2,
            dcp_rank=1,
            topk_padding=2,
            output_padding=4,
        )

    def test_prefill_dcp_empty_batch(self):
        """DCP correctness with zero-length extend (no real tokens)."""
        self._check_prefill_dcp_case(
            [0, 0],
            128,
            page_table_is_expanded=False,
            dcp_size=2,
            dcp_rank=0,
           output_padding=4,
       )

    # ------------------------------------------------------------------
    # DCP decode correctness tests
    # ------------------------------------------------------------------

    def _expected_decode_dcp(
        self,
        page_table: torch.Tensor,
        topk_indices: torch.Tensor,
        dcp_size: int,
        dcp_rank: int,
        compact_dcp: bool,
    ) -> torch.Tensor:
        """Ref implementation of decode transform with DCP owner-filter,
        global-to-local mapping, and optional compaction."""
        expected = torch.empty(
            (topk_indices.shape[0], TOPK), dtype=torch.int32, device=self.device
        )
        torch.gather(
            page_table.to(torch.int32),
            dim=1,
            index=topk_indices.clamp(min=0),
            out=expected,
        )
        expected[topk_indices < 0] = -1

        if dcp_size > 1:
            valid = expected >= 0
            owned = valid & (expected % dcp_size == dcp_rank)
            expected[valid & ~owned] = -1
            expected[owned] //= dcp_size

        if compact_dcp and dcp_size > 1:
            result = torch.full_like(expected, -1)
            for i in range(expected.shape[0]):
                valid_vals = expected[i, expected[i] >= 0]
                result[i, : valid_vals.shape[0]] = valid_vals
            expected = result

        return expected

    def _check_decode_dcp_case(
        self,
        batch_size: int,
        context_length: int,
        *,
        dcp_size: int,
        dcp_rank: int,
        compact_dcp: bool,
        zero_row_stride: bool = False,
        provide_result: bool = False,
    ) -> None:
        if zero_row_stride:
            page_table = self._make_page_table(1, context_length).expand(
                batch_size, -1
            )
        else:
            page_table = self._make_page_table(batch_size, context_length)
        topk_indices = self._make_topk(batch_size, context_length)
        expected = self._expected_decode_dcp(
            page_table, topk_indices, dcp_size, dcp_rank, compact_dcp
        )
        result = torch.empty_like(expected) if provide_result else None
        actual = transform_index_page_table_decode_fast(
            page_table=page_table,
            topk_indices=topk_indices,
            result=result,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            compact_dcp=compact_dcp,
        )
        torch.cuda.synchronize()
        if result is not None:
            self.assertIs(actual, result)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_decode_dcp_owner_filter(self):
        """DCP decode owner filtering + local index mapping (no compaction).

        With dcp_size=2, rank 0 keeps even KV slots, rank 1 keeps odd.
        Global indices are mapped to local via // dcp_size.
        Non-owned and invalid slots become -1 in-place.
        """
        for dcp_rank in (0, 1):
            with self.subTest(dcp_rank=dcp_rank):
                self._check_decode_dcp_case(
                    17, 8192,
                    dcp_size=2, dcp_rank=dcp_rank, compact_dcp=False,
                )

    def test_decode_dcp_compact(self):
        """DCP decode with compact_dcp=True: valid entries compacted to front.

        After owner filtering, owned slots are moved to the front of each row
        via prefix-sum compaction. Remaining positions are filled with -1.
        """
        for dcp_rank in (0, 1):
            with self.subTest(dcp_rank=dcp_rank):
                self._check_decode_dcp_case(
                    17, 8192,
                    dcp_size=2, dcp_rank=dcp_rank, compact_dcp=True,
                )

    def test_decode_dcp_across_widths(self):
        """DCP decode correctness across page-table widths.

        Widths 64/127/128/4096 exercise different page_table_row_stride
        values (tl.constexpr in the kernel) to ensure correctness regardless
        of stride specialization.
        """
        for context_length in (64, 127, 128, 4096):
            for dcp_rank in (0, 1):
                for compact_dcp in (False, True):
                    with self.subTest(
                        context_length=context_length,
                        dcp_rank=dcp_rank,
                        compact_dcp=compact_dcp,
                    ):
                        self._check_decode_dcp_case(
                            8, context_length,
                            dcp_size=2, dcp_rank=dcp_rank,
                            compact_dcp=compact_dcp,
                        )

    def test_decode_dcp_strides_and_result(self):
        """DCP decode with expanded (zero-stride) page table and pre-allocated result."""
        for compact_dcp in (False, True):
            with self.subTest(compact_dcp=compact_dcp):
                self._check_decode_dcp_case(
                    17, 8192,
                    dcp_size=2, dcp_rank=0, compact_dcp=compact_dcp,
                    zero_row_stride=True, provide_result=True,
                )

    def test_decode_dcp_extreme_shapes(self):
        """DCP decode with large batch and large context length."""
        self._check_decode_dcp_case(
            8192, 4096,
            dcp_size=2, dcp_rank=0, compact_dcp=True,
        )
        self._check_decode_dcp_case(
            2, 1_000_000,
            dcp_size=2, dcp_rank=1, compact_dcp=True,
        )

    def test_decode_dcp_baseline_no_dcp(self):
        """compact_dcp=True with dcp_size=1 is a no-op.

        No owner filtering or compaction should occur; result matches the
        plain decode transform (invalid slots -> -1, valid slots unchanged).
        """
        self._check_decode_dcp_case(
            17, 8192,
            dcp_size=1, dcp_rank=0, compact_dcp=True,
        )

if __name__ == "__main__":
    unittest.main()
