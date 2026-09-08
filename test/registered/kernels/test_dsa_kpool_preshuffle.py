"""GPU regressions for the DSA kpool preshuffled cache layout."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa import aiter_paged_mqa_logits
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention.dsa import kpool_fp8_index
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    INDEX_HEAD_DIM,
    gather_index_k_scale_prefix_into,
    kpool_softmax_rotate_write_cache,
)
from sglang.srt.layers.attention.dsa.utils import (
    aiter_can_use_preshuffle_paged_mqa,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
class TestDsaKpoolPreshuffle(CustomTestCase):
    POOL_SIZE = 4
    PAGE_SIZE = 64
    SLOTS_PER_PAGE = 64
    NUM_INDEX_HEADS = 32

    def _pool(self) -> SimpleNamespace:
        return SimpleNamespace(
            page_size=self.PAGE_SIZE,
            index_head_dim=INDEX_HEAD_DIM,
            slots_per_page=self.SLOTS_PER_PAGE,
            index_kpool=self.POOL_SIZE,
        )

    def _empty_cache(self, num_pages: int = 1) -> torch.Tensor:
        page_nbytes = self.SLOTS_PER_PAGE * (INDEX_HEAD_DIM + 4)
        return torch.zeros(
            (num_pages, page_nbytes), dtype=torch.uint8, device="cuda"
        )

    def test_preshuffled_cache_round_trip_and_raw_layout(self):
        torch.manual_seed(43)
        pool = self._pool()
        num_slots = self.SLOTS_PER_PAGE
        tile = 16
        num_col_tiles = INDEX_HEAD_DIM // tile

        slot_k = torch.randn(
            num_slots,
            self.POOL_SIZE,
            INDEX_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        slot_score = torch.zeros_like(slot_k)
        ape = torch.zeros(
            self.POOL_SIZE,
            INDEX_HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
        loc = torch.arange(num_slots, dtype=torch.int64, device="cuda")
        cache = self._empty_cache()

        with patch.object(kpool_fp8_index, "_preshuffle_tile", return_value=tile):
            compressed_k, compressed_scale = kpool_softmax_rotate_write_cache(
                pool=pool,
                buf=cache,
                slot_k=slot_k,
                slot_score=slot_score,
                ape=ape,
                loc=loc,
                return_compressed=True,
            )

            gathered_k = torch.empty(
                (num_slots, INDEX_HEAD_DIM), dtype=torch.uint8, device="cuda"
            )
            gathered_scale = torch.empty(
                (num_slots,), dtype=torch.float32, device="cuda"
            )
            gather_index_k_scale_prefix_into(
                pool=pool,
                buf=cache,
                page_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
                seq_len=num_slots,
                k_out=gathered_k,
                scale_out=gathered_scale,
            )

        compressed_k_u8 = compressed_k.view(torch.uint8)
        torch.testing.assert_close(gathered_k, compressed_k_u8, atol=0, rtol=0)
        torch.testing.assert_close(gathered_scale, compressed_scale, atol=0, rtol=0)

        raw_k = cache[0, : num_slots * INDEX_HEAD_DIM].view(
            num_slots // tile, num_col_tiles, tile, tile
        )
        expected_raw_k = (
            compressed_k_u8.view(num_slots // tile, tile, num_col_tiles, tile)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        torch.testing.assert_close(raw_k, expected_raw_k, atol=0, rtol=0)

    @unittest.skipUnless(
        is_hip() and aiter_can_use_preshuffle_paged_mqa(),
        "requires the ROCm AITER preshuffle paged-MQA path",
    )
    def test_preshuffled_paged_mqa_matches_gathered_mqa(self):
        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

        torch.manual_seed(44)
        pool = self._pool()
        tile = 16
        num_token_tiles = self.SLOTS_PER_PAGE // tile
        num_col_tiles = INDEX_HEAD_DIM // tile

        for batch_size in (1, 8):
            with self.subTest(batch_size=batch_size):
                cache = self._empty_cache(batch_size)
                k = (
                    torch.randn(
                        batch_size,
                        self.SLOTS_PER_PAGE,
                        INDEX_HEAD_DIM,
                        device="cuda",
                    )
                    * 2
                ).to(fp8_dtype)
                k_scale = (
                    torch.rand(batch_size, self.SLOTS_PER_PAGE, device="cuda") + 0.25
                )

                preshuffled = (
                    k.view(
                        batch_size,
                        num_token_tiles,
                        tile,
                        num_col_tiles,
                        tile,
                    )
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                    .view(batch_size, -1)
                    .view(torch.uint8)
                )
                k_region_bytes = self.SLOTS_PER_PAGE * INDEX_HEAD_DIM
                cache[:, :k_region_bytes].copy_(preshuffled)
                cache[:, k_region_bytes:].view(torch.float32).copy_(k_scale)

                q = (
                    torch.randn(
                        batch_size,
                        self.NUM_INDEX_HEADS,
                        INDEX_HEAD_DIM,
                        device="cuda",
                    )
                    * 2
                ).to(fp8_dtype)
                weights = torch.randn(
                    batch_size,
                    self.NUM_INDEX_HEADS,
                    dtype=torch.float32,
                    device="cuda",
                )
                if batch_size == 1:
                    seq_lens = torch.tensor([53], dtype=torch.int32, device="cuda")
                else:
                    seq_lens = torch.tensor(
                        [1, 7, 16, 31, 32, 47, 63, 64],
                        dtype=torch.int32,
                        device="cuda",
                    )
                block_tables = torch.arange(
                    batch_size, dtype=torch.int32, device="cuda"
                ).view(batch_size, 1)

                paged_logits = aiter_paged_mqa_logits(
                    q,
                    cache.view(
                        batch_size,
                        self.SLOTS_PER_PAGE,
                        1,
                        INDEX_HEAD_DIM + 4,
                    ),
                    weights,
                    seq_lens,
                    block_tables,
                    self.SLOTS_PER_PAGE,
                    preshuffle=True,
                    kv_block_size=self.SLOTS_PER_PAGE,
                )

                gathered_k = []
                gathered_scale = []
                for batch_idx, seq_len in enumerate(seq_lens.tolist()):
                    k_out = torch.empty(
                        (seq_len, INDEX_HEAD_DIM), dtype=torch.uint8, device="cuda"
                    )
                    scale_out = torch.empty(
                        (seq_len,), dtype=torch.float32, device="cuda"
                    )
                    gather_index_k_scale_prefix_into(
                        pool=pool,
                        buf=cache,
                        page_indices=torch.tensor(
                            [batch_idx], dtype=torch.int32, device="cuda"
                        ),
                        seq_len=seq_len,
                        k_out=k_out,
                        scale_out=scale_out,
                    )
                    gathered_k.append(k_out.view(fp8_dtype))
                    gathered_scale.append(scale_out)

                flat_k = torch.cat(gathered_k)
                flat_scale = torch.cat(gathered_scale)
                ends = seq_lens.cumsum(0)
                starts = ends - seq_lens
                gathered_logits = fp8_mqa_logits(
                    q,
                    flat_k,
                    flat_scale,
                    weights,
                    starts,
                    ends,
                    clean_logits=True,
                )

                for batch_idx, seq_len in enumerate(seq_lens.tolist()):
                    start = starts[batch_idx].item()
                    torch.testing.assert_close(
                        paged_logits[batch_idx, :seq_len],
                        gathered_logits[batch_idx, start : start + seq_len],
                        atol=2e-2,
                        rtol=2e-2,
                    )


if __name__ == "__main__":
    unittest.main()
