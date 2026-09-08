import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeReqToTokenPool,
    HybridMambaDecodeReqToTokenPool,
)
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _init_decode_pool(pool):
    DecodeReqToTokenPool.__init__(
        pool,
        size=1,
        max_context_len=4,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=1,
    )
    return pool


def test_decode_pool_reports_physical_capacity():
    pool = _init_decode_pool(DecodeReqToTokenPool.__new__(DecodeReqToTokenPool))

    assert pool.schedulable_token_capacity(17) == 17


def test_decode_pool_supports_noop_aux_cache_contract():
    pool = _init_decode_pool(DecodeReqToTokenPool.__new__(DecodeReqToTokenPool))
    req_to_token = pool.req_to_token.clone()

    pool.alloc_aux_to_lengths(
        req_pool_indices_cpu=torch.tensor([1]),
        target_seq_lens_cpu=torch.tensor([3]),
    )
    pool.reset_aux_cache_allocator()

    assert torch.equal(pool.req_to_token, req_to_token)


def test_hybrid_decode_pool_initializes_aux_cache_contract():
    pool = _init_decode_pool(
        HybridMambaDecodeReqToTokenPool.__new__(HybridMambaDecodeReqToTokenPool)
    )

    assert pool.schedulable_token_capacity(17) == 17


@pytest.mark.parametrize("free_states", [0, 1, 3])
@pytest.mark.parametrize("track_slots", [1, 2])
def test_decode_prealloc_evicts_cached_mamba_before_allocating_track_buffers(
    free_states, track_slots
):
    pool = _init_decode_pool(
        HybridMambaDecodeReqToTokenPool.__new__(HybridMambaDecodeReqToTokenPool)
    )
    pool.mamba_allocator = MambaSlotAllocator(size=3, device="cpu")
    cached_states = pool.mamba_allocator.alloc(3 - free_states)
    pool.mamba_pool = SimpleNamespace(
        size=3,
        replayssm_write_pos=None,
        replayssm_cache_base=None,
    )
    pool.enable_mamba_extra_buffer = True
    pool.enable_mamba_extra_buffer_lazy = False
    pool.mamba_ping_pong_track_buffer_size = track_slots
    pool.req_index_to_mamba_index_mapping = torch.zeros(3, dtype=torch.int32)
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
        (3, track_slots), dtype=torch.int64
    )

    cache = MagicMock()
    cache.supports_mamba.return_value = True

    def evict(params):
        assert params == EvictParams(num_tokens=0, mamba_num=3 - free_states)
        pool.mamba_allocator.free(cached_states)

    cache.evict_for_alloc.side_effect = evict
    req = SimpleNamespace(
        rid="pd-mamba-pressure",
        origin_input_ids=[11, 12, 13],
        output_ids=[],
        kv=SimpleNamespace(
            holds_kv=False,
            holds_mamba=False,
            req_pool_idx=None,
            mamba_ping_pong_track_buffer=None,
        ),
        set_extend_range=MagicMock(),
    )
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.req_to_token_pool = pool
    queue.tree_cache = cache
    queue.scheduler = SimpleNamespace(
        enable_hisparse=False,
        server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=True),
    )
    queue.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=1, alloc=lambda size: torch.arange(size)
    )
    queue._radix_full_available = lambda: 16
    queue._uses_swa_tail_prealloc = lambda: False
    queue._swa_tail_len = lambda _: 0

    kv_indices = queue._pre_alloc(req)

    assert torch.equal(kv_indices, torch.arange(3))
    assert req.kv.mamba_ping_pong_track_buffer.numel() == track_slots
    assert pool.mamba_allocator.available_size() == 2 - track_slots
    req.set_extend_range.assert_called_once_with(0, 3)
    if free_states < 3:
        cache.evict_for_alloc.assert_called_once()
    else:
        cache.evict_for_alloc.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
