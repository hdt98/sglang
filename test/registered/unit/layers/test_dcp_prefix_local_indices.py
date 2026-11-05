"""DCP prefix gather with prefixes that are not multiples of dcp_size."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.dcp import comm, layout, planner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DCP = 8


def _fill_prefix_indices(req_to_token, prefix_lens):
    def fill(_grid):
        def run(r2t, pool_idx, starts, lens, cu_lens, out, _stride):
            for i, row in enumerate(pool_idx.tolist()):
                n = int(lens[i])
                out[int(cu_lens[i]) : int(cu_lens[i]) + n] = r2t[row, :n]

        return run

    return MagicMock(__getitem__=lambda _self, grid: fill(grid))


def _local_prefix_ids(rank, prefix_lens, req_to_token):
    parallel = SimpleNamespace(dcp_enabled=True, dcp_size=_DCP, dcp_rank=rank)
    backend = SimpleNamespace(
        runner_ids=SimpleNamespace(translate_dcp_read_ids=lambda x: x)
    )
    bs = len(prefix_lens)
    lens = torch.tensor(prefix_lens, dtype=torch.int32)
    with patch.object(planner, "get_parallel", return_value=parallel), patch.object(
        layout, "get_parallel", return_value=parallel
    ), patch.object(
        planner, "get_device", return_value=SimpleNamespace(device="cpu")
    ), patch.object(
        planner, "get_attn_backend", return_value=backend
    ), patch.object(
        planner, "create_dcp_kv_indices", MagicMock()
    ):
        metadata = planner.prepare_decode_context_parallel_metadata(
            seq_lens=lens + 1,
            extend_prefix_lens=lens,
            extend_prefix_lens_cpu=list(prefix_lens),
            extend_seq_lens=torch.ones(bs, dtype=torch.int32),
            req_pool_indices=torch.arange(bs),
            req_to_token=req_to_token,
            seq_lens_sum=sum(prefix_lens) + bs,
            kv_buffer_shape=torch.Size([1, 1, 2]),
            kv_cache_dtype=torch.float32,
            kv_cache_device="cpu",
            create_chunked_prefix_cache_kv_indices_fn=_fill_prefix_indices(
                req_to_token, prefix_lens
            ),
        )
    return metadata.dcp_local_prefix_kv_indices


def _gather_all_ranks(prefix_lens, req_to_token):
    """Run every rank's planner + all_gather_kv_cache_for_dcp; return rank 0's view."""
    lens_cpu = torch.tensor(prefix_lens)
    local = [
        _local_prefix_ids(rank, prefix_lens, req_to_token).float().view(-1, 1, 1)
        for rank in range(_DCP)
    ]
    padded = []

    def record(kv):
        padded.append(kv)
        return kv.new_zeros((kv.shape[0] * _DCP, *kv.shape[1:]))

    for rank in range(_DCP):
        parallel = SimpleNamespace(dcp_enabled=True, dcp_size=_DCP, dcp_rank=rank)
        with patch.object(comm, "get_parallel", return_value=parallel), patch.object(
            comm, "_all_gather_dcp_kv_cache", record
        ):
            comm.all_gather_kv_cache_for_dcp(local[rank], local[rank], lens_cpu)
    stacked = torch.stack(padded).transpose(0, 1).reshape(-1, 1, 2)
    parallel = SimpleNamespace(dcp_enabled=True, dcp_size=_DCP, dcp_rank=0)
    with patch.object(comm, "get_parallel", return_value=parallel), patch.object(
        comm, "_all_gather_dcp_kv_cache", lambda _kv: stacked
    ):
        out = comm.all_gather_kv_cache_for_dcp(local[0], local[0], lens_cpu)
    return out[:, 0, 0].long()


def _req_to_token(prefix_lens):
    # Each row starts on its own dcp_size-wide page, as the widened allocator lays it out.
    width = max(prefix_lens) + _DCP
    r2t = torch.zeros((len(prefix_lens), width), dtype=torch.int32)
    base = 0
    for i, n in enumerate(prefix_lens):
        r2t[i, :n] = torch.arange(base, base + n, dtype=torch.int32)
        base += (n + _DCP - 1) // _DCP * _DCP + _DCP
    return r2t


class TestDcpPrefixLocalIndices(CustomTestCase):
    def _assert_gather(self, prefix_lens):
        r2t = _req_to_token(prefix_lens)
        expected = torch.cat([r2t[i, :n].long() for i, n in enumerate(prefix_lens)])
        for rank in range(_DCP):
            ids = _local_prefix_ids(rank, prefix_lens, r2t)
            self.assertTrue(bool((ids % _DCP == rank).all()), (rank, prefix_lens))
        self.assertTrue(torch.equal(_gather_all_ranks(prefix_lens, r2t), expected))

    def test_mixed_pass_from_c48_crash(self):
        # A 16,381-token chunk continuation (prefix 15,743) mixed with three
        # running rows (prefix = seq_len - 1); the global stride crashed rank 0
        # with 54 expected vs 52 local rows.
        self._assert_gather([15743, 418, 418, 429])

    def test_split_head_unaligned_chunk_then_cached_row(self):
        self._assert_gather([15743, 64])

    def test_aligned_prefixes(self):
        self._assert_gather([64, 128, 0, 192])


if __name__ == "__main__":
    unittest.main()
