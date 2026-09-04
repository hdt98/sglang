import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import (
    ScheduleBatch,
    set_mamba_track_indices_from_reqs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_batch():
    live_req = SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=1))
    freed_req = SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=None))
    batch = ScheduleBatch(reqs=[live_req, freed_req])
    batch.req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
    batch.req_to_token_pool = SimpleNamespace(
        req_index_to_mamba_ping_pong_track_buffer_mapping=torch.tensor(
            [[7, 8], [9, 10]], dtype=torch.int64
        )
    )
    return batch


_real_tensor = torch.tensor


def _unpinned_tensor(*args, **kwargs):
    # Keep the CPU unit test on the pageable constructor; pinned host memory is
    # a GPU-launch optimization and is not available on every test platform.
    kwargs["pin_memory"] = False
    return _real_tensor(*args, **kwargs)


class TestMambaTrackIndices(unittest.TestCase):
    def test_freed_overlap_row_is_a_no_op_destination(self):
        batch = _make_batch()

        with patch.object(torch, "tensor", side_effect=_unpinned_tensor):
            set_mamba_track_indices_from_reqs(batch)

        self.assertEqual(batch.mamba_track_buffer_indices, [1, 0])
        self.assertEqual(batch.mamba_track_indices.tolist(), [8, -1])

    def test_lazy_plan_cannot_resurrect_a_freed_row(self):
        batch = _make_batch()

        with patch.object(torch, "tensor", side_effect=_unpinned_tensor):
            set_mamba_track_indices_from_reqs(batch, track_positions=[1, 0])

        self.assertEqual(batch.mamba_track_buffer_indices, [1, 0])
        self.assertEqual(batch.mamba_track_indices.tolist(), [8, -1])


if __name__ == "__main__":
    unittest.main()
