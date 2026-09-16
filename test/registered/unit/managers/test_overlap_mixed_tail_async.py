"""Unit tests for the mixed DSPARK spec-tail resolve async contract.

resolve_mixed_spec_tails rebuilds a mixed batch's decode-tail seq_lens /
out_cache_loc / CPU mirror at forward entry. The overlap caller runs on the
forward stream after forward_stream.wait_stream(schedule_stream), and the
publish scatter + record were enqueued on that same stream, so the GPU
gathers are stream-ordered after the publish without any fence: the legacy
HIP host-blocking publish_ready.synchronize() must never be called, while the
host mirror values must still be produced by the D2H paths
(bootstrap .cpu() on HIP, pinned private-stream mirror on CUDA) for the
scheduler.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.overlap_utils import FutureMap
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_REQ_POOL = 8
# Mixed batch shape: 2 extend rows followed by 2 decode tails.
_BASE_SEQ_LENS = [50, 60]
_TAIL_SEQ_LENS = [10, 20]
_SLOTS = [3, 7]
_FRESH = [11, 21]  # committed lengths published by the in-flight verify


class _RecordingEvent:
    """Stands in for the publish device event; records host-side calls."""

    def __init__(self):
        self.synchronize_calls = 0
        self.wait_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1

    def wait(self):
        self.wait_calls += 1


class _RecordingStream:
    """Stands in for the private D2H stream of the pinned-mirror path."""

    def __init__(self):
        self.wait_event_calls = []
        self.synchronize_calls = 0

    def wait_event(self, event):
        self.wait_event_calls.append(event)

    def synchronize(self):
        self.synchronize_calls += 1


@contextmanager
def _null_stream_ctx():
    yield


def _make_future_map():
    fm = object.__new__(FutureMap)
    fm.device = "cpu"
    fm.req_pool_size = _REQ_POOL
    fm.new_seq_lens_buf = torch.full((_REQ_POOL,), -1, dtype=torch.int64)
    for slot, fresh in zip(_SLOTS, _FRESH):
        fm.new_seq_lens_buf[slot] = fresh
    fm.req_to_token = torch.arange(
        _REQ_POOL * 32, dtype=torch.int64
    ).reshape(_REQ_POOL, 32)
    fm.publish_ready = _RecordingEvent()
    fm.fwd_prepare_d2h_stream = None
    fm.new_seq_lens_cpu_pinned = None
    return fm


def _make_batch():
    return SimpleNamespace(
        mix_running_indices=torch.tensor(_SLOTS, dtype=torch.int64),
        mix_running_indices_cpu=torch.tensor(_SLOTS, dtype=torch.int64),
        seq_lens=torch.tensor(_BASE_SEQ_LENS + _TAIL_SEQ_LENS, dtype=torch.int64),
        out_cache_loc=torch.arange(100, 104, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(_BASE_SEQ_LENS + _TAIL_SEQ_LENS, dtype=torch.int64),
        prefix_lens=[49, 59, 9, 19],
    )


def _expected():
    # req_to_token is arange(8, 32): row slot, col fresh.
    fresh_slots = [
        int(fm_idx * 32 + fresh) for fm_idx, fresh in zip(_SLOTS, _FRESH)
    ]
    return SimpleNamespace(
        seq_lens=_BASE_SEQ_LENS + [f + 1 for f in _FRESH],
        out_cache_loc_tails=fresh_slots,
        seq_lens_sum=sum(_BASE_SEQ_LENS) + sum(f + 1 for f in _FRESH),
        prefix_lens=[49, 59] + _FRESH,
    )


class TestMixedTailAsyncContract(CustomTestCase):
    def test_hip_mixed_tail_resolves_without_host_event_sync(self):
        # The legacy HIP path blocked the host on the publish event before the
        # gathers; the gathers are same-stream ordered, so no host sync (and
        # no stream-ordered fence either) may be enqueued on HIP.
        fm = _make_future_map()
        batch = _make_batch()
        exp = _expected()
        with patch("sglang.srt.managers.overlap_utils._is_hip", True):
            fm.resolve_mixed_spec_tails(batch)

        self.assertEqual(fm.publish_ready.synchronize_calls, 0)
        self.assertEqual(fm.publish_ready.wait_calls, 0)
        self.assertEqual(batch.seq_lens.tolist(), exp.seq_lens)
        self.assertEqual(batch.out_cache_loc[-2:].tolist(), exp.out_cache_loc_tails)
        # Bootstrap .cpu() mirror still feeds the scheduler's host data.
        self.assertEqual(batch.seq_lens_cpu.tolist(), exp.seq_lens)
        self.assertEqual(batch.seq_lens_sum, exp.seq_lens_sum)
        self.assertEqual(batch.prefix_lens, exp.prefix_lens)

    def test_non_hip_mixed_tail_keeps_stream_ordered_wait(self):
        # Non-HIP platforms keep the cheap stream-ordered wait; the blocking
        # synchronize must not come back on any platform.
        fm = _make_future_map()
        batch = _make_batch()
        exp = _expected()
        with patch("sglang.srt.managers.overlap_utils._is_hip", False):
            fm.resolve_mixed_spec_tails(batch)

        self.assertEqual(fm.publish_ready.synchronize_calls, 0)
        self.assertEqual(fm.publish_ready.wait_calls, 1)
        self.assertEqual(batch.seq_lens.tolist(), exp.seq_lens)
        self.assertEqual(batch.seq_lens_cpu.tolist(), exp.seq_lens)
        self.assertEqual(batch.seq_lens_sum, exp.seq_lens_sum)

    def test_pinned_mirror_path_preserves_private_stream_cpu_data(self):
        # CUDA-style graph-era mechanism: the host mirror is produced by the
        # private D2H stream gated on the publish event, and the pinned rows
        # are gathered with mix_running_indices_cpu.
        fm = _make_future_map()
        fm.fwd_prepare_d2h_stream = _RecordingStream()
        fm.new_seq_lens_cpu_pinned = torch.empty(_REQ_POOL, dtype=torch.int64)
        batch = _make_batch()
        exp = _expected()
        fake_module = SimpleNamespace(stream=lambda s: _null_stream_ctx())
        with patch("sglang.srt.managers.overlap_utils._is_hip", False), patch(
            "torch.get_device_module", return_value=fake_module
        ):
            fm.resolve_mixed_spec_tails(batch)

        self.assertEqual(fm.publish_ready.synchronize_calls, 0)
        self.assertEqual(fm.publish_ready.wait_calls, 1)
        self.assertEqual(
            fm.fwd_prepare_d2h_stream.wait_event_calls, [fm.publish_ready]
        )
        self.assertEqual(fm.fwd_prepare_d2h_stream.synchronize_calls, 1)
        self.assertEqual(batch.seq_lens_cpu.tolist(), exp.seq_lens)
        self.assertEqual(batch.seq_lens_sum, exp.seq_lens_sum)
        self.assertEqual(batch.prefix_lens, exp.prefix_lens)

    def test_eager_rebuild_rebinds_without_mutating_schedule_staging(self):
        # The tail rebuild is an eager host-side clone+scatter that rebinds
        # the batch fields; the schedule-side staging tensors must never be
        # mutated in place (isolation snapshot/restore and graph replay read
        # the originals).
        fm = _make_future_map()
        batch = _make_batch()
        orig_seq_lens = batch.seq_lens
        orig_out_cache_loc = batch.out_cache_loc
        with patch("sglang.srt.managers.overlap_utils._is_hip", True):
            fm.resolve_mixed_spec_tails(batch)

        self.assertIsNot(batch.seq_lens, orig_seq_lens)
        self.assertIsNot(batch.out_cache_loc, orig_out_cache_loc)
        self.assertEqual(orig_seq_lens.tolist(), _BASE_SEQ_LENS + _TAIL_SEQ_LENS)
        self.assertEqual(orig_out_cache_loc.tolist(), [100, 101, 102, 103])

    def test_empty_tail_noop(self):
        fm = _make_future_map()
        batch = _make_batch()
        batch.mix_running_indices = torch.tensor([], dtype=torch.int64)
        batch.mix_running_indices_cpu = torch.tensor([], dtype=torch.int64)
        orig_seq_lens = batch.seq_lens
        with patch("sglang.srt.managers.overlap_utils._is_hip", True):
            fm.resolve_mixed_spec_tails(batch)

        self.assertEqual(fm.publish_ready.synchronize_calls, 0)
        self.assertEqual(fm.publish_ready.wait_calls, 0)
        self.assertIs(batch.seq_lens, orig_seq_lens)
        self.assertEqual(batch.prefix_lens, [49, 59, 9, 19])


if __name__ == "__main__":
    unittest.main()
