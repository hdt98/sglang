"""The decode metadata gate must read all bootstrap rooms in one D2H, not one
sync per queued request.

Bug mechanism: the gate runs on every decode-scheduler poll, and it used to call
`.item()` on the device-side bootstrap_room per queued request. Each `.item()`
is a full device sync, so a poll with N queued transfers cost N syncs; the C8
decode profile showed ~270 device syncs per decode step with the GPU idle for
roughly 40% of decode wall time. The fix batches the reads into a single copy.
These cases pin the gate's downgrading semantics, which must survive the
batching: only Success polls whose metadata has not landed (room 0) are
downgraded to Transferring, fake transfers are exempt, and already-landed or
non-Success polls are left untouched.
"""

import unittest

import torch

from sglang.srt.disaggregation.utils import KVPoll, _apply_metadata_gate


def _fake_transfer_req():
    # DecodeRequest-shaped wrapper: the gate reads `.req.bootstrap_host`.
    from types import SimpleNamespace

    return SimpleNamespace(req=SimpleNamespace(bootstrap_host="2.2.2.2"))


def _real_transfer_req():
    from types import SimpleNamespace

    return SimpleNamespace(req=SimpleNamespace(bootstrap_host="10.0.0.1:1"))


class _FakeMetadataBuffers:
    """bootstrap_room behaves like the device tensor the real buffers expose."""

    def __init__(self, rooms):
        # rooms[i, 0] is the bootstrap room for metadata slot i.
        self.bootstrap_room = torch.tensor(
            [[r] for r in rooms], dtype=torch.int64
        )


class TestApplyMetadataGateBatchedD2H(unittest.TestCase):
    def test_zero_room_downgrades_success_poll(self):
        reqs = [_real_transfer_req(), _real_transfer_req()]
        bufs = _FakeMetadataBuffers(rooms=[7, 0])
        polls = [int(KVPoll.Success), int(KVPoll.Success)]
        _apply_metadata_gate(polls, reqs, bufs)
        self.assertEqual(polls, [int(KVPoll.Success), int(KVPoll.Transferring)])

    def test_landed_room_keeps_success(self):
        reqs = [_real_transfer_req(), _real_transfer_req()]
        bufs = _FakeMetadataBuffers(rooms=[3, 9])
        polls = [int(KVPoll.Success), int(KVPoll.Success)]
        _apply_metadata_gate(polls, reqs, bufs)
        self.assertEqual(polls, [int(KVPoll.Success), int(KVPoll.Success)])

    def test_fake_transfer_exempt_from_gate(self):
        reqs = [_fake_transfer_req()]
        # Room 0 would be downgraded for a real request; fake transfers are
        # never gated because their metadata buffer is not populated.
        bufs = _FakeMetadataBuffers(rooms=[0])
        polls = [int(KVPoll.Success)]
        _apply_metadata_gate(polls, reqs, bufs)
        self.assertEqual(polls, [int(KVPoll.Success)])

    def test_non_success_polls_untouched(self):
        reqs = [_real_transfer_req(), _real_transfer_req()]
        bufs = _FakeMetadataBuffers(rooms=[0, 0])
        polls = [int(KVPoll.Transferring), int(KVPoll.Failed)]
        _apply_metadata_gate(polls, reqs, bufs)
        self.assertEqual(polls, [int(KVPoll.Transferring), int(KVPoll.Failed)])

    def test_batch_read_preserves_per_slot_rooms(self):
        # Advanced indexing must map each poll back to its own metadata slot,
        # not to slot 0 or a shifted index: a wrong mapping would downgrade the
        # wrong request when only one of several concurrent transfers is
        # incomplete.
        reqs = [_real_transfer_req() for _ in range(4)]
        bufs = _FakeMetadataBuffers(rooms=[11, 0, 13, 14])
        polls = [int(KVPoll.Success)] * 4
        _apply_metadata_gate(polls, reqs, bufs)
        self.assertEqual(
            polls,
            [
                int(KVPoll.Success),
                int(KVPoll.Transferring),
                int(KVPoll.Success),
                int(KVPoll.Success),
            ],
        )

    def test_no_success_polls_skips_device_read(self):
        # With nothing to gate, the batched path must not touch the device
        # buffer at all — a broken short-circuit that still indexes the buffer
        # would reintroduce a per-poll sync even when the queue is empty.
        reqs = [_real_transfer_req()]
        bufs = _FakeMetadataBuffers(rooms=[5])
        polls = [int(KVPoll.Transferring)]
        _apply_metadata_gate(polls, reqs, bufs)
        self.assertEqual(polls, [int(KVPoll.Transferring)])


if __name__ == "__main__":
    unittest.main()
