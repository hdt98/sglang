import types
import unittest
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.mori import conn as mori_conn
from sglang.srt.disaggregation.mori.conn import (
    BatchTransferPlan,
    KVArgsRegisterInfo,
    MoriKVManager,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-a-test-1-gpu-small-amd")


class TestMoriDSATailTransfer(unittest.TestCase):
    def test_ring_segments_use_descriptor_relative_offsets(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_args = types.SimpleNamespace(
            prefill_start_layer=1,
            prefill_end_layer=2,
        )

        submitted = []

        def submit(src_desc, dst_desc, plan):
            submitted.append((src_desc, dst_desc, plan))
            return []

        manager._submit_batch_transfer_plan = submit
        src_descs = ["src-key", "src-score"]
        dst_descs = ["dst-key-0", "dst-key-1", "dst-score-0", "dst-score-1"]

        manager._send_dsa_tail_state(
            np.asarray([2, 9, 1, 0, 2, 10], dtype=np.int32),
            np.asarray([5, 4, 3, 0, 0, 10], dtype=np.int32),
            src_descs,
            dst_descs,
            [160, 160],
            [160, 160, 160, 160],
        )

        self.assertEqual(
            [(src, dst) for src, dst, _ in submitted],
            [("src-key", "dst-key-1"), ("src-score", "dst-score-1")],
        )
        for _, _, plan in submitted:
            self.assertEqual(plan.local_offsets, [464, 320])
            self.assertEqual(plan.remote_offsets, [864, 880])
            self.assertEqual(plan.sizes, [16, 32])

    def test_dflash_mla_transfer_limits_draft_to_live_suffix(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = True
        manager.kv_mem_descs = ["target", "draft-k", "draft-v"]
        manager.kv_args = types.SimpleNamespace(
            prefill_start_layer=0,
            prefill_end_layer=1,
            kv_item_lens=[64, 16, 16],
        )

        submitted = []

        def submit(src_desc, dst_desc, plan, **kwargs):
            submitted.append((src_desc, dst_desc, plan, kwargs.get("context")))
            return []

        manager._submit_batch_transfer_plan = submit
        peer = types.SimpleNamespace(
            dst_kv_mem_descs=["dst-target", "dst-draft-k", "dst-draft-v"],
            dst_kv_item_lens=[64, 16, 16],
        )
        src_indices = np.arange(10, 20, dtype=np.int32)
        dst_indices = np.arange(30, 40, dtype=np.int32)

        manager.send_kvcache(
            peer,
            src_indices,
            dst_indices,
            draft_suffix_pages=2,
        )

        self.assertEqual([entry[0] for entry in submitted], manager.kv_mem_descs)
        target_plan = submitted[0][2]
        self.assertEqual(target_plan.local_offsets, [10 * 64])
        self.assertEqual(target_plan.remote_offsets, [30 * 64])
        self.assertEqual(target_plan.sizes, [10 * 64])
        for _, _, plan, context in submitted[1:]:
            self.assertEqual(plan.local_offsets, [18 * 16])
            self.assertEqual(plan.remote_offsets, [38 * 16])
            self.assertEqual(plan.sizes, [2 * 16])
            self.assertIn("Mori MLA KV descriptor", context)

    def test_dflash_nonfinal_chunk_skips_draft_descriptors(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = True
        manager.kv_mem_descs = ["target", "draft-k", "draft-v"]
        manager.kv_args = types.SimpleNamespace(
            prefill_start_layer=0,
            prefill_end_layer=1,
            kv_item_lens=[64, 16, 16],
        )
        submitted = []
        manager._submit_batch_transfer_plan = lambda src, dst, plan, **kwargs: (
            submitted.append(src) or []
        )
        peer = types.SimpleNamespace(
            dst_kv_mem_descs=["dst-target", "dst-draft-k", "dst-draft-v"],
            dst_kv_item_lens=[64, 16, 16],
        )

        manager.send_kvcache(
            peer,
            np.arange(4, dtype=np.int32),
            np.arange(4, dtype=np.int32),
        )

        self.assertEqual(submitted, ["target"])

    def test_transfer_plan_rejects_registered_memory_overflow(self):
        plan = BatchTransferPlan(local_offsets=[32], remote_offsets=[48], sizes=[32])
        with self.assertRaisesRegex(ValueError, "exceeds registered memory"):
            MoriKVManager._validate_batch_transfer_plan(
                types.SimpleNamespace(size=63),
                types.SimpleNamespace(size=128),
                plan,
                context="test descriptor",
            )

    def test_mori_registration_carries_per_descriptor_item_lengths(self):
        payload = [
            b"None",
            b"127.0.0.1",
            b"1234",
            b"engine",
            b"kv-descs",
            b"",
            b"",
            b"0",
            b"4",
            b"2",
            b"64",
            b"",
            b"",
            np.asarray([64, 16, 16], dtype=np.uint64).tobytes(),
        ]
        with patch.object(mori_conn.EngineDesc, "unpack", return_value="engine"), patch.object(
            mori_conn,
            "_unpack_mem_desc_list",
            side_effect=[["target", "draft-k", "draft-v"], []],
        ):
            info = KVArgsRegisterInfo.from_zmq(payload)

        self.assertEqual(info.dst_kv_item_lens, [64, 16, 16])

    def test_compact_dflash_window_includes_page_alignment_slack(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.server_args = types.SimpleNamespace(
            speculative_algorithm="DFLASH",
            speculative_draft_window_size=2048,
        )
        manager.kv_args = types.SimpleNamespace(page_size=64)

        self.assertEqual(manager._resolve_draft_suffix_pages(), 33)


if __name__ == "__main__":
    unittest.main()
