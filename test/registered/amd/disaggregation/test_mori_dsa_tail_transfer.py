import types
import unittest

import numpy as np

from sglang.srt.disaggregation.mori.conn import MoriKVManager
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


if __name__ == "__main__":
    unittest.main()
