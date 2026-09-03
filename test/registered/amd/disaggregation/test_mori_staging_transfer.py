import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.configs import model_config

with patch.object(
    model_config,
    "get_dsa_mtp_topk_width",
    lambda config: 1,
    create=True,
):
    from sglang.srt.disaggregation.mori import conn as mori_conn
    from sglang.srt.disaggregation.mori.conn import (
        KVArgsRegisterInfo,
        MoriKVManager,
    )

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, suite="stage-a-test-1-gpu-small-amd")


class TestMoriStagingTransfer(unittest.TestCase):
    def test_staged_prefill_write_uses_target_descriptors_and_layer_offsets(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_args = types.SimpleNamespace(
            num_target_kv_entries=2,
            kv_item_lens=[100, 200, 30],
        )
        manager.kv_mem_descs = ["target-0", "target-1", "draft"]
        submitted = []
        manager._submit_batch_transfer_plan = (
            lambda src, dst, plan, **kwargs: submitted.append((src, dst, plan)) or []
        )
        peer = types.SimpleNamespace(
            staging_mem_desc="decode-staging",
            dst_num_target_kv_entries=2,
            dst_kv_item_lens=[100, 200, 30],
        )

        manager._send_staged_kvcache(
            peer,
            np.asarray([10, 11], dtype=np.int32),
            staging_offset=1234,
        )

        self.assertEqual(len(submitted), 2)
        self.assertEqual(submitted[0][0], "target-0")
        self.assertEqual(submitted[0][1], "decode-staging")
        self.assertEqual(submitted[0][2].local_offsets, [1000])
        self.assertEqual(submitted[0][2].remote_offsets, [1234])
        self.assertEqual(submitted[0][2].sizes, [200])

        self.assertEqual(submitted[1][0], "target-1")
        self.assertEqual(submitted[1][2].local_offsets, [2000])
        self.assertEqual(submitted[1][2].remote_offsets, [1234 + 100])
        self.assertEqual(submitted[1][2].sizes, [400])

    def test_staged_prefill_write_rejects_target_geometry_mismatch(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_args = types.SimpleNamespace(
            num_target_kv_entries=2,
            kv_item_lens=[100, 200],
        )
        manager.kv_mem_descs = ["target-0", "target-1"]
        manager._submit_batch_transfer_plan = Mock(return_value=[])
        peer = types.SimpleNamespace(
            staging_mem_desc="decode-staging",
            dst_num_target_kv_entries=1,
            dst_kv_item_lens=[100],
        )

        with self.assertRaisesRegex(ValueError, "target descriptor count mismatch"):
            manager._send_staged_kvcache(
                peer,
                np.asarray([10], dtype=np.int32),
                staging_offset=1234,
            )

    def test_local_staging_copy_uses_target_descriptors_and_layer_offsets(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_args = types.SimpleNamespace(
            num_target_kv_entries=2,
            kv_item_lens=[100, 200, 30],
        )
        manager.staging_mem_desc = "decode-staging"
        manager.kv_mem_descs = ["decode-0", "decode-1", "draft"]
        submitted = []
        manager._submit_batch_transfer_plan = (
            lambda src, dst, plan, **kwargs: submitted.append((src, dst, plan)) or []
        )
        manager._wait_transfer_completion = Mock(return_value=None)

        manager.copy_staged_kv_to_pool(
            staging_offset=1234,
            dst_kv_indices=np.asarray([50, 51], dtype=np.int32),
        )

        self.assertEqual(len(submitted), 2)
        self.assertEqual(submitted[0][0], "decode-staging")
        self.assertEqual(submitted[0][1], "decode-0")
        self.assertEqual(submitted[0][2].local_offsets, [1234])
        self.assertEqual(submitted[0][2].remote_offsets, [5000])
        self.assertEqual(submitted[0][2].sizes, [200])

        self.assertEqual(submitted[1][1], "decode-1")
        self.assertEqual(submitted[1][2].local_offsets, [1234 + 100])
        self.assertEqual(submitted[1][2].remote_offsets, [10000])
        self.assertEqual(submitted[1][2].sizes, [400])

    def test_staging_request_sizes_only_target_kv_entries(self):
        class FakeAllocator:
            ALLOC_OVERSIZED = -2

            def __init__(self):
                self.required = None

            def assign(self, required):
                self.required = required
                return (7, 4096, 3)

        manager = MoriKVManager.__new__(MoriKVManager)
        manager.enable_staging = True
        manager.kv_mem_descs = ["decode-0", "decode-1", "draft"]
        manager.kv_args = types.SimpleNamespace(
            num_target_kv_entries=2,
            kv_item_lens=[100, 200, 30],
        )
        allocator = FakeAllocator()
        receiver = types.SimpleNamespace(chunk_staging_infos=[])
        manager._staging_ctx = types.SimpleNamespace(
            allocator=allocator,
            room_receivers={5: receiver},
            room_bootstrap={5: []},
        )
        manager._staging_handler = types.SimpleNamespace(
            register_wm_subscriber=Mock()
        )
        manager._send_staging_rsp = Mock()

        manager._handle_staging_req(
            [
                b"STAGING_REQ",
                b"5",
                b"0",
                b"2",
                b"session",
                b"0",
            ]
        )

        self.assertEqual(allocator.required, 2 * (100 + 200))
        self.assertEqual(receiver.chunk_staging_infos[0], (7, 4096, 3, 4696, 2))
        manager._send_staging_rsp.assert_called_once()
        manager._staging_handler.register_wm_subscriber.assert_called_once_with(
            receiver, "session"
        )

    def test_registration_carries_target_descriptor_count(self):
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
            np.asarray([64, 32], dtype=np.uint64).tobytes(),
            b"",
            b"",
            b"staging-desc",
            b"2",
        ]

        with patch.object(
            mori_conn.EngineDesc,
            "unpack",
            return_value=types.SimpleNamespace(key="engine"),
        ), patch.object(
            mori_conn,
            "_unpack_mem_desc_list",
            side_effect=[["target-0", "target-1"], [], ["staging"]],
        ), patch.object(
            mori_conn,
            "_unpack_mem_desc_lists",
            return_value=[],
        ):
            info = KVArgsRegisterInfo.from_zmq(payload)

        self.assertEqual(info.dst_num_target_kv_entries, 2)
        self.assertEqual(info.staging_mem_desc, "staging")
        self.assertEqual(info.dst_kv_item_lens, [64, 32])


if __name__ == "__main__":
    unittest.main()
