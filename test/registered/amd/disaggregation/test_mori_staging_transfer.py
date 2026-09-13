import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.configs import model_config
from sglang.srt.disaggregation.common.conn import PrefillServerInfo
from sglang.srt.disaggregation.common.staging_handler import (
    prefetch_staging_reqs,
)

with patch.object(
    model_config,
    "get_dsa_mtp_topk_width",
    lambda config: 1,
    create=True,
):
    from sglang.srt.disaggregation.mori import conn as mori_conn
    from sglang.srt.disaggregation.mori.conn import (
        MoriKVManager,
        MoriKVReceiver,
    )

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, suite="stage-a-test-1-gpu-small-amd")


class TestMoriStagingTransfer(unittest.TestCase):
    def test_prefetch_reports_incomplete_without_shared_socket_cache(self):
        sockets = []

        def socket_getter(endpoint, is_ipv6=False):
            sockets.append(endpoint)
            return Mock()

        tinfo = types.SimpleNamespace(
            is_dummy=False,
            endpoint="127.0.0.1",
            dst_port=1111,
            dst_kv_indices=np.arange(128),
        )
        requested = set()
        emitted, complete = prefetch_staging_reqs(
            5,
            {5: {"session": tinfo}},
            {"page_size": 64},
            1024,
            requested,
            {},
            requester_pp_rank=0,
            socket_getter=socket_getter,
            socket_cache=False,
            return_complete=True,
        )

        self.assertEqual(emitted, 8)
        self.assertTrue(complete)
        self.assertEqual(len(sockets), 8)
        self.assertEqual(requested, {(5, idx, "session") for idx in range(8)})

    def test_prefetch_reports_partial_failure_as_incomplete(self):
        def socket_getter(endpoint, is_ipv6=False):
            if endpoint == "tcp://127.0.0.1:1112":
                raise AssertionError("intentional test failure")
            return Mock()

        infos = {
            "good": types.SimpleNamespace(
                is_dummy=False,
                endpoint="127.0.0.1",
                dst_port=1111,
                dst_kv_indices=np.arange(64),
            ),
            "bad": types.SimpleNamespace(
                is_dummy=False,
                endpoint="127.0.0.1",
                dst_port=1112,
                dst_kv_indices=np.arange(64),
            ),
        }
        requested = set()
        emitted, complete = prefetch_staging_reqs(
            5,
            {5: infos},
            {"page_size": 64},
            1024,
            requested,
            {},
            requester_pp_rank=0,
            socket_getter=socket_getter,
            socket_cache=False,
            return_complete=True,
        )

        self.assertEqual(emitted, 4)
        self.assertFalse(complete)
        self.assertEqual(requested, {(5, idx, "good") for idx in range(4)})

    def test_chunk_ready_send_failure_is_reported(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager._staging_full_chunk_pages = 64
        manager._compute_prefill_unique_rank = lambda: "writer"
        manager._connect_threadsafe = Mock(
            side_effect=[
                Mock(send_multipart=Mock()),
                AssertionError("intentional test failure"),
            ]
        )
        infos = [
            types.SimpleNamespace(engine_key="good", endpoint="1", dst_port=1),
            types.SimpleNamespace(engine_key="bad", endpoint="2", dst_port=2),
        ]

        self.assertFalse(manager._send_chunk_ready(infos, 5, slice(0, 64), 64))

    def test_mori_staging_requires_prefill_capability(self):
        receiver = MoriKVReceiver.__new__(MoriKVReceiver)
        receiver.kv_mgr = types.SimpleNamespace(enable_staging=True, pp_size=4)
        receiver.prefill_info = PrefillServerInfo(
            attn_tp_size=4,
            attn_cp_size=1,
            dp_size=1,
            pp_size=4,
            page_size=64,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            enable_staging=False,
        )

        with self.assertRaisesRegex(
            RuntimeError, "both the prefill and decode servers"
        ):
            receiver._validate_prefill_staging_capability()

    def test_staged_prefill_write_uses_target_descriptors_and_layer_offsets(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_args = types.SimpleNamespace(
            num_target_kv_entries=2,
            kv_item_lens=[100, 200, 30],
        )
        manager.kv_mem_descs = ["target-0", "target-1", "draft"]
        submitted = []
        manager._submit_batch_transfer_plan = lambda src, dst, plan, **kwargs: (
            submitted.append((src, dst, plan)) or []
        )
        peer = types.SimpleNamespace(
            staging_mem_desc="decode-staging",
            dst_num_target_kv_entries=2,
            dst_kv_item_lens=[100, 200, 30],
            decode_tp_size=1,
        )
        manager.attn_tp_size = 1

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
        self.assertEqual(submitted[1][2].remote_offsets, [1234 + 200])
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
        manager._submit_batch_transfer_plan = lambda src, dst, plan, **kwargs: (
            submitted.append((src, dst, plan)) or []
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
        self.assertEqual(submitted[1][2].local_offsets, [1234 + 200])
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
        manager.transfer_lock = threading.Lock()
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
        manager._staging_handler = types.SimpleNamespace(register_wm_subscriber=Mock())
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

    def test_state_only_dummy_rank_stays_dummy_with_partial_prefix(self):
        state_bytes = mori_conn._pack_state_indices([np.asarray([11], dtype=np.int32)])
        payload = [
            b"5",
            b"decode",
            b"1234",
            b"engine",
            b"",
            b"",
            state_bytes,
            b"1",
            b"32768",
        ]

        info = mori_conn.TransferInfo.from_zmq(payload)

        self.assertTrue(info.is_dummy)
        self.assertEqual(info.decode_prefix_len, 32768)
        self.assertEqual(info.dst_state_indices, [np.asarray([11])])


if __name__ == "__main__":
    unittest.main()
