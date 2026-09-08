import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.configs import model_config

with patch.object(
    model_config,
    "get_dsa_mtp_topk_width",
    lambda config: 1,
    create=True,
):
    from sglang.srt.disaggregation.common.staging_handler import (
        prefetch_staging_reqs,
    )
    from sglang.srt.disaggregation.mori import conn as mori_conn
    from sglang.srt.disaggregation.mori.conn import (
        KVArgsRegisterInfo,
        MoriKVManager,
    )

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, suite="stage-a-test-1-gpu-small-amd")


class TestMoriStagingTransfer(unittest.TestCase):
    def test_empty_final_chunk_transfers_state_and_aux_without_staging(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.enable_staging = True
        manager.disaggregation_mode = mori_conn.DisaggregationMode.PREFILL
        manager.request_status = {5: mori_conn.KVPoll.WaitingForInput}
        manager.transfer_lock = threading.Lock()
        info = types.SimpleNamespace(
            engine_key="peer",
            is_dummy=False,
            dst_kv_indices=np.asarray([], dtype=np.int32),
            dst_state_indices=[np.asarray([9], dtype=np.int32)],
            dst_aux_index=7,
        )
        peer = Mock()
        manager.transfer_infos = {5: {"peer": info}}
        manager.decode_kv_args_table = {"peer": peer}
        manager.state_mem_descs = [Mock()]
        manager.pp_group = types.SimpleNamespace(is_last_rank=True)
        manager._should_skip_transfer = Mock(return_value=False)
        manager._staging_room_ready = Mock(
            side_effect=AssertionError("no KV to stage")
        )
        manager._staging_chunk_ready = Mock(
            side_effect=AssertionError("no KV to stage")
        )
        manager.send_kvcache = Mock(side_effect=AssertionError("no KV to send"))
        manager.send_state = Mock(return_value=["state"])
        manager.send_aux = Mock(return_value=["aux"])
        manager._wait_transfer_completion = Mock(return_value=None)
        manager._release_completed_chunk_mappings = Mock()
        manager._send_chunk_ready = Mock(
            side_effect=AssertionError("no chunk to scatter")
        )
        manager._notify_decode_for_room = Mock()
        manager.update_status = Mock()
        chunk = mori_conn.TransferKVChunk(
            room=5,
            prefill_kv_indices=np.asarray([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=True,
            prefill_aux_index=3,
            state_indices=[np.asarray([4], dtype=np.int32)],
        )

        manager._process_transfer_chunk(chunk)

        manager.send_state.assert_called_once_with(
            peer, chunk.state_indices, info.dst_state_indices
        )
        manager.send_aux.assert_called_once_with(peer, 3, 7, 5)
        manager._wait_transfer_completion.assert_called_once_with(["state", "aux"])
        manager._notify_decode_for_room.assert_called_once_with(
            5, mori_conn.KVPoll.Success, target_infos=[info]
        )
        manager.update_status.assert_any_call(5, mori_conn.KVPoll.Success)

    def test_bounded_staging_prefetch_advances_one_chunk_per_call(self):
        socket = Mock()
        context = Mock()
        context.socket.return_value = socket
        address = Mock()
        address.is_ipv6 = False
        address.to_tcp.return_value = "tcp://decode:1234"
        transfer_infos = {
            5: {
                "session": types.SimpleNamespace(
                    is_dummy=False,
                    dst_kv_indices=np.arange(10, dtype=np.int32),
                    endpoint="decode",
                    dst_port=1234,
                )
            }
        }
        requested = set()
        sockets = {}

        with patch(
            "sglang.srt.disaggregation.common.staging_buffer.staging_grid_tokens",
            return_value=4,
        ), patch.object(mori_conn.zmq, "Context", return_value=context), patch(
            "sglang.srt.utils.network.NetworkAddress", return_value=address
        ):
            counts = [
                prefetch_staging_reqs(
                    5,
                    transfer_infos,
                    {"page_size": 1},
                    4,
                    requested,
                    sockets,
                    max_new_chunks_per_session=1,
                )
                for _ in range(4)
            ]

        self.assertEqual(counts, [1, 1, 1, 0])
        self.assertEqual(socket.send_multipart.call_count, 3)
        sent_chunk_ids = [
            int(call.args[0][2].decode("ascii"))
            for call in socket.send_multipart.call_args_list
        ]
        self.assertEqual(sent_chunk_ids, [0, 1, 2])

    def test_mori_sliding_prefetch_uses_thread_local_socket_factory(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.transfer_infos = {5: {"session": Mock()}}
        manager.kv_args = types.SimpleNamespace(page_size=1)
        manager._staging_ctx = types.SimpleNamespace(
            prefetch_requested=set(),
            prefetch_sockets={},
        )
        manager.pp_rank = 0
        manager._connect_threadsafe = Mock()

        with patch.object(
            mori_conn, "get_schedule", return_value=types.SimpleNamespace(
                chunked_prefill_size=4
            )
        ), patch.object(mori_conn, "prefetch_staging_reqs", return_value=1) as prefetch:
            requested = manager._request_next_staging_chunks(5)

        self.assertEqual(requested, 1)
        self.assertIs(
            prefetch.call_args.kwargs["socket_getter"],
            manager._connect_threadsafe,
        )

    def test_staged_prefill_write_uses_all_descriptors_and_layer_offsets(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.attn_tp_size = 4
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager.kv_args = types.SimpleNamespace(
            num_target_kv_entries=2,
            kv_item_lens=[100, 200, 30],
        )
        manager.kv_mem_descs = ["target-0", "target-1", "draft"]
        submitted = []
        manager._submit_batch_transfer_plan = (
            lambda src, dst, plan, **kwargs: submitted.append((src, dst, plan))
            or []
        )
        peer = types.SimpleNamespace(
            staging_mem_desc="decode-staging",
            dst_num_target_kv_entries=2,
            dst_kv_item_lens=[100, 200, 30],
            dst_kv_mem_descs=["decode-0", "decode-1", "decode-draft"],
            decode_tp_size=4,
        )

        manager._send_staged_kvcache(
            peer,
            np.asarray([10, 11], dtype=np.int32),
            staging_offset=1234,
        )

        self.assertEqual(len(submitted), 3)
        self.assertEqual(submitted[0][0], "target-0")
        self.assertEqual(submitted[0][1], "decode-staging")
        self.assertEqual(submitted[0][2].local_offsets, [1000])
        self.assertEqual(submitted[0][2].remote_offsets, [1234])
        self.assertEqual(submitted[0][2].sizes, [200])

        self.assertEqual(submitted[1][0], "target-1")
        self.assertEqual(submitted[1][2].local_offsets, [2000])
        self.assertEqual(submitted[1][2].remote_offsets, [1234 + 2 * 100])
        self.assertEqual(submitted[1][2].sizes, [400])

        self.assertEqual(submitted[2][0], "draft")
        self.assertEqual(submitted[2][2].local_offsets, [300])
        self.assertEqual(submitted[2][2].remote_offsets, [1234 + 2 * 300])
        self.assertEqual(submitted[2][2].sizes, [60])

    def test_staged_prefill_write_rejects_descriptor_geometry_mismatch(self):
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
            dst_kv_mem_descs=["decode-0"],
        )

        with self.assertRaisesRegex(ValueError, "descriptor count mismatch"):
            manager._send_staged_kvcache(
                peer,
                np.asarray([10], dtype=np.int32),
                staging_offset=1234,
            )

    def test_staged_tp4_to_tp2_writers_use_disjoint_head_slices(self):
        src_item_lens = [100, 200, 30]
        dst_item_lens = [200, 400, 60]
        staging_offset = 1234
        submitted_by_rank = {}

        for rank in (0, 1):
            manager = MoriKVManager.__new__(MoriKVManager)
            src_descs = [
                f"prefill-{rank}-0",
                f"prefill-{rank}-1",
                f"prefill-{rank}-2",
            ]
            manager.attn_tp_size = 4
            manager.is_mla_backend = False
            manager.is_hybrid_mla_backend = False
            manager.kv_args = types.SimpleNamespace(
                num_target_kv_entries=2,
                kv_item_lens=src_item_lens,
                page_size=1,
                engine_rank=rank,
                kv_head_num=2,
                total_kv_head_num=8,
            )
            manager.kv_mem_descs = src_descs
            submitted = []
            manager._submit_batch_transfer_plan = (
                lambda src, dst, plan, **kwargs: submitted.append((src, dst, plan))
                or []
            )
            peer = types.SimpleNamespace(
                staging_mem_desc="decode-staging",
                dst_num_target_kv_entries=2,
                dst_kv_item_lens=dst_item_lens,
                dst_kv_mem_descs=["decode-0", "decode-1", "decode-draft"],
                decode_tp_size=2,
                decode_tp_rank=0,
            )

            manager._send_staged_kvcache(
                peer,
                np.asarray([10, 11], dtype=np.int32),
                staging_offset=staging_offset,
            )
            submitted_by_rank[rank] = submitted

        layer_offsets = [0, 2 * 200, 2 * 200 + 2 * 400]
        for rank, submitted in submitted_by_rank.items():
            self.assertEqual(len(submitted), 3)
            for layer_id, (src_item_len, dst_item_len) in enumerate(
                zip(src_item_lens, dst_item_lens)
            ):
                src, dst, plan = submitted[layer_id]
                self.assertEqual(src, f"prefill-{rank}-{layer_id}")
                self.assertEqual(dst, "decode-staging")
                self.assertEqual(len(plan.local_offsets), 2)
                self.assertEqual(len(plan.remote_offsets), 2)
                self.assertEqual(plan.sizes, [src_item_len, src_item_len])
                for page_id, (src_page, dst_page) in enumerate(
                    zip((10, 11), (0, 1))
                ):
                    self.assertEqual(
                        plan.local_offsets[page_id],
                        src_page * src_item_len,
                    )
                    self.assertEqual(
                        plan.remote_offsets[page_id],
                        staging_offset
                        + layer_offsets[layer_id]
                        + dst_page * dst_item_len
                        + rank * src_item_len,
                    )

        for layer_id in range(3):
            for page_id in range(2):
                first = submitted_by_rank[0][layer_id][2]
                second = submitted_by_rank[1][layer_id][2]
                first_range = range(
                    first.remote_offsets[page_id],
                    first.remote_offsets[page_id] + first.sizes[page_id],
                )
                second_range = range(
                    second.remote_offsets[page_id],
                    second.remote_offsets[page_id] + second.sizes[page_id],
                )
                self.assertFalse(set(first_range) & set(second_range))

    def test_staged_tp4_to_tp2_round_trip_preserves_head_order(self):
        src_item_lens = [100, 200, 30]
        dst_item_lens = [200, 400, 60]
        src_indices = np.asarray([10, 11], dtype=np.int32)
        dst_indices = np.asarray([50, 51], dtype=np.int32)
        staging_offset = 17
        memories = {
            "staging": bytearray(
                staging_offset + len(src_indices) * sum(dst_item_lens)
            )
        }

        def copy_plan(src, dst, plan, **kwargs):
            for src_offset, dst_offset, size in zip(
                plan.local_offsets, plan.remote_offsets, plan.sizes
            ):
                self.assertLessEqual(src_offset + size, len(memories[src]))
                self.assertLessEqual(dst_offset + size, len(memories[dst]))
                memories[dst][dst_offset : dst_offset + size] = memories[src][
                    src_offset : src_offset + size
                ]
            return []

        peer = types.SimpleNamespace(
            staging_mem_desc="staging",
            dst_num_target_kv_entries=2,
            dst_kv_item_lens=dst_item_lens,
            dst_kv_mem_descs=["decode-0", "decode-1", "decode-draft"],
            decode_tp_size=2,
            decode_tp_rank=0,
        )
        dst_descs = ["decode-0", "decode-1", "decode-2"]

        for layer_id, (src_item_len, dst_item_len) in enumerate(
            zip(src_item_lens, dst_item_lens)
        ):
            for rank in (0, 1):
                memories[f"prefill-{rank}-{layer_id}"] = bytearray(
                    (int(src_indices[-1]) + 1) * src_item_len
                )
            memories[dst_descs[layer_id]] = bytearray(
                (int(dst_indices[-1]) + 1) * dst_item_len
            )

        for rank in (0, 1):
            manager = MoriKVManager.__new__(MoriKVManager)
            manager.attn_tp_size = 4
            manager.is_mla_backend = False
            manager.is_hybrid_mla_backend = False
            manager.kv_args = types.SimpleNamespace(
                num_target_kv_entries=2,
                kv_item_lens=src_item_lens,
                page_size=1,
                engine_rank=rank,
                kv_head_num=2,
                total_kv_head_num=8,
            )
            manager.kv_mem_descs = [
                f"prefill-{rank}-0",
                f"prefill-{rank}-1",
                f"prefill-{rank}-2",
            ]
            manager._submit_batch_transfer_plan = copy_plan
            manager._send_staged_kvcache(peer, src_indices, staging_offset)

        decode_manager = MoriKVManager.__new__(MoriKVManager)
        decode_manager.is_mla_backend = False
        decode_manager.is_hybrid_mla_backend = False
        decode_manager.kv_args = types.SimpleNamespace(
            num_target_kv_entries=2,
            kv_item_lens=dst_item_lens,
        )
        decode_manager.staging_mem_desc = "staging"
        decode_manager.kv_mem_descs = dst_descs
        decode_manager._submit_batch_transfer_plan = copy_plan
        decode_manager._wait_transfer_completion = Mock(return_value=None)
        decode_manager.copy_staged_kv_to_pool(staging_offset, dst_indices)

        for layer_id, (src_item_len, dst_item_len) in enumerate(
            zip(src_item_lens, dst_item_lens)
        ):
            for src_index, dst_index in zip(src_indices, dst_indices):
                expected = (
                    memories[f"prefill-0-{layer_id}"][
                        src_index * src_item_len : (src_index + 1) * src_item_len
                    ]
                    + memories[f"prefill-1-{layer_id}"][
                        src_index * src_item_len : (src_index + 1) * src_item_len
                    ]
                )
                actual = memories[dst_descs[layer_id]][
                    dst_index * dst_item_len : (dst_index + 1) * dst_item_len
                ]
                self.assertEqual(actual, expected)

    def test_local_staging_copy_uses_all_descriptors_and_layer_offsets(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_args = types.SimpleNamespace(
            num_target_kv_entries=2,
            kv_item_lens=[100, 200, 30],
        )
        manager.staging_mem_desc = "decode-staging"
        manager.kv_mem_descs = ["decode-0", "decode-1", "draft"]
        submitted = []
        manager._submit_batch_transfer_plan = (
            lambda src, dst, plan, **kwargs: submitted.append((src, dst, plan))
            or []
        )
        manager._wait_transfer_completion = Mock(return_value=None)

        manager.copy_staged_kv_to_pool(
            staging_offset=1234,
            dst_kv_indices=np.asarray([50, 51], dtype=np.int32),
        )

        self.assertEqual(len(submitted), 3)
        self.assertEqual(submitted[0][0], "decode-staging")
        self.assertEqual(submitted[0][1], "decode-0")
        self.assertEqual(submitted[0][2].local_offsets, [1234])
        self.assertEqual(submitted[0][2].remote_offsets, [5000])
        self.assertEqual(submitted[0][2].sizes, [200])

        self.assertEqual(submitted[1][1], "decode-1")
        self.assertEqual(submitted[1][2].local_offsets, [1234 + 2 * 100])
        self.assertEqual(submitted[1][2].remote_offsets, [10000])
        self.assertEqual(submitted[1][2].sizes, [400])

        self.assertEqual(submitted[2][1], "draft")
        self.assertEqual(submitted[2][2].local_offsets, [1234 + 2 * 300])
        self.assertEqual(submitted[2][2].remote_offsets, [1500])
        self.assertEqual(submitted[2][2].sizes, [60])

    def test_staged_round_trip_preserves_every_descriptor_and_page(self):
        for num_pages in (1, 2, 128):
            with self.subTest(num_pages=num_pages):
                item_lens = [8, 12, 4]
                src_indices = np.arange(num_pages, dtype=np.int32) * 2 + 1
                dst_indices = np.arange(num_pages, dtype=np.int32) * 3 + 2
                src_descs = ["src-0", "src-1", "src-draft"]
                dst_descs = ["dst-0", "dst-1", "dst-draft"]
                staging_offset = 17
                memories = {
                    "staging": bytearray(staging_offset + num_pages * sum(item_lens))
                }
                for layer, (src, dst, item_len) in enumerate(
                    zip(src_descs, dst_descs, item_lens)
                ):
                    memories[src] = bytearray(
                        (layer * 67 + i) % 256
                        for i in range((int(src_indices[-1]) + 1) * item_len)
                    )
                    memories[dst] = bytearray(
                        (int(dst_indices[-1]) + 1) * item_len
                    )

                def copy_plan(src, dst, plan, **kwargs):
                    for src_offset, dst_offset, size in zip(
                        plan.local_offsets, plan.remote_offsets, plan.sizes
                    ):
                        self.assertLessEqual(src_offset + size, len(memories[src]))
                        self.assertLessEqual(dst_offset + size, len(memories[dst]))
                        memories[dst][dst_offset : dst_offset + size] = memories[src][
                            src_offset : src_offset + size
                        ]
                    return []

                manager = MoriKVManager.__new__(MoriKVManager)
                manager.attn_tp_size = 4
                manager.is_mla_backend = False
                manager.is_hybrid_mla_backend = False
                manager.kv_args = types.SimpleNamespace(
                    num_target_kv_entries=2, kv_item_lens=item_lens
                )
                manager.kv_mem_descs = src_descs
                manager.staging_mem_desc = "staging"
                manager._submit_batch_transfer_plan = copy_plan
                manager._wait_transfer_completion = Mock(return_value=None)
                peer = types.SimpleNamespace(
                    staging_mem_desc="staging",
                    dst_num_target_kv_entries=2,
                    dst_kv_item_lens=item_lens,
                    dst_kv_mem_descs=dst_descs,
                    decode_tp_size=4,
                )
                manager._send_staged_kvcache(peer, src_indices, staging_offset)
                manager.kv_mem_descs = dst_descs
                manager.copy_staged_kv_to_pool(staging_offset, dst_indices)

                for src, dst, item_len in zip(src_descs, dst_descs, item_lens):
                    for src_index, dst_index in zip(src_indices, dst_indices):
                        src_start, dst_start = src_index * item_len, dst_index * item_len
                        self.assertEqual(
                            memories[dst][dst_start : dst_start + item_len],
                            memories[src][src_start : src_start + item_len],
                        )

    def test_staging_request_sizes_target_and_draft_kv_entries(self):
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
        manager.transfer_lock = threading.Lock()
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

        self.assertEqual(allocator.required, 2 * (100 + 200 + 30))
        self.assertEqual(receiver.chunk_staging_infos[0], (7, 4096, 3, 4756, 2))
        manager._send_staging_rsp.assert_called_once()
        manager._staging_handler.register_wm_subscriber.assert_called_once_with(
            receiver, "session"
        )

    def test_staging_allocation_cannot_outlive_room_cleanup(self):
        with patch(
            "sglang.srt.disaggregation.common.staging_buffer.StagingBuffer"
        ) as buffer:
            buffer.return_value.data_ptr = 0
            allocator = mori_conn.StagingAllocator(1024, "cpu", 0)
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.enable_staging = True
        manager.transfer_lock = threading.Lock()
        manager.kv_args = types.SimpleNamespace(kv_item_lens=[16])
        receiver = types.SimpleNamespace(chunk_staging_infos=[])
        decode_req = types.SimpleNamespace(_chunk_events=[])
        manager._staging_ctx = types.SimpleNamespace(
            allocator=allocator,
            room_receivers={5: receiver},
            room_bootstrap={5: []},
        )
        handler = mori_conn.MoriDecodeStagingHandler.__new__(
            mori_conn.MoriDecodeStagingHandler
        )
        handler.kv_manager = manager
        handler.staging_allocator = allocator
        handler._room_to_decode_req = {5: decode_req}
        handler._room_to_receiver = {5: receiver}
        handler._writer_counts = {}
        handler._wm_subscribers = {}
        handler.register_wm_subscriber = Mock()
        manager._staging_handler = handler
        manager._send_staging_rsp = Mock()
        assign_entered = threading.Event()
        resume_assign = threading.Event()
        cleanup_entered = threading.Event()
        original_assign = allocator.assign

        def paused_assign(required):
            assign_entered.set()
            if not resume_assign.wait(5):
                raise TimeoutError("test did not resume allocation")
            return original_assign(required)

        def cleanup():
            cleanup_entered.set()
            handler.unregister_decode_req(5)

        with patch.object(allocator, "assign", side_effect=paused_assign):
            with ThreadPoolExecutor(max_workers=2) as workers:
                allocation = workers.submit(
                    manager._handle_staging_req,
                    [b"STAGING_REQ", b"5", b"0", b"2", b"session", b"0"],
                )
                self.assertTrue(assign_entered.wait(5))
                release = workers.submit(cleanup)
                self.assertTrue(cleanup_entered.wait(5))
                try:
                    # Cleanup must not finish while a receiver-checked
                    # allocation can still publish into that room.
                    try:
                        release.result(timeout=0.1)
                        cleaned_early = True
                    except TimeoutError:
                        cleaned_early = False
                finally:
                    resume_assign.set()
                    allocation.result(timeout=5)
                    release.result(timeout=5)

        self.assertEqual(allocator.allocations, {})
        self.assertFalse(cleaned_early)
        self.assertEqual(handler._room_to_receiver, {})
        self.assertEqual(manager._staging_ctx.room_receivers, {})

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

    def test_state_only_hybrid_mla_dummy_rank_stays_dummy_for_kv(self):
        state_bytes = mori_conn._pack_state_indices(
            [np.asarray([11], dtype=np.int32)]
        )
        payload = [
            b"5",
            b"decode",
            b"1234",
            b"engine",
            b"",
            b"",
            state_bytes,
            b"1",
            b"",
        ]

        info = mori_conn.TransferInfo.from_zmq(payload)

        self.assertTrue(info.is_dummy)
        self.assertEqual(info.dst_state_indices, [np.asarray([11])])

    def test_hybrid_mla_state_fan_in_waits_for_both_tp_ranks(self):
        from sglang.srt.disaggregation.common.conn import CommonKVManager

        manager = CommonKVManager.__new__(CommonKVManager)
        manager.attn_tp_size = 2
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = True
        manager.state_mem_descs = [Mock()]
        manager.kv_args = types.SimpleNamespace(engine_rank=0)
        info = types.SimpleNamespace(
            attn_tp_size=4,
            attn_cp_size=1,
            pp_size=1,
        )

        CommonKVManager._resolve_rank_mapping(manager, info)

        self.assertEqual(info.target_tp_ranks, [0, 1])
        self.assertEqual(info.required_prefill_response_num, 2)

    def test_staging_room_ready_ignores_state_only_dummy_rank(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.enable_staging = True
        manager.transfer_lock = threading.Lock()
        manager._staging_chunk_ready = Mock(return_value=(True, 0, 1234))
        manager.transfer_infos = {
            5: {
                "real": types.SimpleNamespace(is_dummy=False),
                "state-only": types.SimpleNamespace(is_dummy=True),
            }
        }

        self.assertTrue(manager._staging_room_ready(5, slice(0, 2)))

    def test_state_only_dummy_rank_sends_mamba_state_without_kv(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.disaggregation_mode = mori_conn.DisaggregationMode.PREFILL
        manager.request_status = {5: mori_conn.KVPoll.WaitingForInput}
        manager.transfer_lock = threading.Lock()
        manager.transfer_infos = {
            5: {
                "engine": types.SimpleNamespace(
                    engine_key="engine",
                    is_dummy=True,
                    dst_kv_indices=np.asarray([], dtype=np.int32),
                    dst_aux_index=-1,
                    dst_state_indices=[np.asarray([11], dtype=np.int32)],
                )
            }
        }
        manager.decode_kv_args_table = {"engine": Mock()}
        manager.enable_staging = False
        manager.state_mem_descs = [Mock()]
        manager.pp_group = types.SimpleNamespace(is_last_rank=True)
        manager.update_status = Mock()
        manager.send_kvcache = Mock(side_effect=AssertionError("no KV to send"))
        manager.send_state = Mock(return_value=["state"])
        manager.send_aux = Mock(side_effect=AssertionError("no aux to send"))
        manager._wait_transfer_completion = Mock(return_value=None)
        manager._release_completed_chunk_mappings = Mock()
        manager._notify_decode_for_room = Mock()
        manager._send_chunk_ready = Mock(
            side_effect=AssertionError("no chunk to scatter")
        )
        manager._request_next_staging_chunks = Mock()
        manager._should_skip_transfer = Mock(return_value=False)

        statuses, target_infos = manager._submit_kv_transfer(
            5,
            np.asarray([10, 11], dtype=np.int32),
            slice(0, 2),
            True,
            aux_index=3,
            state_indices=[np.asarray([4], dtype=np.int32)],
        )

        self.assertEqual(statuses, ["state"])
        self.assertEqual(len(target_infos), 1)
        manager.send_state.assert_called_once_with(
            manager.decode_kv_args_table["engine"],
            [np.asarray([4], dtype=np.int32)],
            [np.asarray([11], dtype=np.int32)],
        )

    def test_hybrid_mla_tp4_to_tp2_mamba_conv_state_uses_row_slices(self):
        submitted_by_rank = {}
        for prefill_rank in (0, 1):
            manager = MoriKVManager.__new__(MoriKVManager)
            manager.attn_tp_size = 4
            manager.kv_args = types.SimpleNamespace(
                engine_rank=prefill_rank,
                state_conv_shard_groups=[[8, 8, 8]],
                state_slice_outer_counts=[3],
            )
            manager.state_mem_descs = [["prefill-conv"]]
            manager.state_mem_desc_offsets = [[100]]
            submitted = []
            manager._submit_batch_transfer_plan = (
                lambda src, dst, plan, **kwargs: submitted.append((src, dst, plan))
                or []
            )
            peer = types.SimpleNamespace(
                decode_tp_size=2,
                decode_tp_rank=0,
                dst_state_mem_descs=[["decode-conv"]],
                dst_state_mem_desc_offsets=[[200]],
                dst_state_item_lens=[[36]],
                dst_state_dim_per_tensor=[[12]],
            )

            manager._send_mamba_state(
                peer,
                src_state_indices=np.asarray([7], dtype=np.int32),
                dst_state_indices=np.asarray([11], dtype=np.int32),
                src_state_mem_descs=["prefill-conv"],
                dst_state_mem_descs=["decode-conv"],
                src_state_mem_desc_offsets=[100],
                dst_state_mem_desc_offsets=[200],
                src_state_item_lens=[18],
                dst_state_item_lens=[36],
                src_state_slot_strides=[18],
                dst_state_slot_strides=[36],
                src_state_dim_per_tensor=[6],
                dst_state_dim_per_tensor=[12],
            )

            submitted_by_rank[prefill_rank] = submitted[0][2]

        self.assertEqual(
            submitted_by_rank[0].local_offsets,
            [226, 228, 230, 232, 234, 236, 238, 240, 242],
        )
        self.assertEqual(
            submitted_by_rank[0].remote_offsets,
            [596, 600, 604, 608, 612, 616, 620, 624, 628],
        )
        self.assertEqual(submitted_by_rank[0].sizes, [2] * 9)

        self.assertEqual(
            submitted_by_rank[1].local_offsets,
            [226, 228, 230, 232, 234, 236, 238, 240, 242],
        )
        self.assertEqual(
            submitted_by_rank[1].remote_offsets,
            [598, 602, 606, 610, 614, 618, 622, 626, 630],
        )
        self.assertEqual(submitted_by_rank[1].sizes, [2] * 9)

    def test_staged_hybrid_mla_tp_mismatch_transfers_full_latent_kv(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.attn_tp_size = 4
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = True
        manager.kv_args = types.SimpleNamespace(
            num_target_kv_entries=2,
            kv_item_lens=[100, 200, 30],
        )
        manager.kv_mem_descs = ["target-0", "target-1", "draft"]
        submitted = []
        manager._submit_batch_transfer_plan = (
            lambda src, dst, plan, **kwargs: submitted.append((src, dst, plan))
            or []
        )
        peer = types.SimpleNamespace(
            staging_mem_desc="decode-staging",
            dst_num_target_kv_entries=2,
            dst_kv_item_lens=[100, 200, 30],
            dst_kv_mem_descs=["decode-0", "decode-1", "decode-draft"],
            decode_tp_size=2,
            decode_tp_rank=0,
        )

        manager._send_staged_kvcache(
            peer,
            np.asarray([10, 11], dtype=np.int32),
            staging_offset=1234,
        )

        self.assertEqual(len(submitted), 3)
        for layer_id, (src, dst, plan) in enumerate(submitted):
            src_item_len = manager.kv_args.kv_item_lens[layer_id]
            self.assertEqual(src, manager.kv_mem_descs[layer_id])
            self.assertEqual(dst, "decode-staging")
            self.assertEqual(
                plan.local_offsets,
                [10 * src_item_len],
            )
            self.assertEqual(
                plan.remote_offsets,
                [
                    1234
                    + sum(
                        2 * item_len
                        for item_len in peer.dst_kv_item_lens[:layer_id]
                    )
                ],
            )
            self.assertEqual(plan.sizes, [2 * peer.dst_kv_item_lens[layer_id]])

    def test_hybrid_mla_tp4_to_tp2_chunk_ready_submits_after_one_writer(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.attn_tp_size = 2
        manager.pp_size = 1
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = True
        receiver = types.SimpleNamespace(
            prefill_info=types.SimpleNamespace(attn_tp_size=4, pp_size=1)
        )
        handler = mori_conn.MoriDecodeStagingHandler.__new__(
            mori_conn.MoriDecodeStagingHandler
        )
        handler.kv_manager = manager
        handler.decode_tp = 2
        handler._room_to_receiver = {5: receiver}
        handler._writer_counts = {}
        handler.submit_chunk_scatter = Mock()

        submitted = handler.handle_chunk_arrived(
            room=5,
            chunk_idx=0,
            page_start=0,
            num_pages=2,
            writer_id="prefill-2",
        )

        self.assertTrue(submitted)
        handler.submit_chunk_scatter.assert_called_once_with(5, 0, 0, 2)
        self.assertNotIn(0, handler._writer_counts[5])


if __name__ == "__main__":
    unittest.main()
