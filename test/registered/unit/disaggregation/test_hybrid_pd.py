import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_hicache_mixin import HiCacheRestoreResult
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHybridDisaggregationMode(unittest.TestCase):
    def test_parses_hybrid_value(self):
        self.assertEqual(DisaggregationMode("hybrid"), DisaggregationMode.HYBRID)
        self.assertEqual(DisaggregationMode.HYBRID.value, "hybrid")

    def test_to_engine_type_hybrid(self):
        self.assertEqual(DisaggregationMode.to_engine_type("hybrid"), "hybrid")


class TestHybridRoleRouting(unittest.TestCase):
    def _new_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.HYBRID
        scheduler.enable_priority_scheduling = False
        scheduler.abort_on_priority_when_disabled = False
        scheduler.waiting_queue = []
        scheduler._prefetch_kvcache = MagicMock()
        scheduler.model_config = SimpleNamespace(num_key_value_heads=8)
        scheduler.disagg_prefill_bootstrap_queue = MagicMock()
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        return scheduler

    def _new_req(self, disagg_role):
        req = MagicMock()
        req.disagg_role = disagg_role
        req.priority = None
        req.rid = "req"
        req.time_stats = MagicMock()
        return req

    def test_decode_role_goes_to_decode_queue(self):
        scheduler = self._new_scheduler()
        req = self._new_req("decode")

        scheduler._add_request_to_queue(req)

        args, kwargs = scheduler.disagg_decode_prealloc_queue.add.call_args
        self.assertIs(args[0], req)
        self.assertEqual(
            kwargs,
            {"is_retracted": False},
        )
        scheduler.disagg_prefill_bootstrap_queue.add.assert_not_called()

    def test_prefill_role_goes_to_prefill_queue(self):
        scheduler = self._new_scheduler()

        scheduler._add_request_to_queue(self._new_req("prefill"))

        scheduler.disagg_prefill_bootstrap_queue.add.assert_called_once()
        scheduler.disagg_decode_prealloc_queue.add.assert_not_called()

    def test_none_role_defaults_to_prefill_queue(self):
        scheduler = self._new_scheduler()

        scheduler._add_request_to_queue(self._new_req(None))

        scheduler.disagg_prefill_bootstrap_queue.add.assert_called_once()
        scheduler.disagg_decode_prealloc_queue.add.assert_not_called()


class TestHybridQueueInitialization(unittest.TestCase):
    def test_init_disaggregation_hybrid_creates_both_queue_sets(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.draft_worker = None
        scheduler.spec_algorithm = MagicMock()
        scheduler.spec_algorithm.carries_draft_hidden_states.return_value = False
        scheduler.model_config = SimpleNamespace(hf_config=MagicMock())
        scheduler.req_to_token_pool = SimpleNamespace(size=16)
        scheduler.max_running_requests = 4
        scheduler.max_total_num_tokens = 64
        scheduler.ps = SimpleNamespace(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            gpu_id=0,
        )
        scheduler.attn_tp_cpu_group = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.get_kvcache.return_value = MagicMock()
        scheduler.token_to_kv_pool_allocator.get_kvcache.return_value.maybe_get_custom_mem_pool.return_value = None
        scheduler.enable_unified_memory = False

        prefill_q = MagicMock()
        prefill_q.enable_staging = False
        decode_prealloc = MagicMock()
        decode_transfer = MagicMock()
        with (
            patch(
                "sglang.srt.managers.scheduler.PrefillBootstrapQueue",
                return_value=prefill_q,
            ),
            patch(
                "sglang.srt.managers.scheduler.DecodePreallocQueue",
                return_value=decode_prealloc,
            ),
            patch(
                "sglang.srt.managers.scheduler.DecodeTransferQueue",
                return_value=decode_transfer,
            ),
            patch(
                "sglang.srt.managers.scheduler.ReqToMetadataIdxAllocator"
            ) as mock_allocator_factory,
            patch(
                "sglang.srt.managers.scheduler.MetadataBuffers"
            ) as mock_metadata_factory,
            patch(
                "sglang.srt.managers.scheduler.get_dsa_seed_metadata_dim",
                return_value=0,
            ),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(dp_size=1),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_mode="hybrid",
                    disaggregation_transfer_backend="fake",
                    disaggregation_bootstrap_port=9000,
                    num_reserved_decode_tokens=1,
                    language_only=False,
                    encoder_transfer_backend="zmq_to_scheduler",
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler.is_minimax_sparse",
                return_value=False,
            ),
        ):
            scheduler.init_disaggregation()

        self.assertIsNotNone(scheduler.disagg_prefill_bootstrap_queue)
        self.assertIsNotNone(scheduler.disagg_prefill_inflight_queue)
        self.assertIsNotNone(scheduler.disagg_prefill_pending_chunk_rids)
        self.assertIsNotNone(scheduler.disagg_decode_prealloc_queue)
        self.assertIsNotNone(scheduler.disagg_decode_transfer_queue)
        self.assertIsNotNone(
            scheduler.disagg_prefill_req_to_metadata_buffer_idx_allocator
        )
        self.assertIsNotNone(
            scheduler.disagg_decode_req_to_metadata_buffer_idx_allocator
        )
        self.assertIsNotNone(scheduler.disagg_prefill_metadata_buffers)
        self.assertIsNotNone(scheduler.disagg_decode_metadata_buffers)
        self.assertIs(
            scheduler.disagg_prefill_metadata_buffers,
            scheduler.disagg_decode_metadata_buffers,
        )
        self.assertEqual(mock_allocator_factory.call_count, 2)
        self.assertEqual(mock_metadata_factory.call_count, 1)


class TestHybridIdleVisibility(unittest.TestCase):
    def _new_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.HYBRID
        scheduler.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        scheduler.chunked_req = None
        scheduler.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler.ps = SimpleNamespace(pp_size=1)
        scheduler.waiting_queue = []
        scheduler.hybrid_waiting_queue_prefill = []
        scheduler.hybrid_waiting_queue_decode = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        return scheduler

    def test_pending_prefill_inflight_is_not_idle(self):
        scheduler = self._new_scheduler()
        scheduler.disagg_prefill_inflight_queue.append(SimpleNamespace(rid="p1"))

        self.assertFalse(scheduler.is_fully_idle())

    def test_empty_hybrid_queues_are_idle(self):
        scheduler = self._new_scheduler()

        self.assertTrue(scheduler.is_fully_idle())


class TestHybridLocalTransfer(unittest.TestCase):
    def test_retry_waits_for_decode_metadata(self):
        scheduler = SchedulerDisaggregationPrefillMixin.__new__(
            SchedulerDisaggregationPrefillMixin
        )
        scheduler.send_kv_chunk = MagicMock()
        req = SimpleNamespace(
            pending_bootstrap=False,
            _hybrid_last_chunk_retried=False,
            disagg_kv_sender=SimpleNamespace(
                bootstrap_room=123,
                kv_mgr=SimpleNamespace(transfer_infos={}),
            ),
        )

        self.assertFalse(
            SchedulerDisaggregationPrefillMixin._hybrid_retry_last_chunk(scheduler, req)
        )
        scheduler.send_kv_chunk.assert_not_called()

        req.disagg_kv_sender.kv_mgr.transfer_infos[123] = object()
        self.assertTrue(
            SchedulerDisaggregationPrefillMixin._hybrid_retry_last_chunk(scheduler, req)
        )
        scheduler.send_kv_chunk.assert_called_once_with(req, last_chunk=True)
        self.assertTrue(req._hybrid_last_chunk_retried)

        scheduler.send_kv_chunk.reset_mock()
        self.assertFalse(
            SchedulerDisaggregationPrefillMixin._hybrid_retry_last_chunk(scheduler, req)
        )
        scheduler.send_kv_chunk.assert_not_called()

    def test_hybrid_prefill_transfer_does_not_insert_shared_prefix(self):
        scheduler = SchedulerDisaggregationPrefillMixin.__new__(
            SchedulerDisaggregationPrefillMixin
        )
        scheduler.disaggregation_mode = DisaggregationMode.HYBRID
        scheduler.attn_cp_cpu_group = None
        scheduler.attn_tp_cpu_group = None
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.tree_cache = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.output_streamer = MagicMock()
        scheduler.disagg_prefill_req_to_metadata_buffer_idx_allocator = MagicMock()
        req = SimpleNamespace(
            rid="p1",
            pending_bootstrap=False,
            finished_reason=None,
            disagg_kv_sender=SimpleNamespace(
                clear=MagicMock(),
                get_transfer_metric=lambda: MagicMock(),
            ),
            time_stats=SimpleNamespace(
                set_prefill_kv_transfer_finish_time=MagicMock(),
                set_completion_time=MagicMock(),
                compute_and_observe_kv_transfer_metrics=MagicMock(),
            ),
            bootstrap_host="host",
            return_logprob=False,
            metadata_buffer_index=0,
        )
        scheduler.disagg_prefill_inflight_queue = [req]

        with (
            patch(
                "sglang.srt.disaggregation.prefill.release_kv_cache"
            ) as release_kv_cache,
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Success],
            ),
            patch.object(
                SchedulerDisaggregationPrefillMixin,
                "_hybrid_commit_local_transfer",
                return_value=True,
            ),
        ):
            done_reqs = scheduler.process_disagg_prefill_inflight_queue()

        self.assertEqual(done_reqs, [req])
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )

    def test_commit_copies_mapping_and_metadata_to_decode(self):
        scheduler = SchedulerDisaggregationPrefillMixin.__new__(
            SchedulerDisaggregationPrefillMixin
        )
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(32, dtype=torch.int64).reshape(2, 16),
            req_index_to_mamba_index_mapping=torch.tensor(
                [[7], [11]], dtype=torch.int32
            ),
            device="cpu",
            translate_mamba_indices=lambda indices: indices,
            mamba_pool=SimpleNamespace(
                copy_from=MagicMock(
                    side_effect=lambda src, dst: setattr(
                        dst,
                        "tolist",
                        lambda: src.tolist(),
                    )
                )
            ),
        )
        kv_cache = SimpleNamespace(
            move_kv_cache=MagicMock(
                side_effect=lambda dst, src: setattr(
                    dst, "tolist", lambda: src.tolist()
                )
            )
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: kv_cache,
            translate_kv_indices_for_transfer=lambda indices: indices + 100,
            page_size=1,
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.disagg_decode_metadata_buffers = SimpleNamespace(
            bootstrap_room=torch.zeros((1, 1), dtype=torch.int64)
        )
        scheduler.disagg_prefill_metadata_buffers = SimpleNamespace(
            copy_row=MagicMock(
                side_effect=lambda src_idx, dst_idx, destination: (
                    destination.bootstrap_room[dst_idx].fill_(123)
                )
            )
        )

        prefill_req = SimpleNamespace(
            rid="prefill-rid",
            bootstrap_room=123,
            metadata_buffer_index=0,
            kv=SimpleNamespace(req_pool_idx=0),
            disagg_decode_prefix_len=2,
            extend_range=SimpleNamespace(end=5),
            disagg_kv_sender=SimpleNamespace(
                kv_mgr=SimpleNamespace(update_status=MagicMock())
            ),
        )
        decode_req = SimpleNamespace(
            metadata_buffer_index=0,
            req=SimpleNamespace(
                rid="decode-rid",
                bootstrap_room=123,
                kv=SimpleNamespace(req_pool_idx=1),
                set_extend_range=MagicMock(),
                prefix_indices=None,
            ),
            kv_receiver=SimpleNamespace(conclude_state=None),
            waiting_for_input=False,
        )
        decode_req.kv_receiver.require_staging = True
        scheduler.disagg_decode_transfer_queue.queue.append(decode_req)

        self.assertTrue(
            SchedulerDisaggregationPrefillMixin._hybrid_commit_local_transfer(
                scheduler, prefill_req
            )
        )

        move_args = kv_cache.move_kv_cache.call_args.args
        self.assertTrue(
            torch.equal(move_args[0], torch.arange(118, 121, dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(move_args[1], torch.arange(102, 105, dtype=torch.int64))
        )
        self.assertEqual(
            decode_req.req.prefix_indices.tolist(),
            torch.arange(18, 21, dtype=torch.int64).tolist(),
        )
        self.assertEqual(decode_req.req.kv.kv_committed_len, 5)
        self.assertTrue(decode_req.waiting_for_input)
        self.assertEqual(decode_req.kv_receiver.conclude_state, KVPoll.Success)
        self.assertEqual(
            scheduler.disagg_decode_metadata_buffers.bootstrap_room[0, 0].item(),
            123,
        )
        scheduler.disagg_prefill_metadata_buffers.copy_row.assert_called_once_with(
            0,
            0,
            destination=scheduler.disagg_decode_metadata_buffers,
        )
        self.assertFalse(decode_req.kv_receiver.require_staging)
        scheduler.req_to_token_pool.mamba_pool.copy_from.assert_called_once()
        prefill_req.disagg_kv_sender.kv_mgr.update_status.assert_called_once_with(
            123, KVPoll.Success
        )

    def test_commit_translates_and_moves_kv_in_bounded_chunks(self):
        scheduler = SchedulerDisaggregationPrefillMixin.__new__(
            SchedulerDisaggregationPrefillMixin
        )
        tokens = torch.arange(8194, dtype=torch.int64)
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=tokens.reshape(2, 4097),
            req_index_to_mamba_index_mapping=torch.tensor([[7], [11]]),
            device="cpu",
            translate_mamba_indices=lambda indices: indices,
            mamba_pool=SimpleNamespace(copy_from=MagicMock()),
        )
        kv_cache = SimpleNamespace(move_kv_cache=MagicMock())
        translate_sizes = []

        def translate(indices):
            translate_sizes.append(indices.numel())
            return indices + 100

        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: kv_cache,
            translate_kv_indices_for_transfer=translate,
            full_v2p_page_table=torch.arange(129, dtype=torch.int64),
            page_size=64,
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.disagg_decode_metadata_buffers = SimpleNamespace(
            bootstrap_room=torch.zeros((1, 1), dtype=torch.int64)
        )
        scheduler.disagg_prefill_metadata_buffers = SimpleNamespace(
            copy_row=MagicMock()
        )
        prefill_req = SimpleNamespace(
            rid="prefill-rid",
            bootstrap_room=123,
            metadata_buffer_index=0,
            kv=SimpleNamespace(req_pool_idx=0),
            disagg_decode_prefix_len=0,
            extend_range=SimpleNamespace(end=4097),
            disagg_kv_sender=SimpleNamespace(
                kv_mgr=SimpleNamespace(update_status=MagicMock())
            ),
        )
        decode_req = SimpleNamespace(
            metadata_buffer_index=0,
            req=SimpleNamespace(
                rid="decode-rid",
                bootstrap_room=123,
                kv=SimpleNamespace(req_pool_idx=1),
                set_extend_range=MagicMock(),
                prefix_indices=None,
            ),
            kv_receiver=SimpleNamespace(require_staging=True, conclude_state=None),
            waiting_for_input=False,
        )
        scheduler.disagg_decode_transfer_queue.queue.append(decode_req)

        self.assertTrue(
            SchedulerDisaggregationPrefillMixin._hybrid_commit_local_transfer(
                scheduler, prefill_req
            )
        )

        self.assertEqual(translate_sizes, [1, 1])
        self.assertEqual(kv_cache.move_kv_cache.call_count, 2)
        first_dst, first_src = kv_cache.move_kv_cache.call_args_list[0].args
        self.assertTrue(torch.equal(first_src, torch.arange(4096, dtype=torch.int64)))
        first_dst_virtual = tokens.reshape(2, 4097)[1, :4096]
        expected_first_dst = (
            torch.arange(129, dtype=torch.int64)[(first_dst_virtual - 1) // 64] * 64
            + first_dst_virtual % 64
        )
        self.assertTrue(torch.equal(first_dst, expected_first_dst))
        second_dst, second_src = kv_cache.move_kv_cache.call_args_list[1].args
        self.assertEqual(
            [second_dst.numel(), second_src.numel()],
            [1, 1],
        )
        self.assertEqual(second_src.item(), 4196)
        self.assertEqual(
            second_dst.item(),
            8293,
        )

    def test_commit_uses_stock_translation_for_paged_allocator(self):
        scheduler = SchedulerDisaggregationPrefillMixin.__new__(
            SchedulerDisaggregationPrefillMixin
        )
        tokens = torch.arange(8194, dtype=torch.int64)
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=tokens.reshape(2, 4097),
            req_index_to_mamba_index_mapping=torch.tensor([[7], [11]]),
            device="cpu",
            translate_mamba_indices=lambda indices: indices,
            mamba_pool=SimpleNamespace(copy_from=MagicMock()),
        )
        kv_cache = SimpleNamespace(move_kv_cache=MagicMock())
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: kv_cache,
            translate_kv_indices_for_transfer=lambda indices: indices,
            page_size=64,
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.disagg_decode_metadata_buffers = SimpleNamespace(
            bootstrap_room=torch.zeros((1, 1), dtype=torch.int64)
        )
        scheduler.disagg_prefill_metadata_buffers = SimpleNamespace(
            copy_row=MagicMock()
        )
        prefill_req = SimpleNamespace(
            rid="prefill-rid",
            bootstrap_room=123,
            metadata_buffer_index=0,
            kv=SimpleNamespace(req_pool_idx=0),
            disagg_decode_prefix_len=0,
            extend_range=SimpleNamespace(end=4097),
            disagg_kv_sender=SimpleNamespace(
                kv_mgr=SimpleNamespace(update_status=MagicMock())
            ),
        )
        decode_req = SimpleNamespace(
            metadata_buffer_index=0,
            req=SimpleNamespace(
                rid="decode-rid",
                bootstrap_room=123,
                kv=SimpleNamespace(req_pool_idx=1),
                set_extend_range=MagicMock(),
                prefix_indices=None,
            ),
            kv_receiver=SimpleNamespace(require_staging=True, conclude_state=None),
            waiting_for_input=False,
        )
        scheduler.disagg_decode_transfer_queue.queue.append(decode_req)

        self.assertTrue(
            SchedulerDisaggregationPrefillMixin._hybrid_commit_local_transfer(
                scheduler, prefill_req
            )
        )

        self.assertEqual(kv_cache.move_kv_cache.call_count, 2)
        first_dst, first_src = kv_cache.move_kv_cache.call_args_list[0].args
        self.assertTrue(torch.equal(first_src, tokens[:4096]))
        self.assertTrue(torch.equal(first_dst, tokens[4097:8193]))
        second_dst, second_src = kv_cache.move_kv_cache.call_args_list[1].args
        self.assertEqual(second_src.tolist(), [4096])
        self.assertEqual(second_dst.tolist(), [8193])

    def test_hybrid_prebuilt_admission_does_not_match_shared_prefill_tree(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.HYBRID
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.waiting_queue = []
        scheduler.enable_priority_scheduling = False
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.batch_size.return_value = 0
        scheduler.req_to_token_pool = MagicMock(size=1)
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.model_config = MagicMock()
        scheduler.enable_overlap = False
        scheduler.spec_algorithm = MagicMock()
        scheduler.max_running_requests = 1
        scheduler.future_map = MagicMock()
        scheduler.policy = MagicMock()
        scheduler.schedule_stream = MagicMock()
        scheduler.forward_stream = MagicMock()

        req = MagicMock()
        req.prefix_indices = [1, 2]
        req.kv = SimpleNamespace(kv_committed_len=5)
        scheduler.waiting_queue = [req]
        new_batch = MagicMock()

        with (
            patch(
                "sglang.srt.disaggregation.decode.ScheduleBatch.init_new",
                return_value=new_batch,
            ),
            get_context().override_server_args(
                disaggregation_mode="hybrid",
                disaggregation_decode_enable_radix_cache=False,
            ),
        ):
            ret = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
                scheduler, scheduler.running_batch
            )

        self.assertIs(ret, new_batch)
        req.init_next_round_input.assert_called_once_with(None)

    def test_metadata_copy_row_copies_all_schema_rows(self):
        from sglang.srt.disaggregation.utils import MetadataBuffers

        source = MetadataBuffers(
            size=2,
            hidden_size=3,
            hidden_states_dtype=torch.float32,
            max_top_logprobs_num=4,
            max_sampling_mask_tokens=2,
            output_dsa_topk_indices_dim=2,
        )
        destination = MetadataBuffers(
            size=2,
            hidden_size=3,
            hidden_states_dtype=torch.float32,
            max_top_logprobs_num=4,
            max_sampling_mask_tokens=2,
            output_dsa_topk_indices_dim=2,
        )
        source.output_ids.fill_(11)
        source.cached_tokens.fill_(12)
        source.output_token_logprobs_val.fill_(13.0)
        source.output_token_logprobs_idx.fill_(14)
        source.output_top_logprobs_val.fill_(15.0)
        source.output_top_logprobs_idx.fill_(16)
        source.output_token_sampling_mask_len.fill_(17)
        source.output_token_sampling_mask_idx.fill_(18)
        source.output_token_sampling_logprobs.fill_(19.0)
        source.output_topk_p.fill_(20.0)
        source.output_topk_index.fill_(21)
        source.output_hidden_states.fill_(22.0)
        source.output_dsa_topk_indices.fill_(23)
        source.bootstrap_room.fill_(24)

        source.copy_row(1, 0, destination=destination)

        self.assertTrue(torch.equal(destination.output_ids[0], source.output_ids[1]))
        self.assertTrue(
            torch.equal(destination.cached_tokens[0], source.cached_tokens[1])
        )
        self.assertTrue(
            torch.equal(
                destination.output_token_logprobs_val[0],
                source.output_token_logprobs_val[1],
            )
        )
        self.assertTrue(
            torch.equal(
                destination.output_token_logprobs_idx[0],
                source.output_token_logprobs_idx[1],
            )
        )
        self.assertTrue(
            torch.equal(
                destination.output_top_logprobs_val[0],
                source.output_top_logprobs_val[1],
            )
        )
        self.assertTrue(
            torch.equal(
                destination.output_top_logprobs_idx[0],
                source.output_top_logprobs_idx[1],
            )
        )
        self.assertTrue(
            torch.equal(
                destination.output_token_sampling_mask_len[0],
                source.output_token_sampling_mask_len[1],
            )
        )
        self.assertTrue(
            torch.equal(
                destination.output_token_sampling_mask_idx[0],
                source.output_token_sampling_mask_idx[1],
            )
        )
        self.assertTrue(
            torch.equal(
                destination.output_token_sampling_logprobs[0],
                source.output_token_sampling_logprobs[1],
            )
        )
        self.assertTrue(
            torch.equal(destination.output_topk_p[0], source.output_topk_p[1])
        )
        self.assertTrue(
            torch.equal(destination.output_topk_index[0], source.output_topk_index[1])
        )
        self.assertTrue(
            torch.equal(
                destination.output_hidden_states[0], source.output_hidden_states[1]
            )
        )
        self.assertTrue(
            torch.equal(
                destination.output_dsa_topk_indices[0],
                source.output_dsa_topk_indices[1],
            )
        )
        self.assertTrue(
            torch.equal(destination.bootstrap_room[0], source.bootstrap_room[1])
        )


class TestHybridBootstrapService(unittest.TestCase):
    def test_hybrid_starts_bootstrap_server(self):
        from sglang.srt.managers import disagg_service

        server = object()
        with (
            patch(
                "sglang.srt.managers.disagg_service.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_mode="hybrid",
                    disaggregation_transfer_backend="mori",
                    disaggregation_bootstrap_port=32500,
                ),
            ),
            patch(
                "sglang.srt.managers.disagg_service.get_serving",
                return_value=SimpleNamespace(host="0.0.0.0"),
            ),
            patch(
                "sglang.srt.managers.disagg_service.get_kv_class",
                return_value=lambda *args, **kwargs: server,
            ),
            patch(
                "sglang.srt.managers.disagg_service.maybe_create_ascend_config_store"
            ) as maybe_create,
        ):
            self.assertIs(disagg_service.start_disagg_service(), server)

        maybe_create.assert_called_once_with(
            transfer_backend=disagg_service.TransferBackend.MORI
        )


class TestHybridOpenAIRolePropagation(unittest.TestCase):
    def test_chat_completion_preserves_disagg_role(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "disagg_role": "decode",
            }
        )

        self.assertEqual(request.disagg_role, "decode")

    def test_completion_preserves_disagg_role(self):
        from sglang.srt.entrypoints.openai.protocol import CompletionRequest

        request = CompletionRequest.model_validate(
            {
                "prompt": "hello",
                "disagg_role": "prefill",
            }
        )

        self.assertEqual(request.disagg_role, "prefill")


class TestHybridWaitingQueueIsolation(unittest.TestCase):
    def test_role_planners_only_see_their_own_waiting_requests(self):
        prefill_req = SimpleNamespace(rid="p1", disagg_role="prefill")
        decode_req = SimpleNamespace(rid="d1", disagg_role="decode")
        scheduler = SimpleNamespace(
            gracefully_exit=False,
            _engine_paused=False,
            chunked_req=None,
            disagg_decode_prealloc_queue=MagicMock(),
            request_receiver=MagicMock(),
            disagg_prefill_bootstrap_queue=MagicMock(),
            ngram_embedding_manager=MagicMock(),
            process_disagg_prefill_inflight_queue=MagicMock(),
            on_idle=MagicMock(),
            run_batch=MagicMock(),
            process_batch_result=MagicMock(),
        )
        scheduler.request_receiver.recv_requests.return_value = []
        scheduler.disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries = (
            MagicMock()
        )
        scheduler.disagg_prefill_bootstrap_queue.pop_bootstrapped.return_value = [
            prefill_req
        ]

        seen_by_decode = []
        seen_by_prefill = []
        decode_calls = 0
        prefill_calls = 0

        def fake_process_input_requests(_reqs):
            scheduler.gracefully_exit = True

        def fake_process_decode_queue():
            scheduler.waiting_queue.append(decode_req)

        def fake_get_next_disagg_decode_batch_to_run(running_batch):
            nonlocal decode_calls
            decode_calls += 1
            seen_by_decode.extend(scheduler.waiting_queue)
            return SimpleNamespace(batch_to_run=None, running_batch=running_batch)

        def fake_get_next_disagg_prefill_batch_to_run(running_batch, last_batch):
            nonlocal prefill_calls
            prefill_calls += 1
            seen_by_prefill.extend(scheduler.waiting_queue)
            return SimpleNamespace(batch_to_run=None, running_batch=running_batch)

        scheduler.process_input_requests = fake_process_input_requests
        scheduler.process_decode_queue = fake_process_decode_queue
        scheduler.get_next_disagg_decode_batch_to_run = (
            fake_get_next_disagg_decode_batch_to_run
        )
        scheduler.get_next_disagg_prefill_batch_to_run = (
            fake_get_next_disagg_prefill_batch_to_run
        )

        SchedulerDisaggregationPrefillMixin.event_loop_normal_disagg_hybrid(scheduler)

        self.assertEqual(decode_calls, 1)
        self.assertEqual(prefill_calls, 1)
        self.assertEqual([req.rid for req in seen_by_decode], ["d1"])
        self.assertEqual([req.rid for req in seen_by_prefill], ["p1"])
        scheduler.process_disagg_prefill_inflight_queue.assert_called_once()


def _hybrid_abort_scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = DisaggregationMode.HYBRID
    scheduler.waiting_queue = []
    scheduler.hybrid_waiting_queue_prefill = []
    scheduler.hybrid_waiting_queue_decode = []
    scheduler.chunked_req = None
    scheduler._pending_chunked_abort_req = None
    scheduler.hybrid_chunked_req_prefill = None
    scheduler.hybrid_chunked_req_decode = None
    scheduler.mm_receiver = None
    scheduler._release_aborted_request = MagicMock()
    scheduler.beam_coordinator = SimpleNamespace(retire_group=MagicMock())
    scheduler.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
    )
    scheduler.dllm_config = None
    scheduler.grammar_manager = MagicMock()
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
    scheduler.disagg_prefill_inflight_queue = []
    scheduler.disagg_prefill_req_to_metadata_buffer_idx_allocator = MagicMock()
    scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
    scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
        queue=[], retracted_queue=[]
    )
    scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
    scheduler.tree_cache = MagicMock()
    scheduler.req_to_token_pool = MagicMock()
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.ps = SimpleNamespace(pp_size=1)
    scheduler.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
    scheduler.last_batch = None
    scheduler.enable_overlap = False
    scheduler.enable_staging = False
    scheduler.hisparse_coordinator = None
    scheduler.metrics_reporter = SimpleNamespace(
        last_gen_throughput=1.0,
        current_scheduler_metrics_enabled=False,
    )
    scheduler.kv_events_publisher = SimpleNamespace(publish_kv_events=MagicMock())
    scheduler._engine_paused = False
    scheduler.enable_hisparse = False
    return scheduler


def _fake_req(rid, role, metadata_index=-1, pending_bootstrap=False):
    req = SimpleNamespace(
        rid=rid,
        disagg_role=role,
        metadata_buffer_index=metadata_index,
        pending_bootstrap=pending_bootstrap,
        finished=lambda: False,
        kv=SimpleNamespace(holds_mamba=False, holds_kv=False, is_kv_released=True),
        weight_version_events=[],
    )
    req.disagg_kv_sender = SimpleNamespace(abort=MagicMock())
    req.output_ids = []
    req.pd_rebootstrap_forced_output_id = None
    req.pd_rebootstrap_in_progress = False
    req.time_stats = SimpleNamespace(set_retract_time=MagicMock())
    return req


class TestHybridAbortVisibility(unittest.TestCase):
    def test_abort_scans_both_role_waiting_queues(self):
        scheduler = _hybrid_abort_scheduler()
        p_req = _fake_req("p1", "prefill")
        d_req = _fake_req("d1", "decode")
        scheduler.hybrid_waiting_queue_prefill = [p_req]
        scheduler.hybrid_waiting_queue_decode = [d_req]

        with patch(
            "sglang.srt.managers.scheduler._make_abort_req", return_value=MagicMock()
        ):
            scheduler.abort_request(SimpleNamespace(rid="", abort_all=True))

        self.assertEqual(scheduler.hybrid_waiting_queue_prefill, [])
        self.assertEqual(scheduler.hybrid_waiting_queue_decode, [])
        self.assertEqual(scheduler._release_aborted_request.call_count, 2)


class TestDedicatedDecodeLifecycle(unittest.TestCase):
    def test_abort_without_request_role_releases_kv_once(self):
        for holds_mamba in (False, True):
            with self.subTest(holds_mamba=holds_mamba):
                scheduler = _hybrid_abort_scheduler()
                scheduler.disaggregation_mode = DisaggregationMode.DECODE
                req = _fake_req("d1", None)
                req.kv.holds_mamba = holds_mamba
                scheduler.waiting_queue = [req]

                with (
                    patch(
                        "sglang.srt.managers.scheduler._make_abort_req",
                        return_value=MagicMock(),
                    ),
                    patch(
                        "sglang.srt.managers.scheduler.release_kv_cache"
                    ) as release_kv_cache,
                ):
                    scheduler.abort_request(SimpleNamespace(rid="d1", abort_all=False))

                self.assertEqual(scheduler.waiting_queue, [])
                release_kv_cache.assert_called_once_with(req, scheduler.tree_cache)

    def test_pause_without_request_role_holds_rebootstrap_until_continue(self):
        scheduler = _hybrid_abort_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[],
            retracted_queue=[],
            hold_rebootstrap=MagicMock(),
            enqueue_held_rebootstrap=MagicMock(),
        )
        scheduler._add_request_to_queue = MagicMock()
        req = _fake_req("d1", None)
        scheduler.running_batch = ScheduleBatch(reqs=[req], batch_is_full=False)

        with patch("sglang.srt.managers.scheduler.retract_all"):
            scheduler.pause_generation(SimpleNamespace(mode="retract"))

        scheduler.disagg_decode_prealloc_queue.hold_rebootstrap.assert_called_once_with(
            req
        )
        scheduler._add_request_to_queue.assert_not_called()
        self.assertTrue(req.pd_rebootstrap_in_progress)
        scheduler.continue_generation(SimpleNamespace(torch_empty_cache=False))
        scheduler.disagg_decode_prealloc_queue.enqueue_held_rebootstrap.assert_called_once()
        self.assertFalse(scheduler._engine_paused)


class TestHybridDecodeFailureReleasesPairedPrefill(unittest.TestCase):
    def _scheduler(self, prefill_req):
        scheduler = _hybrid_abort_scheduler()
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[prefill_req])
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.disagg_prefill_pending_chunk_rids = {prefill_req.rid}
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_collector = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.metrics_reporter.enable_metrics = False
        return scheduler

    def _prefill_req(self, room):
        req = _fake_req("paired-prefill", "prefill", metadata_index=11)
        req.bootstrap_room = room
        req.kv = SimpleNamespace(holds_kv=True, holds_mamba=True, is_kv_released=False)
        req.finished_reason = None
        req.return_logprob = False
        req.effective_kv_committed_len = lambda: 0
        req.kv.kv_allocated_len = 0
        req.time_stats = SimpleNamespace(trace_ctx=SimpleNamespace(abort=MagicMock()))
        return req

    def test_decode_transfer_failure_releases_paired_prefill(self):
        room = 4242
        prefill_req = self._prefill_req(room)
        scheduler = self._scheduler(prefill_req)
        decode_req = SimpleNamespace(
            req=SimpleNamespace(
                rid="paired-decode", bootstrap_room=room, return_logprob=False
            ),
            kv_receiver=SimpleNamespace(
                failure_exception=lambda: None,
                clear=lambda: None,
            ),
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_deferred_kv_release = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()
        queue.scheduler = scheduler

        with (
            patch("sglang.srt.disaggregation.decode.release_kv_cache") as mock_release,
            patch(
                "sglang.srt.disaggregation.decode.prepare_abort"
            ) as mock_prepare_abort,
            patch("sglang.srt.disaggregation.decode.poll_and_all_reduce") as mock_poll,
            patch(
                "sglang.srt.disaggregation.prefill.release_kv_cache"
            ) as mock_prefill_release,
        ):
            mock_poll.return_value = [KVPoll.Failed]
            self.assertEqual(queue.pop_transferred(), [])

        self.assertEqual(scheduler.disagg_prefill_bootstrap_queue.queue, [])
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        self.assertNotIn("paired-prefill", scheduler.disagg_prefill_pending_chunk_rids)
        prefill_req.disagg_kv_sender.abort.assert_called_once()
        mock_prefill_release.assert_called_once_with(
            prefill_req, scheduler.tree_cache, is_insert=False
        )
        scheduler.disagg_prefill_req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(
            11
        )
        self.assertEqual(prefill_req.pending_bootstrap, False)
        self.assertEqual(scheduler.output_streamer.stream_output.call_count, 2)
        mock_release.assert_called_once()
        mock_prefill_release.assert_called_once()
        mock_prepare_abort.assert_called_once()


class TestHybridChunkedIsolation(unittest.TestCase):
    def test_decode_planner_does_not_see_prefill_chunk(self):
        prefill_req = SimpleNamespace(rid="p1", disagg_role="prefill")
        decode_req = SimpleNamespace(rid="d1", disagg_role="decode")
        scheduler = SimpleNamespace(
            gracefully_exit=False,
            _engine_paused=False,
            chunked_req=None,
            hybrid_waiting_queue_prefill=[],
            hybrid_waiting_queue_decode=[],
            hybrid_running_batch_prefill=None,
            hybrid_running_batch_decode=None,
            hybrid_chunked_req_prefill=SimpleNamespace(rid="cp"),
            hybrid_chunked_req_decode=None,
            disagg_decode_prealloc_queue=MagicMock(),
            request_receiver=MagicMock(),
            disagg_prefill_bootstrap_queue=MagicMock(),
            ngram_embedding_manager=MagicMock(),
            process_disagg_prefill_inflight_queue=MagicMock(),
            on_idle=MagicMock(),
            run_batch=MagicMock(),
            process_batch_result=MagicMock(),
        )
        scheduler.request_receiver.recv_requests.return_value = []
        scheduler.disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries = (
            MagicMock()
        )
        scheduler.disagg_prefill_bootstrap_queue.pop_bootstrapped.return_value = [
            prefill_req
        ]
        seen_decode_chunked = []
        seen_decode = []

        def fake_process_input_requests(_reqs):
            scheduler.gracefully_exit = True

        def fake_process_decode_queue():
            scheduler.waiting_queue.append(decode_req)

        def fake_get_next_disagg_decode_batch_to_run(running_batch):
            seen_decode_chunked.append(scheduler.chunked_req)
            seen_decode.extend(scheduler.waiting_queue)
            return SimpleNamespace(batch_to_run=None, running_batch=running_batch)

        def fake_get_next_disagg_prefill_batch_to_run(running_batch, last_batch):
            scheduler.chunked_req = scheduler.hybrid_chunked_req_prefill
            return SimpleNamespace(batch_to_run=None, running_batch=running_batch)

        scheduler.process_input_requests = fake_process_input_requests
        scheduler.process_decode_queue = fake_process_decode_queue
        scheduler.get_next_disagg_decode_batch_to_run = (
            fake_get_next_disagg_decode_batch_to_run
        )
        scheduler.get_next_disagg_prefill_batch_to_run = (
            fake_get_next_disagg_prefill_batch_to_run
        )

        SchedulerDisaggregationPrefillMixin.event_loop_normal_disagg_hybrid(scheduler)

        self.assertTrue(all(chunk is None for chunk in seen_decode_chunked))
        self.assertEqual([req.rid for req in seen_decode], ["d1"])
        self.assertIs(scheduler.hybrid_chunked_req_prefill.rid, "cp")


class TestHybridPauseRetractContinue(unittest.TestCase):
    def test_pause_retract_decode_and_continue(self):
        scheduler = _hybrid_abort_scheduler()
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[],
            retracted_queue=[],
            hold_rebootstrap=MagicMock(),
            enqueue_held_rebootstrap=MagicMock(),
        )
        scheduler._add_request_to_queue = MagicMock()
        decode_req = _fake_req("d1", "decode")
        decode_req.finished = lambda: False
        scheduler.hybrid_running_batch_decode = ScheduleBatch(
            reqs=[decode_req], batch_is_full=False
        )
        scheduler.hybrid_chunked_req_decode = None

        with patch("sglang.srt.managers.scheduler.retract_all"):
            scheduler.pause_generation(SimpleNamespace(mode="retract"))

        self.assertIs(scheduler._engine_paused, True)
        scheduler.disagg_decode_prealloc_queue.hold_rebootstrap.assert_called_once_with(
            decode_req
        )
        scheduler._add_request_to_queue.assert_not_called()
        self.assertIsNotNone(scheduler.hybrid_running_batch_decode)

        scheduler.continue_generation(SimpleNamespace(torch_empty_cache=False))
        scheduler.disagg_decode_prealloc_queue.enqueue_held_rebootstrap.assert_called_once()
        self.assertIs(scheduler._engine_paused, False)


if __name__ == "__main__":
    unittest.main()
