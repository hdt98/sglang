"""Unit tests for the DSpark verify pre-plan stream (opt-in).

Covers the orchestration contract of the surgical verify-preplan extension:

1. Feature off (default): _forward_prepared_verify is byte-identical to the
   pre-patch behavior -- no stream APIs touched at all.
2. Feature on: prep runs inside the plan-stream context, ordered behind the
   pre-draft frontier event (or the caller stream when no event is given),
   joined on the caller stream before the verify forward launch.
3. record_stream marks the plan-stream-read tensors (EAGLE-v2 contract) and
   skips None / non-CUDA entries.
4. The compact/ragged path (_run_ragged) never passes a plan stream: its prep
   reads the confidence-derived ragged layout and stays on the caller stream.
"""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
    _record_plan_stream_reads,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


def _make_executor(plan_stream=None, plan_stream_ctx=None):
    return TargetVerifyExecutor(
        target_worker=SimpleNamespace(name="target"),
        gamma=5,
        verify_num_draft_tokens=6,
        model_runner=SimpleNamespace(device="cpu"),
        kv_injector=SimpleNamespace(),
        tp_sync=SimpleNamespace(),
        verify_epilogue=None,
        simulate_acc_len=0.0,
        plan_stream=plan_stream,
        plan_stream_ctx=plan_stream_ctx,
    )


def _fake_verify_input():
    return SimpleNamespace(
        prepare_for_verify=Mock(
            return_value=(SimpleNamespace(name="verify_forward_batch"), True)
        ),
        draft_token=torch.zeros(4, dtype=torch.int64),
        positions=torch.zeros(4, dtype=torch.int64),
        live_seq_lens_cpu=torch.tensor([5, 6]),
    )


def _fake_batch():
    return SimpleNamespace(
        seq_lens=torch.zeros(2, dtype=torch.int64),
        req_pool_indices=torch.zeros(2, dtype=torch.int64),
        out_cache_loc=torch.zeros(4, dtype=torch.int64),
        seq_lens_cpu=torch.tensor([5, 6]),
        seq_lens_sum=11,
    )


@contextmanager
def _recording_ctx(log, label):
    log.append(f"{label}:enter")
    try:
        yield
    finally:
        log.append(f"{label}:exit")


class TestPlanStreamDisabledPath(CustomTestCase):
    def test_default_executor_holds_no_stream(self):
        executor = _make_executor()
        self.assertIsNone(executor._plan_stream)
        self.assertIsInstance(executor._plan_stream_ctx, type(nullcontext()))

    def test_disabled_prep_never_touches_stream_apis(self):
        executor = _make_executor()
        batch = _fake_batch()
        verify_input = _fake_verify_input()
        calls = []
        executor.target_worker.forward_batch_generation = Mock(
            side_effect=lambda **kw: calls.append("forward") or SimpleNamespace(
                logits_output="logits", can_run_cuda_graph=True
            )
        )
        verify_input.prepare_for_verify.side_effect = (
            lambda *a: calls.append("prep")
            or (SimpleNamespace(name="fb"), True)
        )
        with patch(
            "torch.get_device_module",
            side_effect=AssertionError("stream API used on the disabled path"),
        ):
            result = executor._forward_prepared_verify(
                batch=batch,
                verify_input=verify_input,
                seq_lens_cpu_backup=batch.seq_lens_cpu,
                seq_lens_sum_backup=batch.seq_lens_sum,
            )
        self.assertEqual(calls, ["prep", "forward"])
        self.assertEqual(result.logits_output, "logits")
        self.assertTrue(result.can_run_cuda_graph)
        # Host state restore is unchanged.
        self.assertEqual(int(batch.seq_lens_cpu[0]), 5)
        self.assertEqual(batch.seq_lens_sum, 11)


class TestPlanStreamPrepOrder(CustomTestCase):
    def test_prep_runs_on_plan_stream_and_joins_before_forward(self):
        log = []
        plan_stream = SimpleNamespace(
            wait_event=lambda ev: log.append("plan:wait_event"),
            wait_stream=lambda s: log.append("plan:wait_stream"),
        )
        executor = _make_executor(
            plan_stream=plan_stream,
            plan_stream_ctx=_recording_ctx(log, "plan_ctx"),
        )
        batch = _fake_batch()
        verify_input = _fake_verify_input()
        verify_input.prepare_for_verify.side_effect = (
            lambda *a: log.append("prep")
            or (SimpleNamespace(name="fb"), True)
        )
        executor.target_worker.forward_batch_generation = Mock(
            side_effect=lambda **kw: log.append("forward")
            or SimpleNamespace(logits_output="logits", can_run_cuda_graph=True)
        )
        fake_caller = SimpleNamespace(
            wait_stream=lambda s: log.append("caller:join")
        )
        fake_event = SimpleNamespace(name="pre-draft-frontier")
        fake_module = SimpleNamespace(current_stream=lambda: fake_caller)
        with patch("torch.get_device_module", return_value=fake_module):
            executor._forward_prepared_verify(
                batch=batch,
                verify_input=verify_input,
                seq_lens_cpu_backup=batch.seq_lens_cpu,
                seq_lens_sum_backup=batch.seq_lens_sum,
                plan_stream=plan_stream,
                plan_stream_ctx=executor._plan_stream_ctx,
                plan_ready_event=fake_event,
            )
        self.assertEqual(
            log,
            [
                "plan_ctx:enter",
                "plan:wait_event",
                "prep",
                "plan_ctx:exit",
                "caller:join",
                "forward",
            ],
        )
        # The verify forward launches after the join, on the caller stream.
        kwargs = executor.target_worker.forward_batch_generation.call_args.kwargs
        self.assertTrue(kwargs["is_verify"])
        self.assertEqual(kwargs["skip_attn_backend_init"], True)
        self.assertEqual(kwargs["forward_batch"].name, "fb")

    def test_fallback_orders_behind_caller_stream_without_event(self):
        log = []
        plan_stream = SimpleNamespace(
            wait_event=lambda ev: log.append("plan:wait_event"),
            wait_stream=lambda s: log.append("plan:wait_stream"),
        )
        executor = _make_executor(
            plan_stream=plan_stream,
            plan_stream_ctx=_recording_ctx(log, "plan_ctx"),
        )
        batch = _fake_batch()
        verify_input = _fake_verify_input()
        verify_input.prepare_for_verify.side_effect = (
            lambda *a: log.append("prep")
            or (SimpleNamespace(name="fb"), True)
        )
        executor.target_worker.forward_batch_generation = Mock(
            side_effect=lambda **kw: log.append("forward")
            or SimpleNamespace(logits_output="logits", can_run_cuda_graph=True)
        )
        fake_caller = SimpleNamespace(
            wait_stream=lambda s: log.append("caller:join")
        )
        fake_module = SimpleNamespace(current_stream=lambda: fake_caller)
        with patch("torch.get_device_module", return_value=fake_module):
            executor._forward_prepared_verify(
                batch=batch,
                verify_input=verify_input,
                seq_lens_cpu_backup=batch.seq_lens_cpu,
                seq_lens_sum_backup=batch.seq_lens_sum,
                plan_stream=plan_stream,
                plan_stream_ctx=executor._plan_stream_ctx,
            )
        self.assertIn("plan:wait_stream", log)
        self.assertNotIn("plan:wait_event", log)
        self.assertLess(
            log.index("plan:wait_stream"), log.index("prep")
        )


class TestRecordPlanStreamReads(CustomTestCase):
    def test_records_only_cuda_tensors_and_skips_none(self):
        plan_stream = SimpleNamespace(name="plan")
        gpu_like = SimpleNamespace(is_cuda=True, record_stream=Mock())
        _record_plan_stream_reads(
            plan_stream,
            (None, torch.zeros(2), gpu_like),
        )
        gpu_like.record_stream.assert_called_once_with(plan_stream)

    def test_noop_without_plan_stream(self):
        gpu_like = SimpleNamespace(is_cuda=True, record_stream=Mock())
        _record_plan_stream_reads(None, (gpu_like,))
        gpu_like.record_stream.assert_not_called()


class TestRaggedPathStaysOnCallerStream(CustomTestCase):
    def test_run_ragged_never_passes_plan_stream(self):
        executor = _make_executor(
            plan_stream=SimpleNamespace(name="plan"),
            plan_stream_ctx=nullcontext(),
        )
        captured = {}

        def fake_forward_prepared_verify(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                logits_output="logits", can_run_cuda_graph=True
            )

        executor._forward_prepared_verify = fake_forward_prepared_verify
        batch = SimpleNamespace(
            out_cache_loc=None,
            seq_lens_cpu=torch.tensor([5, 6]),
            seq_lens_sum=11,
        )
        layout = SimpleNamespace(verify_lens_cpu=[2, 3])
        ragged_window = SimpleNamespace(
            verify_ids=torch.zeros(4, dtype=torch.int64),
            positions=torch.zeros(4, dtype=torch.int64),
            verify_cache_loc=torch.zeros(4, dtype=torch.int64),
        )
        executor._run_ragged(
            batch=batch,
            layout=layout,
            ragged_window=ragged_window,
            sampling_info=None,
        )
        self.assertIsNone(captured.get("plan_stream"))
        self.assertIsNone(captured.get("plan_ready_event"))


class TestEnvDefault(CustomTestCase):
    def test_verify_preplan_stream_defaults_off(self):
        self.assertFalse(envs.SGLANG_DSPARK_VERIFY_PREPLAN_STREAM.get())


class TestReusableFrontierEvent(CustomTestCase):
    """The frontier event is allocated once in __init__, not per decode step.

    _forward_decode must record the stored event; per-step Event()
    construction is a host-side allocation on the hot decode path.
    """

    @staticmethod
    def _bare_worker(plan_stream, plan_ready_event):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._plan_stream = plan_stream
        worker._plan_ready_event = plan_ready_event
        return worker

    def test_records_stored_event_without_per_step_allocation(self):
        stored_event = Mock(name="plan_ready_event")
        worker = self._bare_worker(
            plan_stream=SimpleNamespace(name="plan"),
            plan_ready_event=stored_event,
        )
        with patch(
            "torch.get_device_module",
            side_effect=AssertionError("per-step Event allocation on the hot path"),
        ):
            first = worker._plan_frontier_event()
            second = worker._plan_frontier_event()
        # Multiple forwards reuse the one stored event; no device-module or
        # Event construction happens on the step path (patch would raise).
        self.assertIs(first, stored_event)
        self.assertIs(second, stored_event)
        self.assertEqual(stored_event.record.call_count, 2)

    def test_no_frontier_event_when_plan_stream_disabled(self):
        worker = self._bare_worker(plan_stream=None, plan_ready_event=None)
        with patch(
            "torch.get_device_module",
            side_effect=AssertionError("stream API used when disabled"),
        ):
            self.assertIsNone(worker._plan_frontier_event())


class TestPlanStreamHardening(CustomTestCase):
    """Exception-safe join, record-set completeness, and A+B ordering."""

    def test_prep_failure_still_joins_caller_stream(self):
        log = []
        plan_stream = SimpleNamespace(
            wait_event=lambda ev: log.append("plan:wait_event"),
            wait_stream=lambda s: log.append("plan:wait_stream"),
        )
        executor = _make_executor(
            plan_stream=plan_stream,
            plan_stream_ctx=_recording_ctx(log, "plan_ctx"),
        )
        batch = _fake_batch()
        verify_input = _fake_verify_input()
        verify_input.prepare_for_verify.side_effect = RuntimeError("prep boom")
        fake_caller = SimpleNamespace(
            wait_stream=lambda s: log.append("caller:join")
        )
        with patch(
            "torch.get_device_module",
            return_value=SimpleNamespace(current_stream=lambda: fake_caller),
        ):
            with self.assertRaises(RuntimeError):
                executor._forward_prepared_verify(
                    batch=batch,
                    verify_input=verify_input,
                    seq_lens_cpu_backup=batch.seq_lens_cpu,
                    seq_lens_sum_backup=batch.seq_lens_sum,
                    plan_stream=plan_stream,
                    plan_stream_ctx=executor._plan_stream_ctx,
                    plan_ready_event=SimpleNamespace(name="frontier"),
                    record_extra=(torch.zeros(2),),
                )
        # The finally-block join runs after the plan ctx exits, even though
        # prepare_for_verify raised, so the plan stream is never orphaned.
        self.assertEqual(
            log,
            ["plan_ctx:enter", "plan:wait_event", "plan_ctx:exit", "caller:join"],
        )

    def test_record_extra_pins_window_originals_and_out_cache_alias(self):
        executor = _make_executor()
        executor._verify_backend_self_adds_seq_lens = lambda: True
        captured = {}

        def fake_forward_prepared_verify(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                logits_output="logits", can_run_cuda_graph=True
            )

        executor._forward_prepared_verify = fake_forward_prepared_verify
        verify_ids_2d = torch.zeros((2, 6), dtype=torch.int64)
        positions_2d = torch.zeros((2, 6), dtype=torch.int64)
        window = SimpleNamespace(
            positions_2d=positions_2d,
            verify_cache_loc=torch.zeros(12, dtype=torch.int64),
        )
        batch = _fake_batch()
        executor.run_non_compact(
            batch=batch,
            draft_input=SimpleNamespace(nxt_kv_lens_cpu=None),
            verify_ids_2d=verify_ids_2d,
            verify_window=window,
            sampling_info=None,
        )
        # Aliases: batch.out_cache_loc IS the window's verify_cache_loc
        # object; the flattened views share storage with the _2d originals.
        self.assertIs(batch.out_cache_loc, window.verify_cache_loc)
        self.assertIs(captured["record_extra"][0], verify_ids_2d)
        self.assertIs(captured["record_extra"][1], window.verify_cache_loc)
        self.assertIs(captured["record_extra"][2], positions_2d)

    def test_ordering_invariant_with_arm_a_enabled(self):
        # A+B: with the draft-side overlap env (Arm A) on, the verify
        # pre-plan orchestration (Arm B) is unchanged and never consults
        # Arm A's stream factory.
        log = []
        plan_stream = SimpleNamespace(
            wait_event=lambda ev: log.append("plan:wait_event"),
            wait_stream=lambda s: log.append("plan:wait_stream"),
        )
        executor = _make_executor(
            plan_stream=plan_stream,
            plan_stream_ctx=_recording_ctx(log, "plan_ctx"),
        )
        batch = _fake_batch()
        verify_input = _fake_verify_input()
        verify_input.prepare_for_verify.side_effect = (
            lambda *a: log.append("prep")
            or (SimpleNamespace(name="fb"), True)
        )
        executor.target_worker.forward_batch_generation = Mock(
            side_effect=lambda **kw: log.append("forward")
            or SimpleNamespace(logits_output="logits", can_run_cuda_graph=True)
        )
        fake_caller = SimpleNamespace(
            wait_stream=lambda s: log.append("caller:join")
        )
        with patch.object(
            envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM, "get", return_value=True
        ), patch(
            "sglang.srt.speculative.spec_utils.get_plan_stream",
            side_effect=AssertionError("Arm B must not consult Arm A's factory"),
        ):
            with patch(
                "torch.get_device_module",
                return_value=SimpleNamespace(current_stream=lambda: fake_caller),
            ):
                executor._forward_prepared_verify(
                    batch=batch,
                    verify_input=verify_input,
                    seq_lens_cpu_backup=batch.seq_lens_cpu,
                    seq_lens_sum_backup=batch.seq_lens_sum,
                    plan_stream=plan_stream,
                    plan_stream_ctx=executor._plan_stream_ctx,
                    plan_ready_event=SimpleNamespace(name="frontier"),
                )
        self.assertEqual(
            log,
            [
                "plan_ctx:enter",
                "plan:wait_event",
                "prep",
                "plan_ctx:exit",
                "caller:join",
                "forward",
            ],
        )


if __name__ == "__main__":
    unittest.main()
