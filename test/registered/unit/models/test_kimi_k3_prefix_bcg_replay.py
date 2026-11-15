"""Kimi K3 capture-safe prefix-BCG tests.

The CPU tests cover flag scope, dispatch, replay admission, and the BCG break
boundary with a mocked eager MHA core. AITER kernel parity is covered by the
device gate.
"""

import os
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch
from sglang.srt.models import kimi_k3
from sglang.srt.models.deepseek_common import attention_backend_handler as abh
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha as fm
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import DeepseekMHAForwardMixin
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _config():
    return SimpleNamespace(
        architectures=["KimiK3ForConditionalGeneration"],
        hidden_size=4,
        num_attention_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=4,
        rms_norm_eps=1e-5,
        mla_use_output_gate=False,
    )


def _attention(*, dcp_size: int = 1):
    def fake_base_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.dcp_size = dcp_size
        self.o_proj = torch.nn.Identity()
        self.all_reduce_fusion = False
        self.use_output_gate = False

    with mock.patch.object(
        kimi_k3.DeepseekV2AttentionMLA, "__init__", fake_base_init
    ):
        return kimi_k3.KimiK3MLAAttention(_config(), layer_idx=0)


class _RecordingCapture:
    def __init__(self):
        self.events = []
        self.graph = bcg.BreakableCUDAGraph()
        self.capture = SimpleNamespace(
            cuda_graph=self.graph,
            _end_current_segment=lambda: self.events.append("end"),
            _begin_new_segment=lambda: self.events.append("begin"),
        )

    def __enter__(self):
        self.token = bcg._current_capture_var.set(self.capture)
        return self

    def __exit__(self, *args):
        bcg._current_capture_var.reset(self.token)


class TestKimiK3PrefixBcgReplay(CustomTestCase):
    def test_dcp1_enables_capture_safe_route(self):
        with (
            mock.patch.object(kimi_k3, "_is_hip", True),
            envs.SGLANG_ENABLE_K3_PREFIX_BCG_REPLAY.override(True),
        ):
            attention = _attention(dcp_size=1)

        self.assertEqual(attention.dcp_size, 1)
        self.assertTrue(attention.prefix_bcg_capture_safe)

    def test_route_is_off_by_default(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(kimi_k3, "_is_hip", True),
        ):
            attention = _attention(dcp_size=1)

        self.assertFalse(attention.prefix_bcg_capture_safe)

    def test_route_requires_hip_and_dcp1(self):
        for hip, dcp_size in ((False, 1), (True, 2)):
            with (
                self.subTest(hip=hip, dcp_size=dcp_size),
                mock.patch.object(kimi_k3, "_is_hip", hip),
                envs.SGLANG_ENABLE_K3_PREFIX_BCG_REPLAY.override(True),
            ):
                attention = _attention(dcp_size=dcp_size)
                self.assertFalse(attention.prefix_bcg_capture_safe)

    def test_dcp1_dispatches_aiter_bcg_to_chunked_mha(self):
        with (
            mock.patch.object(kimi_k3, "_is_hip", True),
            envs.SGLANG_ENABLE_K3_PREFIX_BCG_REPLAY.override(True),
        ):
            attention = _attention(dcp_size=1)
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False))

        with (
            mock.patch.object(abh, "_is_hip", True),
            mock.patch.object(
                abh, "is_in_tc_piecewise_cuda_graph", return_value=False
            ),
            mock.patch.object(abh, "is_in_breakable_cuda_graph", return_value=True),
        ):
            method = abh.handle_attention_aiter(attention, batch)
            resolved = abh.resolve_rocm_forward_method(method)

        self.assertEqual(method, AttnForwardMethod.MHA_CHUNKED_KV)
        self.assertEqual(resolved, AttnForwardMethod.MHA_CHUNKED_KV_ROCM)

    def test_default_aiter_bcg_stays_on_normal_mha(self):
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False))
        attention = SimpleNamespace(prefix_bcg_capture_safe=False)

        with (
            mock.patch.object(abh, "_is_hip", True),
            mock.patch.object(
                abh, "is_in_tc_piecewise_cuda_graph", return_value=False
            ),
            mock.patch.object(abh, "is_in_breakable_cuda_graph", return_value=True),
        ):
            method = abh.handle_attention_aiter(attention, batch)

        self.assertEqual(method, AttnForwardMethod.MHA)


class TestPrefixBcgBreakBoundary(CustomTestCase):
    def _layer(self):
        layer = DeepseekMHAForwardMixin.__new__(DeepseekMHAForwardMixin)
        layer.prefix_bcg_capture_safe = True
        layer.backend = SimpleNamespace(
            prepare_chunked_prefill_qkv=None,
            pack_prefix_chunk_kv=None,
            fuse_prefix_into_extend=False,
            pack_all_prefix_chunks=False,
        )
        layer.fresh_output = torch.randn(2, 3, 4)
        layer.fresh_lse = torch.randn(2, 3)
        layer.merged_output = torch.randn(2, 3, 4)

        def attn_mha(q, k, v, forward_batch, save_kv_cache=False, **kwargs):
            return layer.fresh_output, layer.fresh_lse

        def chunked_prefix_attn_mha(
            q, accum_output, accum_lse, forward_batch
        ):
            return layer.merged_output

        layer.attn_mha = attn_mha
        layer._chunked_prefix_attn_mha = chunked_prefix_attn_mha
        return layer

    def test_no_capture_warmup_matches_captured_output_layout(self):
        layer = self._layer()
        fresh_tokens, heads, v_head_dim = 2, 3, 4
        q = torch.randn(fresh_tokens, heads, v_head_dim + 2)
        k = torch.randn(fresh_tokens, heads, v_head_dim + 2)
        v = torch.randn(fresh_tokens, heads, v_head_dim)
        capture_batch = SimpleNamespace(token=7, extend_prefix_lens_cpu=[30])
        live_batch = SimpleNamespace(token=11, extend_prefix_lens_cpu=[30])
        capture_batch.num_prefix_chunks = 0
        live_batch.num_prefix_chunks = 0
        capture_batch.mha_return_lse = False
        live_batch.mha_return_lse = False
        capture_batch.attn_attend_prefix_cache = None
        live_batch.attn_attend_prefix_cache = None
        capture_batch.set_attn_attend_prefix_cache = (
            lambda value: setattr(capture_batch, "attn_attend_prefix_cache", value)
        )
        live_batch.set_attn_attend_prefix_cache = (
            lambda value: setattr(live_batch, "attn_attend_prefix_cache", value)
        )

        with (
            mock.patch.object(fm, "get_attn_backend", return_value=layer.backend),
            mock.patch.object(
                fm,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            ),
        ):
            with set_forward_batch(live_batch):
                eager_output = layer._bcg_mha_prefix_attention(q, k, v, live_batch)

            with (
                set_forward_batch(capture_batch),
                _RecordingCapture() as capture,
            ):
                captured_output = layer._bcg_mha_prefix_attention(
                    q, k, v, capture_batch
                )

            with set_forward_batch(live_batch):
                capture.graph._break_fns[0]()

        self.assertEqual(eager_output.shape, captured_output.shape)
        self.assertEqual(eager_output.dtype, captured_output.dtype)
        torch.testing.assert_close(eager_output, captured_output)

    def test_capture_allocates_only_fresh_output_and_replay_reads_live_batch(self):
        layer = self._layer()
        fresh_tokens, heads, v_head_dim = 2, 3, 4
        q = torch.randn(fresh_tokens, heads, v_head_dim + 2)
        k = torch.randn(fresh_tokens, heads, v_head_dim + 2)
        v = torch.randn(fresh_tokens, heads, v_head_dim)
        capture_batch = SimpleNamespace(token=7, extend_prefix_lens_cpu=[30])
        live_batch = SimpleNamespace(token=11, extend_prefix_lens_cpu=[30])
        capture_batch.num_prefix_chunks = 0
        live_batch.num_prefix_chunks = 0
        capture_batch.mha_return_lse = False
        live_batch.mha_return_lse = False
        capture_batch.attn_attend_prefix_cache = None
        live_batch.attn_attend_prefix_cache = None
        capture_batch.set_attn_attend_prefix_cache = (
            lambda value: setattr(capture_batch, "attn_attend_prefix_cache", value)
        )
        live_batch.set_attn_attend_prefix_cache = (
            lambda value: setattr(live_batch, "attn_attend_prefix_cache", value)
        )

        with (
            mock.patch.object(fm, "get_attn_backend", return_value=layer.backend),
            mock.patch.object(
                fm,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            ),
        ):
            expected = layer._mha_prefix_attention(q, k, v, live_batch)

        with set_forward_batch(capture_batch), _RecordingCapture() as capture:
            output = layer._bcg_mha_prefix_attention(q, k, v, capture_batch)

        self.assertEqual(capture.events, ["end", "begin"])
        self.assertEqual(output.shape, (fresh_tokens, heads, v_head_dim))
        self.assertEqual(output.numel(), fresh_tokens * heads * v_head_dim)
        with (
            set_forward_batch(live_batch),
            mock.patch.object(fm, "get_attn_backend", return_value=layer.backend),
            mock.patch.object(
                fm,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            ),
        ):
            capture.graph._break_fns[0]()

        torch.testing.assert_close(output, expected)

        # The wrapper's public return is the graph-owned fresh-token buffer,
        # not a prefix-sized tensor.
        self.assertEqual(output.shape, expected.shape)


if __name__ == "__main__":
    import unittest

    unittest.main()
