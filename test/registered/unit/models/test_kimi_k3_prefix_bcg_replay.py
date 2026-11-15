"""Kimi K3 prefix-BCG absorbed-MLA gating tests.

The tests cover the environment flag, the DCP1-only scope, and the attention
dispatch the prefill graph runner observes.
"""

import os
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.models import kimi_k3
from sglang.srt.models.deepseek_common import attention_backend_handler as abh
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

    with mock.patch.object(
        kimi_k3.DeepseekV2AttentionMLA, "__init__", fake_base_init
    ):
        return kimi_k3.KimiK3MLAAttention(_config(), layer_idx=0)


class TestKimiK3PrefixBcgReplay(CustomTestCase):
    def test_dcp1_pins_absorbed_mla_when_enabled(self):
        with (
            mock.patch.object(kimi_k3, "_is_hip", True),
            envs.SGLANG_ENABLE_K3_PREFIX_BCG_REPLAY.override(True),
        ):
            attention = _attention(dcp_size=1)

        self.assertEqual(attention.dcp_size, 1)
        self.assertTrue(attention.mla_pinned_under_bcg)

    def test_pin_is_off_by_default(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(kimi_k3, "_is_hip", True),
        ):
            attention = _attention(dcp_size=1)

        self.assertFalse(attention.mla_pinned_under_bcg)

    def test_pin_requires_hip(self):
        with (
            mock.patch.object(kimi_k3, "_is_hip", False),
            envs.SGLANG_ENABLE_K3_PREFIX_BCG_REPLAY.override(True),
        ):
            attention = _attention(dcp_size=1)

        self.assertFalse(attention.mla_pinned_under_bcg)

    def test_dcp_above_one_never_pins(self):
        with (
            mock.patch.object(kimi_k3, "_is_hip", True),
            envs.SGLANG_ENABLE_K3_PREFIX_BCG_REPLAY.override(True),
        ):
            attention = _attention(dcp_size=2)

        self.assertFalse(attention.mla_pinned_under_bcg)

    def test_dcp1_pin_dispatches_aiter_bcg_to_absorbed_mla(self):
        with (
            mock.patch.object(kimi_k3, "_is_hip", True),
            envs.SGLANG_ENABLE_K3_PREFIX_BCG_REPLAY.override(True),
        ):
            attention = _attention(dcp_size=1)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: False),
        )

        with (
            mock.patch.object(abh, "_is_hip", True),
            mock.patch.object(
                abh, "is_in_tc_piecewise_cuda_graph", return_value=False
            ),
            mock.patch.object(abh, "is_in_breakable_cuda_graph", return_value=True),
        ):
            method = abh.handle_attention_aiter(attention, batch)
            resolved = abh.resolve_rocm_forward_method(method)

        self.assertEqual(method, AttnForwardMethod.MLA)
        self.assertEqual(resolved, AttnForwardMethod.MLA_ROCM)


if __name__ == "__main__":
    import unittest

    unittest.main()
