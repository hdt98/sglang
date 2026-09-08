from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.layers.communicator_mhc import MHCState
from sglang.srt.models import glm5_next
from sglang.srt.models.glm5_next import Glm5NextDecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _AddOneNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        return x + 1


def test_mhc_state_uses_optional_fused_attention_to_mlp_boundary():
    hidden_states = torch.randn(2, 8)
    residual = torch.randn(2, 32)
    next_hidden_states = torch.randn(2, 8)
    next_residual = torch.randn(2, 32)
    next_h_res = torch.randn(2, 16)
    next_h_post = torch.randn(2, 4)
    fused = MagicMock(
        return_value=(
            next_hidden_states,
            next_residual,
            next_h_res,
            next_h_post,
            False,
        )
    )
    initial_h_res = torch.randn(2, 16)
    initial_h_post = torch.randn(2, 4)
    state = MHCState(
        hc_mult=4,
        hc_attn_pre=MagicMock(),
        hc_ffn_pre=MagicMock(side_effect=AssertionError("unfused pre called")),
        hc_post=MagicMock(side_effect=AssertionError("unfused post called")),
        hc_attn_to_mlp=fused,
        h_res=initial_h_res,
        h_post=initial_h_post,
    )
    norm = _AddOneNorm(8)

    actual_hidden_states, actual_residual = state.attn_to_mlp(
        hidden_states, residual, norm
    )

    torch.testing.assert_close(actual_hidden_states, next_hidden_states + 1)
    assert actual_residual is next_residual
    assert state.h_res is next_h_res
    assert state.h_post is next_h_post
    fused.assert_called_once()
    args = fused.call_args.args
    assert args[0] is hidden_states
    assert args[1] is residual
    assert args[2] is initial_h_res
    assert args[3] is initial_h_post
    torch.testing.assert_close(args[4], norm.weight.data)
    assert args[5] == norm.variance_epsilon


def test_glm_aiter_mhc_boundary_preserves_communicator_shapes():
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        hc_mult=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-5,
        hc_sinkhorn_iters=20,
    )
    layer.hc_ffn_fn = nn.Parameter(torch.randn(24, 32))
    layer.hc_ffn_scale = nn.Parameter(torch.randn(3))
    layer.hc_ffn_base = nn.Parameter(torch.randn(24))

    hidden_states = torch.randn(2, 8)
    residual = torch.randn(2, 32)
    h_res = torch.randn(2, 16)
    h_post = torch.randn(2, 4)
    next_residual = torch.randn(2, 4, 8)
    next_hidden_states = torch.randn(2, 8)
    next_h_post = torch.randn(2, 4)
    next_h_res = torch.randn(2, 4, 4)

    with (
        patch(
            "sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc."
            "apply_mhc_post_pre_boundary",
            return_value=(
                next_residual,
                next_hidden_states,
                next_h_post,
                next_h_res,
                True,
            ),
        ) as fused,
        patch.object(glm5_next, "_GLM_AITER_FUSED_MHC_LOGGED", False),
    ):
        actual = layer.hc_attn_to_mlp(
            hidden_states,
            residual,
            h_res,
            h_post,
            torch.ones(8),
            1e-6,
        )

    actual_hidden, actual_residual, actual_h_res, actual_h_post, norm_fused = (
        actual
    )
    assert actual_hidden is next_hidden_states
    torch.testing.assert_close(actual_residual, next_residual.reshape(2, 32))
    torch.testing.assert_close(actual_h_res, next_h_res.reshape(2, 16))
    torch.testing.assert_close(actual_h_post, next_h_post)
    assert norm_fused
    assert fused.call_args.kwargs["fn_transpose"] is True
    assert fused.call_args.kwargs["residual"].shape == (2, 4, 8)

