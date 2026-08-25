# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe as quark_mxfp4
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
    QuarkW4A4MXFp4MoE,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def test_gfx942_mxfp4_alignment_skips_filtered_experts(monkeypatch):
    module_name = "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size"
    fake_align_module = ModuleType(module_name)
    captured = {}

    def moe_align_block_size(topk_ids, block_size, num_experts, **kwargs):
        captured["args"] = (topk_ids, block_size, num_experts)
        captured["kwargs"] = kwargs
        return topk_ids, topk_ids, torch.zeros(1, dtype=torch.int32)

    fake_align_module.moe_align_block_size = moe_align_block_size
    monkeypatch.setitem(sys.modules, module_name, fake_align_module)

    from sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton import _align

    topk_ids = torch.tensor([[0, -1]], dtype=torch.int32)
    _align(topk_ids, block_size=16, num_experts=2)

    assert captured["args"][0] is topk_ids
    assert captured["args"][1:] == (16, 2)
    assert captured["kwargs"] == {"ignore_invalid_expert": True}


def test_gfx942_triton_route_keeps_raw_e8m0_scales(monkeypatch):
    method = QuarkW4A4MXFp4MoE.__new__(QuarkW4A4MXFp4MoE)
    method.is_checkpoint_mxfp4_serialized = True
    method.dequantization_config = None
    method.use_aiter_mxfp4_triton = True

    w13_scale = torch.arange(16, dtype=torch.uint8).reshape(2, 4, 2)
    w2_scale = torch.arange(8, dtype=torch.uint8).reshape(2, 4, 1)
    layer = SimpleNamespace(
        w13_weight=torch.empty((2, 4, 4), dtype=torch.uint8),
        w2_weight=torch.empty((2, 4, 2), dtype=torch.uint8),
        w13_weight_scale=w13_scale.clone(),
        w2_weight_scale=w2_scale.clone(),
    )

    def fail_if_shuffled(_):
        raise AssertionError("raw gfx942 scales must not be preshuffled")

    monkeypatch.setattr(quark_mxfp4, "e8m0_shuffle", fail_if_shuffled, raising=False)
    method.process_weights_after_loading(layer)

    torch.testing.assert_close(layer.w13_weight_scale, w13_scale)
    torch.testing.assert_close(layer.w2_weight_scale, w2_scale)
    assert layer.w13_weight.is_shuffled is False
    assert layer.w2_weight.is_shuffled is False


def test_gfx942_triton_route_does_not_relabel_raw_weight_view_as_shuffled():
    method = QuarkW4A4MXFp4MoE.__new__(QuarkW4A4MXFp4MoE)
    method.use_aiter_mxfp4_triton = True
    captured = {}

    class Runner:
        def run(self, dispatch_output, quant_info):
            captured["quant_info"] = quant_info
            return dispatch_output

    method.runner = Runner()
    w13_weight = torch.empty((2, 4, 4), dtype=torch.uint8)
    w2_weight = torch.empty((2, 4, 2), dtype=torch.uint8)
    w13_weight.is_shuffled = False
    w2_weight.is_shuffled = False
    layer = SimpleNamespace(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        w13_weight_scale=torch.empty((2, 4, 1), dtype=torch.uint8),
        w2_weight_scale=torch.empty((2, 4, 1), dtype=torch.uint8),
        dispatcher=SimpleNamespace(expert_mask_gpu=None),
    )
    dispatch_output = object()

    assert method.apply_weights(layer, dispatch_output) is dispatch_output
    quant_info = captured["quant_info"]
    assert getattr(quant_info.w13_weight, "is_shuffled", False) is False
    assert getattr(quant_info.w2_weight, "is_shuffled", False) is False


def test_gfx942_triton_route_rejects_padded_intermediate_dimension():
    from sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton import _validate_inputs

    hidden = 64
    padded_intermediate = 64
    with pytest.raises(ValueError, match="padded intermediate dimensions"):
        _validate_inputs(
            torch.empty((1, hidden), dtype=torch.bfloat16),
            torch.empty((2, 2 * padded_intermediate, hidden // 2), dtype=torch.uint8),
            torch.empty((2, hidden, padded_intermediate // 2), dtype=torch.uint8),
            torch.empty((2, 2 * padded_intermediate, hidden // 32), dtype=torch.uint8),
            torch.empty((2, hidden, padded_intermediate // 32), dtype=torch.uint8),
            torch.ones((1, 2), dtype=torch.float32),
            torch.tensor([[0, 1]], dtype=torch.int32),
            expected_hidden_size=hidden,
            expected_intermediate_size=32,
        )


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (1, (16, 128, 128, 4, 4, 2)),
        (576, (16, 128, 128, 4, 4, 2)),
        (577, (32, 128, 128, 8, 4, 2)),
        (1024, (32, 128, 128, 8, 4, 2)),
        (1025, (64, 128, 128, 8, 4, 1)),
    ],
)
def test_gfx942_glm52_tp8_uses_measured_token_buckets(tokens, expected):
    from sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton import (
        _gfx942_config,
    )

    config = _gfx942_config(
        tokens,
        hidden=6144,
        intermediate=256,
        num_experts=257,
        topk=9,
    )

    assert (
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["num_warps"],
        config["num_stages"],
    ) == expected


def test_gfx942_glm52_long_prefill_uses_tuned_down_projection_tile():
    from sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton import (
        _gfx942_phase_configs,
    )

    phase1, phase2 = _gfx942_phase_configs(
        32768,
        hidden=6144,
        intermediate=256,
        num_experts=257,
        topk=9,
    )

    assert phase1["BLOCK_SIZE_N"] == 128
    assert phase1["GROUP_SIZE_M"] == 8
    assert phase2["BLOCK_SIZE_N"] == 256
    assert phase2["GROUP_SIZE_M"] == 4
    assert phase1 is not phase2


def test_gfx942_glm52_short_prefill_keeps_shared_phase_tile():
    from sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton import (
        _gfx942_phase_configs,
    )

    phase1, phase2 = _gfx942_phase_configs(
        1024,
        hidden=6144,
        intermediate=256,
        num_experts=257,
        topk=9,
    )

    assert phase1 == phase2
    assert phase1 is not phase2


@pytest.mark.parametrize(
    ("tokens", "expected_n", "expected_warps"),
    [
        (4095, 64, 8),
        (4096, 256, 4),
        (5632, 256, 4),
        (16384, 256, 4),
        (22016, 256, 4),
    ],
)
def test_gfx942_glm52_tp4_long_prefill_uses_measured_tile(
    tokens, expected_n, expected_warps
):
    from sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton import (
        _gfx942_phase_configs,
    )

    phase1, phase2 = _gfx942_phase_configs(
        tokens,
        hidden=6144,
        intermediate=512,
        num_experts=257,
        topk=9,
    )

    assert phase1["BLOCK_SIZE_M"] == 64
    assert phase1["BLOCK_SIZE_N"] == expected_n
    assert phase1["BLOCK_SIZE_K"] == 64
    assert phase1["GROUP_SIZE_M"] == 4
    assert phase1["num_warps"] == expected_warps
    assert phase1["num_stages"] == 2
    assert phase1 == phase2
    assert phase1 is not phase2


def test_gfx942_other_mxfp4_shapes_keep_conservative_tile():
    from sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton import (
        _gfx942_config,
    )

    config = _gfx942_config(
        512,
        hidden=7168,
        intermediate=256,
        num_experts=385,
        topk=9,
    )

    assert config["BLOCK_SIZE_M"] == 64
    assert config["BLOCK_SIZE_N"] == 64
    assert config["BLOCK_SIZE_K"] == 64
    assert config["num_warps"] == 8
    assert config["num_stages"] == 2
