import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.aiter as aiter_runner
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def _runner_input():
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    return AiterRunnerInput(
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        topk_ids=topk_ids,
        topk_weights=torch.ones(topk_ids.shape, dtype=torch.float32),
        quant_type=AiterQuantType.PER_1X32,
    )


def _quant_info(**overrides):
    kwargs = {
        "w13_weight": torch.empty((2, 8, 2)),
        "w2_weight": torch.empty((2, 4, 2)),
        "quant_type": AiterQuantType.PER_1X32,
        "w13_scale": torch.empty((2, 8, 1), dtype=torch.uint8),
        "w2_scale": torch.empty((2, 4, 1), dtype=torch.uint8),
    }
    kwargs.update(overrides)
    return AiterMoeQuantInfo(**kwargs)


def _install_fake_aiter(monkeypatch, fused_moe):
    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_aiter.ActivationType = SimpleNamespace(Silu="Silu")
    fake_aiter.QuantType = SimpleNamespace(per_1x32="per_1x32")

    fake_fused_moe = ModuleType("aiter.fused_moe")
    fake_fused_moe.fused_moe = fused_moe

    fake_ops = ModuleType("aiter.ops")
    fake_ops.__path__ = []
    fake_flydsl = ModuleType("aiter.ops.flydsl")
    fake_flydsl.__path__ = []
    fake_moe_common = ModuleType("aiter.ops.flydsl.moe_common")
    fake_moe_common.GateMode = SimpleNamespace(
        INTERLEAVE=SimpleNamespace(value="INTERLEAVE")
    )

    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fake_fused_moe)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", fake_flydsl)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.moe_common", fake_moe_common)


def test_aiter_runner_forwards_no_combine_and_extra_fused_moe_kwargs(monkeypatch):
    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )

    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", no_combine=True))

    runner.run(
        _runner_input(),
        _quant_info(fused_moe_kwargs={"custom_fused_moe_kwarg": "enabled"}),
        running_state={},
    )

    assert captured["activation"] == "Silu"
    assert captured["quant_type"] == "per_1x32"
    assert captured["no_combine"] is True
    assert captured["custom_fused_moe_kwarg"] == "enabled"


def test_aiter_runner_rejects_no_combine_when_fused_moe_does_not_support_it(
    monkeypatch,
):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: False
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))

    with pytest.raises(NotImplementedError, match="no_combine=True"):
        runner.run(_runner_input(), _quant_info(), running_state={})


def test_aiter_runner_preserves_no_combine_rank_for_empty_input(monkeypatch):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))
    runner_input = _runner_input()
    runner_input.hidden_states = torch.zeros((0, 4), dtype=torch.bfloat16)
    runner_input.topk_ids = torch.zeros((0, 2), dtype=torch.int32)
    runner_input.topk_weights = torch.zeros((0, 2), dtype=torch.float32)

    output = runner.run(runner_input, _quant_info(), running_state={})

    assert output.hidden_states.shape == (0, 2, 4)


def test_aiter_runner_routes_marked_mxfp4_to_triton(monkeypatch):
    fused_moe_called = False
    captured = {}

    def fused_moe(**kwargs):
        nonlocal fused_moe_called
        fused_moe_called = True
        return kwargs["hidden_states"]

    def fused_moe_mxfp4_triton(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return args[0]

    _install_fake_aiter(monkeypatch, fused_moe)
    fake_triton_route = ModuleType(
        "sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton"
    )
    fake_triton_route.fused_moe_mxfp4_triton = fused_moe_mxfp4_triton
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton",
        fake_triton_route,
    )

    runner_input = _runner_input()
    runner = AiterRunnerCore(
        MoeRunnerConfig(
            num_experts=2,
            num_local_experts=2,
            activation="silu",
        )
    )
    output = runner.run(
        runner_input,
        _quant_info(mxfp4_triton=True),
        running_state={},
    )

    assert fused_moe_called is False
    assert output.hidden_states is runner_input.hidden_states
    assert captured["args"][0] is runner_input.hidden_states
    assert captured["kwargs"]["activation"] == "silu"
    assert captured["kwargs"]["expected_hidden_size"] is None
    assert captured["kwargs"]["expected_intermediate_size"] is None
    assert captured["kwargs"]["apply_router_weight_on_input"] is False


def test_aiter_runner_rejects_mxfp4_triton_activation_scale(monkeypatch):
    _install_fake_aiter(monkeypatch, lambda **kwargs: kwargs["hidden_states"])
    runner_input = _runner_input()
    runner_input.a1_scale = torch.ones((1,), dtype=torch.float32)
    runner = AiterRunnerCore(
        MoeRunnerConfig(num_experts=2, num_local_experts=2, activation="silu")
    )

    with pytest.raises(NotImplementedError, match="activation scales"):
        runner.run(
            runner_input,
            _quant_info(mxfp4_triton=True),
            running_state={},
        )


@pytest.mark.parametrize(
    ("config_overrides", "quant_overrides", "match"),
    [
        ({"is_gated": False}, {}, "is_gated=False"),
        ({"gemm1_alpha": 1.0}, {}, "gemm1 alpha/beta/clamp"),
        ({"num_local_experts": 1}, {}, "expert parallelism"),
        ({}, {"quant_type": AiterQuantType.NONE}, "quant_type="),
    ],
)
def test_aiter_runner_rejects_unsupported_mxfp4_triton_contract(
    monkeypatch, config_overrides, quant_overrides, match
):
    _install_fake_aiter(monkeypatch, lambda **kwargs: kwargs["hidden_states"])
    runner_input = _runner_input()
    config = {
        "num_experts": 2,
        "num_local_experts": 2,
        "activation": "silu",
    }
    config.update(config_overrides)
    runner = AiterRunnerCore(MoeRunnerConfig(**config))

    with pytest.raises(NotImplementedError, match=match):
        runner.run(
            runner_input,
            _quant_info(mxfp4_triton=True, **quant_overrides),
            running_state={},
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
