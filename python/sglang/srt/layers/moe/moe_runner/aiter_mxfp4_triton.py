# SPDX-License-Identifier: Apache-2.0
"""Run raw MXFP4 MoE weights with AITER's Triton kernels on gfx942.

AITER's regular ``fused_moe`` dispatcher has no working A4W4 path on CDNA3.
The Triton MXFP4 kernels can instead consume BF16 activations and raw packed
FP4 weights, lowering ``tl.dot_scaled`` to instructions available on gfx942.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@functools.cache
def _arch() -> str:
    try:
        name = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # noqa: BLE001
        return ""
    return name.split(":")[0]


@functools.cache
def _kernels_importable() -> bool:
    try:
        import aiter.ops.triton.moe.moe_op_mxfp4  # noqa: F401
        import aiter.ops.triton.moe.moe_op_mxfp4_silu_fused  # noqa: F401

        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("AITER Triton MXFP4 MoE kernels are unavailable: %s", exc)
        return False


@functools.cache
def use_triton_mxfp4_moe() -> bool:
    """Return whether the CDNA3 compatibility route is usable."""

    return _arch() == "gfx942" and _kernels_importable()


def _gfx942_config(
    num_tokens: int | None = None,
    *,
    hidden: int | None = None,
    intermediate: int | None = None,
    num_experts: int | None = None,
    topk: int | None = None,
) -> dict[str, Any]:
    """Return a gfx942 tile that fits the device's 64 KiB LDS.

    Pinned AITER builds do not ship a ``gfx942-MOE-MX_FP4.json`` file and fall
    back to a 256x256 tile that does not compile.  This starts from AITER's
    known gfx942 small-M defaults. GLM-5.2's exact TP8 target and TP4 long
    prefills use tiles measured on MI325X; other shapes retain the conservative
    compatibility tile.
    """

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 2,
    }
    if (
        num_tokens is not None
        and num_tokens >= 4096
        and hidden == 6144
        and intermediate == 512
        and num_experts == 257
        and topk == 9
    ):
        config.update(BLOCK_SIZE_N=256, num_warps=4)
        return config

    if not (
        num_tokens is not None
        and hidden == 6144
        and intermediate == 256
        and num_experts == 257
        and topk == 9
    ):
        return config

    config.update(BLOCK_SIZE_N=128, BLOCK_SIZE_K=128)
    if num_tokens <= 576:
        config.update(BLOCK_SIZE_M=16, num_warps=4)
    elif num_tokens <= 1024:
        config.update(BLOCK_SIZE_M=32, GROUP_SIZE_M=8, num_warps=4)
    else:
        config.update(GROUP_SIZE_M=8, num_warps=4, num_stages=1)
    return config


def _gfx942_phase_configs(
    num_tokens: int | None = None,
    *,
    hidden: int | None = None,
    intermediate: int | None = None,
    num_experts: int | None = None,
    topk: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return independent configs for gate/up and down MXFP4 GEMMs."""

    phase1 = _gfx942_config(
        num_tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
    )
    phase2 = phase1.copy()
    if (
        num_tokens is not None
        and num_tokens > 1024
        and hidden == 6144
        and intermediate == 256
        and num_experts == 257
        and topk == 9
    ):
        # The down projection is N=6144/K=256, unlike the gate/up phase's
        # N=512/K=6144.  On MI325X this wider N tile wins throughout the M64
        # bucket and reduces the 32K two-kernel pair by about 7%.
        phase2.update(BLOCK_SIZE_N=256, GROUP_SIZE_M=4)
    return phase1, phase2


@functools.lru_cache(maxsize=8)
def _unit_scales(num_experts: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # The kernels carry an additional scalar scale over the E8M0 microscales.
    # Quark checkpoints do not have one, so both factors are one.
    return (
        torch.ones(1, dtype=torch.float32, device=device),
        torch.ones(num_experts, dtype=torch.float32, device=device),
    )


def _align(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(
        topk_ids, block_size, num_experts, ignore_invalid_expert=True
    )


def _validate_inputs(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expected_hidden_size: int | None = None,
    expected_intermediate_size: int | None = None,
) -> tuple[int, int, int, int]:
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be rank 2, got {hidden_states.shape}")
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            "gfx942 MXFP4 Triton MoE expects BF16 or FP16 activations, "
            f"got {hidden_states.dtype}"
        )
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("MXFP4 expert weights must be rank 3")
    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise TypeError("MXFP4 expert weights must be packed uint8 views")
    if w13_scale.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
        raise TypeError("MXFP4 expert microscales must be uint8 E8M0")
    tensors = (w13, w2, w13_scale, w2_scale, topk_weights, topk_ids)
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("all MXFP4 MoE inputs must be on the same device")
    if not all(x.is_contiguous() for x in (w13, w2, w13_scale, w2_scale)):
        raise ValueError("MXFP4 expert weights and microscales must be contiguous")

    num_experts, gate_up, packed_hidden = w13.shape
    tokens, hidden = hidden_states.shape
    if expected_hidden_size is not None and hidden != expected_hidden_size:
        raise ValueError(
            f"hidden_states has hidden={hidden}, model expects {expected_hidden_size}"
        )
    if gate_up % 2 != 0:
        raise ValueError(f"gate/up rows must be even, got {gate_up}")
    intermediate = gate_up // 2
    if hidden % 32 != 0 or intermediate % 32 != 0:
        raise ValueError(
            "MXFP4 hidden and intermediate dimensions must be multiples of 32, "
            f"got hidden={hidden}, intermediate={intermediate}"
        )
    if (
        expected_intermediate_size is not None
        and intermediate != expected_intermediate_size
    ):
        raise ValueError(
            "padded intermediate dimensions are unsupported: "
            f"weight has {intermediate}, model expects {expected_intermediate_size}"
        )
    if packed_hidden * 2 != hidden:
        raise ValueError(
            f"w13 packed K={packed_hidden * 2} does not match hidden={hidden}"
        )
    if tuple(w2.shape) != (num_experts, hidden, intermediate // 2):
        raise ValueError(
            "w2 must have shape "
            f"{(num_experts, hidden, intermediate // 2)}, got {tuple(w2.shape)}"
        )
    if tuple(w13_scale.shape) != (num_experts, gate_up, hidden // 32):
        raise ValueError("w13_scale is not in raw [E, 2I, H/32] layout")
    if tuple(w2_scale.shape) != (num_experts, hidden, intermediate // 32):
        raise ValueError("w2_scale is not in raw [E, H, I/32] layout")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("topk ids and weights must have the same rank-2 shape")
    if topk_ids.shape[0] != tokens:
        raise ValueError("topk token count does not match hidden_states")
    if topk_ids.shape[1] == 0 or topk_ids.shape[1] > num_experts:
        raise ValueError(f"topk must be in [1, {num_experts}], got {topk_ids.shape[1]}")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_ids must be int32 or int64, got {topk_ids.dtype}")
    if not topk_weights.is_floating_point():
        raise TypeError(
            f"topk_weights must be floating point, got {topk_weights.dtype}"
        )
    return tokens, hidden, intermediate, num_experts


def fused_moe_mxfp4_triton(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "silu",
    expected_hidden_size: int | None = None,
    expected_intermediate_size: int | None = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Compute a TP-local SwiGLU MoE from raw Quark MXFP4 weights."""

    if activation != "silu":
        raise NotImplementedError(
            f"gfx942 MXFP4 Triton route only supports silu, got {activation}"
        )

    from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4
    from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import (
        fused_moe_mxfp4_silu,
    )
    from aiter.ops.triton.utils.types import torch_to_triton_dtype

    w13 = w13_weight.view(torch.uint8)
    w2 = w2_weight.view(torch.uint8)
    tokens, hidden, intermediate, num_experts = _validate_inputs(
        hidden_states,
        w13,
        w2,
        w13_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        expected_hidden_size,
        expected_intermediate_size,
    )
    topk_ids = topk_ids.to(dtype=torch.int32).contiguous()
    topk_weights = topk_weights.to(dtype=torch.float32).contiguous()
    topk = topk_ids.shape[1]

    # AITER loads sorted token ids as 32-bit values in some pinned builds.
    # Split before a flat activation/output offset can overflow that address.
    max_rows = (2**31 - 1) // max(intermediate, hidden)
    max_tokens = max(1, max_rows // max(topk, 1))
    if tokens > max_tokens:
        return torch.cat(
            [
                fused_moe_mxfp4_triton(
                    hidden_states[start : start + max_tokens],
                    w13,
                    w2,
                    w13_scale,
                    w2_scale,
                    topk_weights[start : start + max_tokens],
                    topk_ids[start : start + max_tokens],
                    activation=activation,
                    expected_hidden_size=expected_hidden_size,
                    expected_intermediate_size=expected_intermediate_size,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                )
                for start in range(0, tokens, max_tokens)
            ],
            dim=0,
        )

    phase1_config, phase2_config = _gfx942_phase_configs(
        tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
    )
    if phase1_config["BLOCK_SIZE_M"] != phase2_config["BLOCK_SIZE_M"]:
        raise ValueError(
            "MXFP4 phase configs must share BLOCK_SIZE_M because they reuse "
            "one routed-token alignment"
        )
    sorted_token_ids, expert_ids, num_tokens_post_padded = _align(
        topk_ids, phase1_config["BLOCK_SIZE_M"], num_experts
    )
    debug_sync = envs.SGLANG_DEBUG_MXFP4_TRITON_SYNC.get()
    if debug_sync:
        torch.cuda.synchronize()
    a_scale, b_scale = _unit_scales(num_experts, str(hidden_states.device))
    compute_type = torch_to_triton_dtype[hidden_states.dtype]

    intermediate_states = torch.empty(
        (tokens * topk, intermediate),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    # Quark's loader stores gate rows followed by up rows.  AITER's fused
    # kernel deliberately alternates reads from the two halves before its
    # final ``reshape(..., 2).split()``, so no row reordering is needed here.
    fused_moe_mxfp4_silu(
        hidden_states,
        w13,
        intermediate_states,
        a_scale,
        b_scale,
        None,
        w13_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk,
        False,
        False,
        phase1_config,
        compute_type,
    )
    if debug_sync:
        torch.cuda.synchronize()

    down = torch.empty(
        (tokens * topk, 1, hidden),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    fused_moe_mxfp4(
        intermediate_states,
        w2,
        down,
        a_scale,
        b_scale,
        None,
        w2_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        False,
        False,
        phase2_config,
        compute_type,
    )
    if debug_sync:
        torch.cuda.synchronize()
    return down.view(tokens, topk, hidden).sum(dim=1)
