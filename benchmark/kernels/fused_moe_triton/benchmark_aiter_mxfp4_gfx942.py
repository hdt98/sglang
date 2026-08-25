# SPDX-License-Identifier: Apache-2.0
"""Correctness smoke and microbenchmark for the gfx942 MXFP4 MoE route.

The defaults use GLM-5.2's TP8-local hidden and intermediate dimensions while
keeping the expert count small enough for a quick correctness run.  Pass
``--experts 257 --topk 9`` for GLM-5.2-MXFP4's production target shape after
fusing its MXFP4 shared expert; add ``--no-check`` only for timing after that
shape passes.  Tile flags override the adapter's automatic shape selection
when supplied.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

import sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton as mxfp4_triton

_FP4_E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _packed_fp4(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    low = torch.randint(0, 16, shape, dtype=torch.uint8, device=device)
    high = torch.randint(0, 16, shape, dtype=torch.uint8, device=device)
    return low | (high << 4)


def _dequant_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    lut = _FP4_E2M1.to(weight.device)
    low = lut[(weight & 0xF).long()]
    high = lut[(weight >> 4).long()]
    unpacked = torch.empty(
        (*weight.shape[:-1], weight.shape[-1] * 2),
        dtype=torch.float32,
        device=weight.device,
    )
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high

    exponent = scale.to(torch.int32)
    decoded_scale = torch.exp2((exponent - 127).to(torch.float32))
    decoded_scale = torch.where(exponent == 255, torch.nan, decoded_scale)
    return unpacked * decoded_scale.repeat_interleave(32, dim=-1)


def _reference(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    w13_dequant = _dequant_mxfp4(w13, w13_scale)
    w2_dequant = _dequant_mxfp4(w2, w2_scale)
    outputs = []
    for token in range(hidden_states.shape[0]):
        token_output = torch.zeros(
            hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device
        )
        for route in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, route])
            route_weight = float(topk_weights[token, route])
            if expert < 0 or expert >= w13.shape[0]:
                if route_weight != 0.0:
                    raise ValueError(
                        f"invalid expert {expert} has nonzero weight {route_weight}"
                    )
                continue
            route_input = hidden_states[token].float()
            if apply_router_weight_on_input:
                route_input = route_input * route_weight
            gate_up = F.linear(route_input, w13_dequant[expert])
            gate, up = gate_up.chunk(2)
            intermediate = F.silu(gate) * up
            token_output.add_(
                F.linear(intermediate, w2_dequant[expert]),
                alpha=1.0 if apply_router_weight_on_input else route_weight,
            )
        outputs.append(token_output)
    return torch.stack(outputs).to(hidden_states.dtype)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument(
        "--route-pattern",
        choices=("round-robin", "concentrated", "random"),
        default="round-robin",
        help="Distribution of routed expert ids before optional sentinel padding.",
    )
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--block-m", type=int)
    parser.add_argument("--block-n", type=int)
    parser.add_argument("--block-k", type=int)
    parser.add_argument("--group-size-m", type=int)
    parser.add_argument("--num-warps", type=int)
    parser.add_argument("--num-stages", type=int)
    parser.add_argument("--kpack", type=int)
    for phase in ("phase1", "phase2"):
        parser.add_argument(f"--{phase}-block-n", type=int)
        parser.add_argument(f"--{phase}-block-k", type=int)
        parser.add_argument(f"--{phase}-group-size-m", type=int)
        parser.add_argument(f"--{phase}-num-warps", type=int)
        parser.add_argument(f"--{phase}-num-stages", type=int)
        parser.add_argument(f"--{phase}-kpack", type=int)
    parser.add_argument(
        "--stock-aiter",
        action="store_true",
        help=(
            "Run the stock aiter.fused_moe path once to reproduce its gfx942 "
            "failure."
        ),
    )
    parser.add_argument("--router-weight-on-input", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.experts < 1 or args.topk < 1 or args.topk > args.experts:
        raise ValueError("experts and topk must satisfy 1 <= topk <= experts")
    if args.hidden % 32 or args.intermediate % 32:
        raise ValueError("hidden and intermediate dimensions must be multiples of 32")

    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden_states = (
        torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16) * 0.1
    )
    w13 = _packed_fp4((args.experts, 2 * args.intermediate, args.hidden // 2), device)
    w2 = _packed_fp4((args.experts, args.hidden, args.intermediate // 2), device)
    # Vary the E8M0 exponents across rows and K blocks so the reference catches
    # raw-scale layout/indexing mistakes instead of only validating FP4 nibbles.
    # These bounds cover the 112..125 range observed in routed and fused-shared
    # expert tensors from the amd/GLM-5.2-MXFP4 checkpoint.
    w13_scale = torch.randint(
        112,
        126,
        (args.experts, 2 * args.intermediate, args.hidden // 32),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.randint(
        118,
        126,
        (args.experts, args.hidden, args.intermediate // 32),
        dtype=torch.uint8,
        device=device,
    )
    if args.topk > 1:
        # GLM-5.2 normalizes its routed weights, then appends the fused shared
        # expert as the final route with weight one.
        routed_experts = args.experts - 1
        if args.topk - 1 > routed_experts:
            raise ValueError("topk leaves too few experts for the shared slot")
        if args.route_pattern == "concentrated":
            routed_ids = torch.zeros(
                (args.tokens, args.topk - 1), dtype=torch.int64, device=device
            )
        elif args.route_pattern == "random":
            routed_ids = torch.randint(
                routed_experts,
                (args.tokens, args.topk - 1),
                dtype=torch.int64,
                device=device,
            )
        else:
            routed_ids = (
                torch.arange(args.tokens, device=device)[:, None]
                + torch.arange(args.topk - 1, device=device)[None, :]
            ) % routed_experts
        shared_ids = torch.full(
            (args.tokens, 1), args.experts - 1, dtype=torch.int64, device=device
        )
        topk_ids = torch.cat((routed_ids, shared_ids), dim=1).to(torch.int32)
    else:
        topk_ids = (
            (torch.arange(args.tokens, device=device) % args.experts)
            .unsqueeze(1)
            .to(torch.int32)
        )
    topk_weights = torch.rand(
        args.tokens, args.topk, device=device, dtype=torch.float32
    )
    if args.topk > 1:
        topk_weights[:, :-1] /= topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        topk_weights[:, -1] = 1.0
    else:
        topk_weights.fill_(1.0)
    if args.stock_aiter:
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe
        from aiter.utility.fp4_utils import e8m0_shuffle

        if not hasattr(torch, "float4_e2m1fn_x2"):
            raise RuntimeError("stock AITER MXFP4 requires torch.float4_e2m1fn_x2")
        stock_w13_scale = e8m0_shuffle(
            w13_scale.view(args.experts * 2 * args.intermediate, -1)
        ).view_as(w13_scale)
        stock_w2_scale = e8m0_shuffle(
            w2_scale.view(args.experts * args.hidden, -1)
        ).view_as(w2_scale)
        print("launching stock aiter.fused_moe; success is unexpected on gfx942")
        output = fused_moe(
            hidden_states=hidden_states,
            w1=w13.view(torch.float4_e2m1fn_x2),
            w2=w2.view(torch.float4_e2m1fn_x2),
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=stock_w13_scale,
            w2_scale=stock_w2_scale,
        )
        torch.cuda.synchronize()
        print(f"stock aiter.fused_moe unexpectedly succeeded: {tuple(output.shape)}")
        return

    phase1_config, phase2_config = mxfp4_triton._gfx942_phase_configs(
        args.tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_experts=args.experts,
        topk=args.topk,
    )
    overrides = {
        "BLOCK_SIZE_M": args.block_m,
        "BLOCK_SIZE_N": args.block_n,
        "BLOCK_SIZE_K": args.block_k,
        "GROUP_SIZE_M": args.group_size_m,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "kpack": args.kpack,
    }
    for config in (phase1_config, phase2_config):
        config.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
    phase_fields = {
        "BLOCK_SIZE_N": "block_n",
        "BLOCK_SIZE_K": "block_k",
        "GROUP_SIZE_M": "group_size_m",
        "num_warps": "num_warps",
        "num_stages": "num_stages",
        "kpack": "kpack",
    }
    for phase, config in (("phase1", phase1_config), ("phase2", phase2_config)):
        config.update(
            {
                key: value
                for key, field in phase_fields.items()
                if (value := getattr(args, f"{phase}_{field}")) is not None
            }
        )
    mxfp4_triton._gfx942_phase_configs = lambda *_args, **_kwargs: (
        phase1_config,
        phase2_config,
    )
    _, _, num_tokens_post_padded = mxfp4_triton._align(
        topk_ids, phase1_config["BLOCK_SIZE_M"], args.experts
    )
    padded_routes = int(num_tokens_post_padded.item())

    def run() -> torch.Tensor:
        return mxfp4_triton.fused_moe_mxfp4_triton(
            hidden_states,
            w13,
            w2,
            w13_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input=args.router_weight_on_input,
        )

    compile_start = time.perf_counter()
    output = run()
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start

    if not args.no_check:
        reference = _reference(
            hidden_states,
            w13,
            w2,
            w13_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input=args.router_weight_on_input,
        )
        output_float = output.float()
        reference_float = reference.float()
        diff = (output_float - reference_float).abs()
        cosine = F.cosine_similarity(
            output_float.flatten(), reference_float.flatten(), dim=0
        )
        relative_l2 = (diff.norm() / reference_float.norm()).item()
        print(
            f"correctness max_abs={diff.max().item():.6f} "
            f"mean_abs={diff.mean().item():.6f} "
            f"ref_abs_max={reference_float.abs().max().item():.6f} "
            f"rel_l2={relative_l2:.8f} cosine={cosine.item():.8f}"
        )
        cosine_value = cosine.item()
        if not torch.isfinite(cosine).item() or cosine_value < 0.999:
            raise AssertionError(f"cosine similarity is too low: {cosine_value:.8f}")
        torch.testing.assert_close(output, reference, atol=0.25, rtol=0.08)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run()
    end.record()
    end.synchronize()

    props = torch.cuda.get_device_properties(0)
    milliseconds = start.elapsed_time(end) / args.iters
    print(
        f"arch={props.gcnArchName.split(':')[0]} tokens={args.tokens} "
        f"experts={args.experts} topk={args.topk} hidden={args.hidden} "
        f"intermediate={args.intermediate} phase1_config={phase1_config} "
        f"phase2_config={phase2_config} "
        f"valid_routes={args.tokens * args.topk} "
        f"padded_routes={padded_routes} "
        f"compile_s={compile_seconds:.3f} latency_ms={milliseconds:.4f}"
    )


if __name__ == "__main__":
    main()
