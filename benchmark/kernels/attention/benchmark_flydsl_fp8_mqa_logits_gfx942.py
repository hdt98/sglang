"""Benchmark AITER FlyDSL FP8 MQA logits on GLM-5.2 long-prefill shapes.

This benchmark mirrors the ragged-indexer calls observed in an exact 128K
SGLang trace on gfx942.  It intentionally uses a small shape for cross-variant
correctness so the reference does not materialize a multi-gigabyte tensor.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch
from aiter.ops.flydsl import (
    FP8_MQA_LOGITS_VARIANTS,
    flydsl_fp8_mqa_logits,
)


@dataclass(frozen=True)
class Case:
    name: str
    q_len: int
    kv_len: int
    query_start: int


# The fourth 32K scheduler chunk is split by SGLang's logits-memory budget.
# q_len=27,817 and 4,951 are taken directly from the TP0 128K trace.
REAL_128K_CASES = (
    Case("32k", q_len=32_768, kv_len=32_768, query_start=0),
    Case("64k", q_len=32_768, kv_len=65_536, query_start=32_768),
    Case("96k", q_len=32_768, kv_len=98_304, query_start=65_536),
    Case("128k-main", q_len=27_817, kv_len=131_072, query_start=98_304),
    Case("128k-tail", q_len=4_951, kv_len=131_072, query_start=126_121),
)
BENCHMARK_CASES = REAL_128K_CASES + (
    Case("128k-full", q_len=32_768, kv_len=131_072, query_start=98_304),
)


def make_inputs(case: Case, *, seed: int = 0) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    num_heads = 32
    head_size = 128
    fp8_dtype = torch.float8_e4m3fnuz

    q = torch.randn(
        case.q_len,
        num_heads,
        head_size,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(fp8_dtype)
    kv = torch.randn(case.kv_len, head_size, device="cuda", dtype=torch.bfloat16).to(
        fp8_dtype
    )
    kv_scales = torch.ones(case.kv_len, device="cuda", dtype=torch.float32)
    weights = torch.randn(case.q_len, num_heads, device="cuda", dtype=torch.float32)
    starts = torch.zeros(case.q_len, device="cuda", dtype=torch.int32)
    ends = torch.arange(
        case.query_start + 1,
        case.query_start + case.q_len + 1,
        device="cuda",
        dtype=torch.int32,
    )
    assert int(ends[-1]) <= case.kv_len
    return q, kv, kv_scales, weights, starts, ends


def run_kernel(inputs: tuple[torch.Tensor, ...], variant: str) -> torch.Tensor:
    return flydsl_fp8_mqa_logits(
        *inputs,
        clean_logits=False,
        variant=variant,
    )


def check_variant(variant: str) -> None:
    case = Case("correctness", q_len=64, kv_len=512, query_start=448)
    inputs = make_inputs(case)
    expected = run_kernel(inputs, "mfma_r4_w4")
    actual = run_kernel(inputs, variant)
    torch.cuda.synchronize()

    starts, ends = inputs[-2:]
    columns = torch.arange(case.kv_len, device="cuda")
    valid = (columns >= starts[:, None]) & (columns < ends[:, None])
    expected = expected[valid]
    actual = actual[valid]
    diff = (actual - expected).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    is_close = torch.allclose(actual, expected, rtol=1e-2, atol=1e-2)
    print(
        f"correctness variant={variant} allclose={is_close} "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}",
        flush=True,
    )
    if not is_close:
        raise AssertionError(f"{variant} does not match mfma_r4_w4")


def benchmark_case(
    case: Case,
    variant: str,
    *,
    warmup: int,
    repeats: int,
) -> float:
    inputs = make_inputs(case)

    for _ in range(warmup):
        output = run_kernel(inputs, variant)
        torch.cuda.synchronize()
        del output

    samples_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = run_kernel(inputs, variant)
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))
        del output

    median_ms = statistics.median(samples_ms)
    window_elems = (
        case.q_len * (case.query_start + 1) + case.q_len * (case.q_len - 1) // 2
    )
    flops = 2 * 32 * 128 * window_elems
    tflops = flops / median_ms / 1e9
    logits_gib = case.q_len * case.kv_len * 4 / 2**30
    print(
        f"case={case.name} variant={variant} q={case.q_len} kv={case.kv_len} "
        f"logits_gib={logits_gib:.3f} median_ms={median_ms:.4f} "
        f"mean_ms={statistics.mean(samples_ms):.4f} "
        f"min_ms={min(samples_ms):.4f} max_ms={max(samples_ms):.4f} "
        f"tflops={tflops:.2f}",
        flush=True,
    )
    del inputs
    torch.cuda.empty_cache()
    return median_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=FP8_MQA_LOGITS_VARIANTS)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--cases",
        default=",".join(case.name for case in REAL_128K_CASES),
        help="Comma-separated case names.",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = {name.strip() for name in args.cases.split(",") if name.strip()}
    cases = [case for case in BENCHMARK_CASES if case.name in requested]
    unknown = requested - {case.name for case in cases}
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    print(
        f"device={torch.cuda.get_device_name(0)} variant={args.variant} "
        f"warmup={args.warmup} repeats={args.repeats}",
        flush=True,
    )
    if not args.skip_correctness:
        check_variant(args.variant)
    for case in cases:
        benchmark_case(
            case,
            args.variant,
            warmup=args.warmup,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
