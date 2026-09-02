"""Benchmark RCCL and AITER reduce-scatter at prefill-CP hidden-state shapes.

The input shape is the full tensor before scattering along dimension zero. For
example, GLM-5.3-Flash with a 16K aggregate prefill batch and hidden size 4096
uses ``(16384, 4096)`` on a four-rank context-parallel group.

Usage:
    AITER_AOT_IMPORT=1 torchrun --nproc_per_node=4 \
      benchmark/kernels/all_gather/benchmark_aiter_reduce_scatter.py \
      --shapes "8192,4096;16384,4096" --dtype float16
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist

Shape = tuple[int, ...]


def parse_shape_list(value: str) -> list[Shape]:
    shapes: list[Shape] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        shape = tuple(int(dim.strip()) for dim in item.split(",") if dim.strip())
        if not shape or any(dim <= 0 for dim in shape):
            raise argparse.ArgumentTypeError(f"invalid shape: {item!r}")
        shapes.append(shape)
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return shapes


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RCCL vs AITER reduce-scatter for CP shapes."
    )
    parser.add_argument(
        "--backend",
        default="cpu:gloo,cuda:nccl",
        help="Process group backend for torch.distributed.",
    )
    parser.add_argument(
        "--shapes",
        type=parse_shape_list,
        default=parse_shape_list("8192,4096;16384,4096"),
        help='Semicolon-separated full input shapes, e.g. "8192,4096;16384,4096".',
    )
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--custom-ar-max-size-bytes",
        type=int,
        default=0,
        help=(
            "AITER custom collective cutoff. Zero selects the largest input; "
            "a positive value is forwarded through AITER_CUSTOM_AR_MAX_SIZE."
        ),
    )
    parser.add_argument(
        "--max-size-bytes",
        type=int,
        default=0,
        help="AITER IPC pool size. Zero selects twice the largest input.",
    )
    return parser.parse_args()


def install_aiter_aot_import_shims() -> None:
    if os.getenv("AITER_AOT_IMPORT") != "1":
        return

    import aiter
    from aiter.jit.utils.torch_guard import torch_compile_guard
    from aiter.ops import custom_all_reduce
    from aiter.ops.quant import get_hip_quant

    aiter.torch_compile_guard = torch_compile_guard
    aiter.get_hip_quant = get_hip_quant
    for name in dir(custom_all_reduce):
        if not name.startswith("_"):
            setattr(aiter, name, getattr(custom_all_reduce, name))


def numel(shape: Shape) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def time_us(fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples), statistics.mean(samples)


def sync_max(value: float, device: torch.device, pg: dist.ProcessGroup) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=pg)
    return float(tensor.item())


def main() -> None:
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    input_bytes = [numel(shape) * dtype_size for shape in args.shapes]
    largest_input = max(input_bytes)
    custom_ar_max_size = args.custom_ar_max_size_bytes or largest_input
    max_size = args.max_size_bytes or largest_input * 2
    if custom_ar_max_size > max_size:
        raise ValueError(
            "--custom-ar-max-size-bytes cannot exceed --max-size-bytes: "
            f"{custom_ar_max_size} > {max_size}"
        )
    os.environ["AITER_CUSTOM_AR_MAX_SIZE"] = str(custom_ar_max_size)

    dist.init_process_group(backend=args.backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    pg = dist.group.WORLD

    install_aiter_aot_import_shims()
    from aiter.dist.device_communicators.custom_all_reduce import (
        CustomAllreduce as AiterCustomAllreduce,
    )

    gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    comm = AiterCustomAllreduce(group=gloo_group, device=device, max_size=max_size)

    rows: list[dict[str, object]] = []
    for shape, size_bytes in zip(args.shapes, input_bytes):
        if shape[0] % world_size != 0:
            raise ValueError(
                f"shape[0] must divide world_size={world_size}: shape={shape}"
            )
        output_shape = (shape[0] // world_size,) + shape[1:]
        inp = torch.full(shape, rank + 1, dtype=dtype, device=device)
        rccl_out = torch.empty(output_shape, dtype=dtype, device=device)
        aiter_out = torch.empty_like(rccl_out)

        can_aiter = bool(comm.should_custom_rs(inp, dim=0))
        if not can_aiter:
            raise RuntimeError(
                "AITER rejected the requested shape; increase the custom cutoff "
                f"or IPC pool: shape={shape}, input_bytes={size_bytes}, "
                f"custom_ar_max_size={custom_ar_max_size}, max_size={max_size}"
            )

        comm.reduce_scatter(inp, aiter_out, dim=0, registered=False)
        torch.cuda.synchronize()
        expected = world_size * (world_size + 1) / 2
        max_error = float((aiter_out.float() - expected).abs().max().item())
        if max_error != 0:
            raise AssertionError(
                f"AITER output mismatch for shape={shape}, max_error={max_error}"
            )

        dist.barrier(group=pg)
        rccl_median, rccl_mean = time_us(
            lambda: dist.reduce_scatter_tensor(rccl_out, inp, group=pg),
            args.warmup,
            args.iters,
        )
        dist.barrier(group=pg)
        aiter_median, aiter_mean = time_us(
            lambda: comm.reduce_scatter(
                inp, aiter_out, dim=0, registered=False
            ),
            args.warmup,
            args.iters,
        )
        dist.barrier(group=pg)

        rccl_median = sync_max(rccl_median, device, pg)
        rccl_mean = sync_max(rccl_mean, device, pg)
        aiter_median = sync_max(aiter_median, device, pg)
        aiter_mean = sync_max(aiter_mean, device, pg)
        rows.append(
            {
                "shape": shape,
                "input_bytes": size_bytes,
                "rccl_median_us": rccl_median,
                "rccl_mean_us": rccl_mean,
                "aiter_median_us": aiter_median,
                "aiter_mean_us": aiter_mean,
                "speedup": rccl_median / aiter_median,
            }
        )

    if hasattr(comm, "close"):
        comm.close()

    if rank == 0:
        print("\nResults (maximum rank latency)")
        print(
            f"{'Shape':>18}  {'Input MiB':>10}  {'RCCL us':>10}  "
            f"{'AITER us':>10}  {'Speedup':>8}"
        )
        for row in rows:
            print(
                f"{str(row['shape']):>18}  "
                f"{row['input_bytes'] / (1024**2):>10.1f}  "
                f"{row['rccl_median_us']:>10.2f}  "
                f"{row['aiter_median_us']:>10.2f}  "
                f"{row['speedup']:>7.2f}x"
            )

    dist.barrier(group=pg)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
