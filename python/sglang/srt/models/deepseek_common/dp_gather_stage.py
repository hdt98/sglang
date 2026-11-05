"""Stage the DP MoE gather input straight into AITER's registered IPC input pool.

The served decode gather masks DP pad rows (arange + compare + masked_fill), then the
captured AITER all-gather copies the shard into the pre-registered input pool
(SGLANG_AITER_CAPTURE_COPY_IN, #43077) before allgather_vec reads it. Here one kernel
writes the masked shard into that pool and the gather reads it in place: peers read the
same registered memory as before, so the #43077 stale-replay fix still holds.

Reduce-scatter side (lever F-RS): the MoE output is written into the same pool base, so the
registered reduce-scatter reads it in place instead of copying it in. Pool reuse within a
layer is ordered on one stream: all-gather reads the pool (its end barrier releases it), the
MoE writes it, the reduce-scatter reads it.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# MoE output destination staged by the transport for the next fused_moe call, and how often
# the registered reduce-scatter found the MoE output in the pool (counted at call/capture time).
_moe_output: Optional[torch.Tensor] = None
rs_route = {"pool": 0, "served": 0}


def stage_moe_output(dst: Optional[torch.Tensor]):
    global _moe_output
    _moe_output = dst


def take_moe_output() -> Optional[torch.Tensor]:
    global _moe_output
    out, _moe_output = _moe_output, None
    return out


@triton.jit
def _stage_masked_rows_kernel(
    src_ptr, dst_ptr, n_real_ptr, H, src_stride, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H
    keep = row < tl.load(n_real_ptr)
    x = tl.load(src_ptr + row * src_stride + offs, mask=mask & keep, other=0.0)
    tl.store(dst_ptr + row * H + offs, x, mask=mask)


def pool_view(ca_comm, local: torch.Tensor) -> Optional[torch.Tensor]:
    """A tensor over the registered input pool's base shaped like ``local``, or None when
    the pool has no torch backing (raw hipMalloc) or is too small."""
    buf = ca_comm._pool["input"]
    nbytes = local.numel() * local.element_size()
    if getattr(buf, "_buffer", None) is None or nbytes > buf.max_size:
        return None
    return buf.tensor[:nbytes].view(local.dtype).view(local.shape)


def stage_masked_rows(local: torch.Tensor, n_real: torch.Tensor, dst: torch.Tensor):
    """dst[r] = local[r] if r < n_real else 0, matching masked_fill_ bitwise."""
    rows, h = local.shape
    if rows == 0:
        return
    block = 1024
    _stage_masked_rows_kernel[(rows, triton.cdiv(h, block))](
        local, dst, n_real, h, local.stride(0), BLOCK=block, num_warps=4
    )
