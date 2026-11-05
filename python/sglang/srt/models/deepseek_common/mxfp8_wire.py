"""MXFP8 wire for the decode DP MoE gather: each rank quantizes its own rows and routes them,
then gathers one packed row per token instead of the bf16 row.

Packed row (ROW_BYTES): HIDDEN fp8 e4m3 values, HIDDEN/32 e8m0 scales, TOPK int32 ids,
TOPK fp32 weights, zero pad to a multiple of 16 bytes (the AITER all-gather contract).
The scale is aiter's default MX scale (RoundUp, ceil_pow2(max(amax, 1e-10) / 448)), so stage 1
can take the values and scales as already quantized (fused_moe's MXFP8 passthrough branch).

The transport stages the gathered scale for the MoE runner and the gathered top-k for the MoE
router: both are consumed once, by the next MoE call on this stream.
"""

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

HIDDEN = 5120
GROUP = 32
GROUPS = HIDDEN // GROUP
TOPK = 7  # six routed experts plus the fused shared expert
_PAYLOAD = HIDDEN + GROUPS + 8 * TOPK
ROW_BYTES = (_PAYLOAD + 15) // 16 * 16
_SCALE_OFF = HIDDEN
_IDS_OFF = HIDDEN + GROUPS
_W_OFF = _IDS_OFF + 4 * TOPK
_BLOCK_G = 32


class GatheredRoute(NamedTuple):
    q: torch.Tensor  # [rows, HIDDEN] fp8 e4m3
    scale: torch.Tensor  # [rows, GROUPS] e8m0
    topk_weights: torch.Tensor  # [rows, TOPK] fp32
    topk_ids: torch.Tensor  # [rows, TOPK] int32


_staged_route: Optional[GatheredRoute] = None
_staged_scale: Optional[torch.Tensor] = None


def stage_route(route: Optional[GatheredRoute]):
    global _staged_route
    _staged_route = route


def take_route() -> Optional[GatheredRoute]:
    global _staged_route
    route, _staged_route = _staged_route, None
    return route


def stage_scale(scale: Optional[torch.Tensor]):
    global _staged_scale
    _staged_scale = scale


def take_scale() -> Optional[torch.Tensor]:
    global _staged_scale
    scale, _staged_scale = _staged_scale, None
    return scale


@triton.jit
def _pack_kernel(
    x_ptr, n_real_ptr, ids_ptr, w_ptr, out_ptr, x_stride, ids_stride, w_stride,
    HIDDEN: tl.constexpr, ROW_BYTES: tl.constexpr, IDS_OFF: tl.constexpr,
    W_OFF: tl.constexpr, TOPK: tl.constexpr, BLOCK_G: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
    offs = g[:, None] * 32 + tl.arange(0, 32)[None, :]
    keep = row < tl.load(n_real_ptr)
    x = tl.load(x_ptr + row * x_stride + offs, mask=keep, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    bits = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    exp = tl.where(((bits & 0x7FFFFF) != 0) & (exp < 0xFF), exp + 1, exp)
    inv = ((254 - exp) << 23).to(tl.float32, bitcast=True)
    q = (x * inv[:, None]).to(tl.float8e4nv)
    base = out_ptr + row * ROW_BYTES
    tl.store(base + offs, q.to(tl.uint8, bitcast=True))
    tl.store(base + HIDDEN + g, exp.to(tl.uint8))
    if tl.program_id(1) == 0:
        k = tl.arange(0, 8)
        kmask = k < TOPK
        ids = tl.load(ids_ptr + row * ids_stride + k, mask=kmask)
        w = tl.load(w_ptr + row * w_stride + k, mask=kmask)
        tl.store((base + IDS_OFF).to(tl.pointer_type(tl.int32)) + k, ids, mask=kmask)
        tl.store((base + W_OFF).to(tl.pointer_type(tl.float32)) + k, w, mask=kmask)


@triton.jit
def _unpack_kernel(
    src_ptr, q_ptr, s_ptr, ids_ptr, w_ptr,
    HIDDEN: tl.constexpr, GROUPS: tl.constexpr, ROW_BYTES: tl.constexpr,
    IDS_OFF: tl.constexpr, W_OFF: tl.constexpr, TOPK: tl.constexpr, BLOCK_G: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
    offs = g[:, None] * 32 + tl.arange(0, 32)[None, :]
    base = src_ptr + row * ROW_BYTES
    tl.store(q_ptr + row * HIDDEN + offs, tl.load(base + offs))
    tl.store(s_ptr + row * GROUPS + g, tl.load(base + HIDDEN + g))
    if tl.program_id(1) == 0:
        k = tl.arange(0, 8)
        kmask = k < TOPK
        ids = tl.load((base + IDS_OFF).to(tl.pointer_type(tl.int32)) + k, mask=kmask)
        w = tl.load((base + W_OFF).to(tl.pointer_type(tl.float32)) + k, mask=kmask)
        tl.store(ids_ptr + row * TOPK + k, ids, mask=kmask)
        tl.store(w_ptr + row * TOPK + k, w, mask=kmask)


def packed_shape(rows: int) -> tuple:
    """The packed rows as a bf16 tensor shape: what the AITER all-gather moves."""
    return (rows, ROW_BYTES // 2)


def pack_rows(local, n_real, topk_ids, topk_weights, out_u8):
    """out_u8[r] = packed MXFP8 row r of local (rows >= n_real quantize as zeros)."""
    rows = local.shape[0]
    if rows == 0:
        return
    assert local.shape[1] == HIDDEN and local.dtype == torch.bfloat16
    assert topk_ids.shape == (rows, TOPK) and topk_ids.dtype == torch.int32
    assert topk_weights.shape == (rows, TOPK) and topk_weights.dtype == torch.float32
    assert out_u8.dtype == torch.uint8 and out_u8.is_contiguous()
    assert out_u8.numel() == rows * ROW_BYTES
    _pack_kernel[(rows, GROUPS // _BLOCK_G)](
        local, n_real, topk_ids, topk_weights, out_u8, local.stride(0),
        topk_ids.stride(0), topk_weights.stride(0), HIDDEN=HIDDEN, ROW_BYTES=ROW_BYTES,
        IDS_OFF=_IDS_OFF, W_OFF=_W_OFF, TOPK=TOPK, BLOCK_G=_BLOCK_G, num_warps=4,
    )


def unpack_rows(gathered_u8) -> GatheredRoute:
    rows = gathered_u8.numel() // ROW_BYTES
    device = gathered_u8.device
    q = torch.empty((rows, HIDDEN), dtype=torch.uint8, device=device)
    scale = torch.empty((rows, GROUPS), dtype=torch.uint8, device=device)
    ids = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
    weights = torch.empty((rows, TOPK), dtype=torch.float32, device=device)
    if rows:
        _unpack_kernel[(rows, GROUPS // _BLOCK_G)](
            gathered_u8, q, scale, ids, weights, HIDDEN=HIDDEN, GROUPS=GROUPS,
            ROW_BYTES=ROW_BYTES, IDS_OFF=_IDS_OFF, W_OFF=_W_OFF, TOPK=TOPK,
            BLOCK_G=_BLOCK_G, num_warps=4,
        )
    return GatheredRoute(
        q.view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu), weights, ids
    )


def reference_pack(local: torch.Tensor, n_real: int, topk_ids, topk_weights):
    """Host reference of pack_rows (fp32 math, as the kernel): (q, scale_bytes)."""
    x = local.float().clone()
    x[n_real:] = 0
    g = x.view(-1, GROUPS, GROUP)
    amax = g.abs().amax(-1).clamp_min(1e-10)
    bits = (amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)).view(torch.int32)
    exp = (bits >> 23) & 0xFF
    exp = torch.where(((bits & 0x7FFFFF) != 0) & (exp < 0xFF), exp + 1, exp)
    inv = ((254 - exp) << 23).view(torch.float32)
    q = (g * inv[..., None]).view(-1, HIDDEN).to(torch.float8_e4m3fn)
    return q, exp.to(torch.uint8)
