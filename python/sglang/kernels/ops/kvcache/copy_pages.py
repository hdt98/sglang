"""Copy whole page envelopes of the unified pool by physical page id."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_pages_kernel(
    buf_ptr,
    tgt_ptr,
    src_ptr,
    page_words,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    blk = tl.program_id(1)
    tgt = tl.load(tgt_ptr + m).to(tl.int64)
    src = tl.load(src_ptr + m).to(tl.int64)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < page_words
    values = tl.load(buf_ptr + src * page_words + offs, mask=mask)
    tl.store(buf_ptr + tgt * page_words + offs, values, mask=mask)


_BLOCK = 2048


def copy_pages(
    raw: torch.Tensor,
    tgt_pages: torch.Tensor,
    src_pages: torch.Tensor,
    num_pages: int,
    page_bytes: int,
) -> None:
    """Copy the listed physical PAGE envelopes of the uint8 pool raw."""
    m = int(tgt_pages.numel())
    if m == 0:
        return
    assert raw.dtype == torch.uint8, f"expected uint8 pool, got {raw.dtype}"
    assert page_bytes % 8 == 0, f"page_bytes {page_bytes} not int64-aligned"
    page_words = page_bytes // 8
    words = raw[: num_pages * page_bytes].view(torch.int64)
    grid = (m, triton.cdiv(page_words, _BLOCK))
    _copy_pages_kernel[grid](
        words,
        tgt_pages.to(torch.int64),
        src_pages.to(torch.int64),
        page_words,
        BLOCK=_BLOCK,
    )
