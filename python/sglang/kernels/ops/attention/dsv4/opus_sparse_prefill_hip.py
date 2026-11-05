"""DeepSeek-V4 sparse prefill attention on gfx950 through aiter's OPUS kernel.

OPUS attends over two bf16 row sources, each addressed by a CSR index list. The
caller dequantizes rows into bf16 workspaces and builds padded per-query lists
(``SparsePrefillChunkCache``); this module turns those lists into CSR and runs
the kernel.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# (int32 indices, int32 indptr [T + 1])
Csr = Tuple[torch.Tensor, torch.Tensor]

# Activation evidence for the aiter PR #6927 module overlay: one line per process
# on the first H>32 dispatch, naming the sha of the module file that serves it.
_PR6927_MODULE = "/sgl-workspace/aiter/aiter/jit/module_mla_v4_prefill_opus.so"
_PR6927_SHA = "648ee1d230313e9538de3c9d3c82da3d2700c30ebe5df7139c34c8d4df6eda74"
_pr6927_reported = False


def _report_pr6927_module(num_heads: int) -> None:
    global _pr6927_reported
    if _pr6927_reported or num_heads <= 32:
        return
    _pr6927_reported = True
    import hashlib
    import os
    import sys

    with open(_PR6927_MODULE, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    verdict = "ACTIVE" if sha == _PR6927_SHA else "REFUSE"
    print(
        f"[OPUS-PR6927] {verdict} H={num_heads} module_sha={sha} pid={os.getpid()}",
        file=sys.stderr,
        flush=True,
    )


@triton.jit
def _combined_to_csr_kernel(
    indices_ptr,
    indices_stride,
    lens_ptr,
    indptr_ptr,
    out_ptr,
    row_start,
    row_end,
    BLOCK: tl.constexpr,
):
    # OPUS reads every CSR entry, so -1 holes inside the scanned prefix and
    # entries outside [row_start, row_end) are dropped here.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    length = tl.load(lens_ptr + row)
    idx = tl.load(
        indices_ptr + row * indices_stride + cols, mask=cols < length, other=-1
    )
    keep = (idx >= row_start) & (idx < row_end)
    dst = tl.cumsum(keep.to(tl.int32), axis=0) - 1
    start = tl.load(indptr_ptr + row)
    tl.store(out_ptr + start + dst, idx - row_start, mask=keep)


def combined_to_csr(
    indices: torch.Tensor, lens: torch.Tensor, row_start: int, row_end: int
) -> Csr:
    """int32 [T, W] padded index lists with per-row scanned lengths [T] -> the CSR
    (T * W capacity) of their entries in [row_start, row_end), rebased to row_start."""
    num_rows, width = indices.shape
    device = indices.device
    cols = torch.arange(width, device=device, dtype=torch.int32)
    keep = (
        (cols[None, :] < lens[:, None]) & (indices >= row_start) & (indices < row_end)
    )
    indptr = torch.zeros(num_rows + 1, dtype=torch.int32, device=device)
    torch.cumsum(keep.sum(dim=1, dtype=torch.int32), dim=0, out=indptr[1:])
    csr = torch.empty(max(num_rows * width, 1), dtype=torch.int32, device=device)
    if num_rows > 0:
        _combined_to_csr_kernel[(num_rows,)](
            indices,
            indices.stride(0),
            lens,
            indptr,
            csr,
            row_start,
            row_end,
            BLOCK=triton.next_power_of_2(width),
        )
    return csr, indptr


def opus_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    csr: Csr,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    extend_kv: Optional[torch.Tensor] = None,
    extend_csr: Optional[Csr] = None,
) -> torch.Tensor:
    """Attention of q [T, H, 512] bf16 over the bf16 rows of kv [R, 512] named by
    csr, then those of extend_kv named by extend_csr, with a per-head sink."""
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus

    _report_pr6927_module(q.shape[1])
    indices, indptr = csr
    if extend_kv is None:
        # every query reads an empty row of the unused extend source
        extend_kv = kv[:1]
        extend_csr = (indices[:1], torch.zeros_like(indptr))
    extend_indices, extend_indptr = extend_csr
    return pa_sparse_prefill_opus(
        q.contiguous(),
        kv,
        indices,
        indptr,
        extend_kv,
        extend_indices,
        extend_indptr,
        attn_sink,
        softmax_scale,
    )
