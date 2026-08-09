"""Mori SHMEM-based P2P collectives for DCP on AMD GPUs.

Replaces NCCL all-gather with zero-copy P2P writes over XGMI using Mori SHMEM.
Each rank writes its data to a symmetric memory buffer, then all ranks barrier
on-stream and read each other's data via P2P.  The barrier is a lightweight
GPU kernel (shmem_barrier_on_stream) instead of a full NCCL collective.

Only used for decode (small, fixed batch size).  Prefill falls back to NCCL
because the batch size can be very large (chunked_prefill_size).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_state: Optional[dict] = None

_workspace_slot_counter: int = 0
_v2_failed: bool = False

def is_mori_dcp_available() -> bool:
    import os
    if os.environ.get("SGLANG_DCP_DISABLE_MORI", "0") == "1":
        return False
    try:
        import mori  # noqa: F401
        return True
    except ImportError:
        return False


def init_mori_dcp(
    dcp_group,
    max_batch_size: int,
    num_heads: int,
    d_total: int,
    dtype: torch.dtype,
) -> bool:
    """Initialize Mori SHMEM for DCP.

    Uses the TP group (all ranks) for Mori SHMEM init because Mori only
    supports a single process group.  DCP communication is then a subset
    of the TP group — we only read from DCP peers.
    """
    global _state

    if _state is not None:
        return False

    try:
        import mori
    except ImportError:
        logger.info("[Mori DCP] mori not installed, skipping init")
        return False

    dcp_world_size = dcp_group.world_size
    dcp_rank = dcp_group.rank_in_group

    if dcp_world_size <= 1:
        return False

    # Use the TP group (all ranks) for Mori SHMEM init.
    from sglang.srt.distributed.parallel_state import get_tp_group
    tp_group = get_tp_group()
    tp_world_size = tp_group.world_size
    tp_rank = tp_group.rank_in_group

    global_rank = torch.distributed.get_rank()
    group_name = "mori_dcp"
    cpu_group = tp_group.cpu_group

    try:
        torch._C._distributed_c10d._register_process_group(group_name, cpu_group)
    except Exception as e:
        if "already registered" not in str(e):
            raise
    mori.shmem.shmem_torch_process_group_init(group_name)

    # DCP peers: the dcp_world_size ranks starting at (global_rank - dcp_rank)
    group_start = global_rank - dcp_rank
    dcp_peers = list(range(group_start, group_start + dcp_world_size))

    logger.info(
        "[Mori DCP] SHMEM init group=%s tp_rank=%d/%d dcp_rank=%d/%d dcp_peers=%s max_bs=%d heads=%d d=%d",
        group_name, tp_rank, tp_world_size, dcp_rank, dcp_world_size, dcp_peers,
        max_batch_size, num_heads, d_total,
    )

    # Symmetric tensor: each rank gets its own [max_bs, H, D_total] buffer.
    # symm_mori_shmem_tensor gives a P2P view of a peer's buffer.
    local_buf = mori.shmem.mori_shmem_create_tensor(
        (max_batch_size, num_heads, d_total), dtype
    )

    peer_bufs = []
    for peer in range(tp_world_size):
        if peer == tp_rank:
            peer_bufs.append(local_buf)
        else:
            peer_bufs.append(
                mori.shmem.symm_mori_shmem_tensor(local_buf, peer)
            )

    # Pre-allocate gathered output: [dcp_world_size, max_bs, H, D_total]
    gathered_buf = torch.empty(
        dcp_world_size, max_batch_size, num_heads, d_total,
        dtype=dtype, device=torch.cuda.current_device(),
    )

    _state = {
        "local_buf": local_buf,
        "peer_bufs": peer_bufs,
        "gathered_buf": gathered_buf,
        "world_size": tp_world_size,
        "dcp_world_size": dcp_world_size,
        "dcp_peers": dcp_peers,
        "dcp_rank": dcp_rank,
        "rank": tp_rank,
        "max_batch_size": max_batch_size,
        "num_heads": num_heads,
        "d_total": d_total,
        "dtype": dtype,
        "group_name": group_name,
    }

    logger.info("[Mori DCP] init complete for %d ranks", dcp_world_size)
    return True


def mori_all_gather_q_if_available(
    q_nope_out: torch.Tensor,
    q_pe: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Try Mori SHMEM P2P all-gather; return None to fall back to NCCL."""
    if _state is None:
        return None

    batch_size = q_nope_out.shape[0]
    if batch_size > _state["max_batch_size"]:
        return None  # Fall back to NCCL for large batches (prefill)

    import mori

    dcp_world_size = _state["dcp_world_size"]
    dcp_peers = _state["dcp_peers"]
    local_buf = _state["local_buf"]
    peer_bufs = _state["peer_bufs"]
    gathered_buf = _state["gathered_buf"]

    num_heads = q_nope_out.shape[1]
    d_nope = q_nope_out.shape[2]
    d_pe = q_pe.shape[2]

    # Combine [B, H, D_nope] + [B, H, D_pe] -> [B, H, D_total]
    combined = torch.cat([q_nope_out, q_pe], dim=-1)

    # Write local Q to symmetric buffer (local memory write)
    local_buf[:batch_size].copy_(combined)

    # Device-side barrier: ensure all ranks finished writing
    mori.shmem.shmem_barrier_on_stream(torch.cuda.current_stream())

    # Read only DCP peers' data (local read + P2P reads)
    for i, peer in enumerate(dcp_peers):
        gathered_buf[i, :batch_size].copy_(peer_bufs[peer][:batch_size])

    # [dcp_size, B, H, D_total] -> [B, dcp_size*H, D_total]
    gathered = gathered_buf[:dcp_world_size, :batch_size].permute(1, 0, 2, 3).reshape(
        batch_size, dcp_world_size * num_heads, -1
    )

    q_nope_gathered, q_pe_gathered = gathered.split([d_nope, d_pe], dim=-1)
    return q_nope_gathered, q_pe_gathered


# ---- Output / LSE merge via Mori SHMEM ----

_output_state: Optional[dict] = None


def init_mori_dcp_output(
    dcp_group,
    max_batch_size: int,
    num_heads: int,
    d_out: int,
    dtype: torch.dtype,
) -> bool:
    """Initialize Mori SHMEM symmetric buffers for DCP output/LSE merge.

    Creates per-rank symmetric buffers for:
    - output: [max_bs, num_heads, d_out]
    - lse:    [max_bs, num_heads]  (always fp32)

    Each rank writes its partial attention output + LSE to its local buffer,
    then all ranks barrier on-stream and read peers' data via P2P.
    The combine is done locally with the existing dcp_lse_combine_triton kernel.
    """
    global _output_state

    if _output_state is not None:
        return False

    try:
        import mori
    except ImportError:
        logger.info("[Mori DCP] mori not installed, skipping output init")
        return False

    dcp_world_size = dcp_group.world_size
    dcp_rank = dcp_group.rank_in_group

    if dcp_world_size <= 1:
        return False

    # Reuse the same group already initialized for Q all-gather (TP group).
    group_name = "mori_dcp"
    global_rank = torch.distributed.get_rank()
    group_start = global_rank - dcp_rank
    dcp_peers = list(range(group_start, group_start + dcp_world_size))

    from sglang.srt.distributed.parallel_state import get_tp_group
    tp_world_size = get_tp_group().world_size
    tp_rank = get_tp_group().rank_in_group

    # Output symmetric buffer: [max_bs, H, D]
    local_out = mori.shmem.mori_shmem_create_tensor(
        (max_batch_size, num_heads, d_out), dtype
    )
    peer_outs = []
    for peer in range(tp_world_size):
        if peer == tp_rank:
            peer_outs.append(local_out)
        else:
            peer_outs.append(mori.shmem.symm_mori_shmem_tensor(local_out, peer))

    # LSE symmetric buffer: [max_bs, H] (fp32)
    local_lse = mori.shmem.mori_shmem_create_tensor(
        (max_batch_size, num_heads), torch.float32
    )
    peer_lses = []
    for peer in range(tp_world_size):
        if peer == tp_rank:
            peer_lses.append(local_lse)
        else:
            peer_lses.append(mori.shmem.symm_mori_shmem_tensor(local_lse, peer))

    # Pre-allocate gathered buffers: [dcp_size, max_bs, H, D] and [dcp_size, max_bs, H]
    gathered_out = torch.empty(
        dcp_world_size, max_batch_size, num_heads, d_out,
        dtype=dtype, device=torch.cuda.current_device(),
    )
    gathered_lse = torch.empty(
        dcp_world_size, max_batch_size, num_heads,
        dtype=torch.float32, device=torch.cuda.current_device(),
    )

    _output_state = {
        "local_out": local_out,
        "peer_outs": peer_outs,
        "local_lse": local_lse,
        "peer_lses": peer_lses,
        "gathered_out": gathered_out,
        "gathered_lse": gathered_lse,
        "world_size": tp_world_size,
        "dcp_world_size": dcp_world_size,
        "dcp_peers": dcp_peers,
        "dcp_rank": dcp_rank,
        "rank": tp_rank,
        "max_batch_size": max_batch_size,
        "num_heads": num_heads,
        "d_out": d_out,
        "dtype": dtype,
    }

    logger.info("[Mori DCP] output/LSE init complete for %d ranks", dcp_world_size)
    return True


def mori_lse_combine_if_available(
    attn_output: torch.Tensor,
    lse: torch.Tensor,
    is_lse_base_on_e: bool = False,
) -> Optional[torch.Tensor]:
    """Try Mori SHMEM v2 output/LSE merge; return None to fall back to NCCL.

    Uses producer-direct P2P writes with atomic flag signaling (no barrier),
    replacing the v1 shmem_barrier_on_stream approach and the NCCL
    all-gather + reduce-scatter path.  The v2 kernels live in
    shared_output.py (ported from PR #32851).
    """
    global _workspace_slot_counter, _v2_failed

    if _v2_failed:
        return None

    # v2 requires BF16 output and FP32 LSE
    if attn_output.dtype != torch.bfloat16:
        return None
    if lse.dtype != torch.float32:
        lse = lse.to(torch.float32)
    if not attn_output.is_contiguous():
        attn_output = attn_output.contiguous()
    if not lse.is_contiguous():
        lse = lse.contiguous()

    from sglang.srt.runtime_context import get_parallel
    dcp_group = get_parallel().dcp_group

    slot = _workspace_slot_counter % 2  # DCP_OUTPUT_MORI_SLOTS = 2
    _workspace_slot_counter += 1

    try:
        from sglang.srt.layers.dcp.shared_output import dcp_output_mori_lse_reduce
        return dcp_output_mori_lse_reduce(
            attn_output,
            lse,
            dcp_group,
            is_lse_base_on_e=is_lse_base_on_e,
            workspace_slot=slot,
        )
    except Exception as e:
        logger.warning(
            "Mori SHMEM v2 output/LSE merge failed, falling back to NCCL: %s", e
        )
        return None
