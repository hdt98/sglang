# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Producer-direct DCP Output/LSE merge over Mori SHMEM P2P storage (AMD).

Replaces the CUDA VMM peer-memory path (PR #32851) with Mori SHMEM symmetric
buffers on AMD GPUs.  Each rank allocates a symmetric buffer; P2P views of
DCP peers' buffers are obtained via mori.shmem.symm_mori_shmem_tensor.

Key optimization vs the previous mori_dcp.py barrier approach:
  - Producers write directly to peer buffers (zero-copy P2P writes)
  - Atomic flag signaling (scope="sys") replaces shmem_barrier_on_stream
  - Consumers read from local memory (no P2P reads needed during merge)

The three Triton kernels are adapted from PR #32851's shared_output.py:
  - _direct_publish_kernel: writes partial output+LSE to all peers
  - _direct_signal_kernel: atomic flag signal to all peers
  - _direct_consumer_merge_kernel: spin-wait on flags, then LSE-weighted merge

AMD/HIP adaptations:
  - PTX _trap_if_nonzero replaced with no-op (spin max_spins is the safeguard)
  - Pointer table (int64 device tensor) replaces rank-major virtual address space
  - Consumer kernel reads local buffer directly (unchanged from PR)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.profiler import record_function

from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

_HEADER_BYTES = 256
_MAX_FENCE_SPINS = 100_000_000
DCP_OUTPUT_MORI_MAX_ROWS = 512
DCP_OUTPUT_MORI_SLOTS = 2
_logged_rows: set[int] = set()

# ---------------------------------------------------------------------------
# Mori SHMEM initialization (shared across output and query workspaces)
# ---------------------------------------------------------------------------

_mori_initialized = False
_mori_group_name = "mori_dcp"


def _ensure_mori_shmem_init(group: GroupCoordinator) -> tuple[int, int, list[int]]:
    """Initialize Mori SHMEM (once per process) and return DCP peer mapping.

    Mori SHMEM requires a single process group; we use the TP group and map
    DCP peers into TP-rank space.  Returns (tp_rank, tp_world_size, dcp_peers)
    where dcp_peers[i] is the TP rank of DCP rank i.
    """
    global _mori_initialized
    import mori

    from sglang.srt.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    tp_world_size = tp_group.world_size
    tp_rank = tp_group.rank_in_group

    dcp_world_size = group.world_size
    dcp_rank = group.rank_in_group
    global_rank = torch.distributed.get_rank()
    group_start = global_rank - dcp_rank
    dcp_peers = list(range(group_start, group_start + dcp_world_size))

    if not _mori_initialized:
        cpu_group = tp_group.cpu_group
        try:
            torch._C._distributed_c10d._register_process_group(
                _mori_group_name, cpu_group
            )
        except Exception as e:
            if "already registered" not in str(e):
                raise
        mori.shmem.shmem_torch_process_group_init(_mori_group_name)
        _mori_initialized = True
        logger.info(
            "[Mori DCP] SHMEM initialized: group=%s tp_rank=%d/%d "
            "dcp_rank=%d/%d dcp_peers=%s",
            _mori_group_name, tp_rank, tp_world_size,
            dcp_rank, dcp_world_size, dcp_peers,
        )

    return tp_rank, tp_world_size, dcp_peers


# ---------------------------------------------------------------------------
# MoriPeerBuffer — replaces RankMajorPeerBuffer from peer_memory.py
# ---------------------------------------------------------------------------

@dataclass
class MoriPeerBuffer:
    """Rank-major peer buffer using Mori SHMEM symmetric memory.

    Each rank allocates a symmetric byte buffer.  P2P views of DCP peers'
    buffers are obtained via symm_mori_shmem_tensor.  Because peer buffers
    live at different virtual addresses (unlike CUDA VMM's single contiguous
    mapping), a device-side pointer table is used for kernel-level access.

    Attributes:
        local_view: Local symmetric buffer as uint8 (flat byte array).
        peer_views: P2P views indexed by DCP rank (peer_views[my_rank] is local).
        peer_ptrs: int64 device tensor of base pointers (one per DCP rank).
        requested_bytes: Usable bytes per rank (excluding alignment padding).
        rank: DCP rank.
        world_size: DCP world size.
    """

    local_view: torch.Tensor
    peer_views: list[torch.Tensor]
    peer_ptrs: torch.Tensor
    requested_bytes: int
    rank: int
    world_size: int
    _closed: bool = False

    @property
    def bytes_per_rank(self) -> int:
        return self.requested_bytes

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize()
        del self.peer_ptrs
        del self.peer_views
        del self.local_view

    def make_typed_ptr_table(self, offset: int, dtype: torch.dtype) -> torch.Tensor:
        """Create an int64 device tensor of typed peer data pointers.

        Each entry is peer_views[i].data_ptr() + offset, pre-computed so
        kernels can load and cast directly to a typed pointer.
        """
        return torch.tensor(
            [v.data_ptr() + offset for v in self.peer_views],
            dtype=torch.int64,
            device=self.local_view.device,
        )


def create_mori_peer_buffer(
    requested_bytes: int,
    *,
    group: GroupCoordinator,
    device: torch.device,
) -> MoriPeerBuffer:
    """Collectively create a Mori SHMEM peer buffer for DCP communication."""
    import mori

    if group.world_size <= 1:
        raise RuntimeError("Mori SHMEM peer buffer requires dcp_size > 1")

    tp_rank, tp_world_size, dcp_peers = _ensure_mori_shmem_init(group)
    dcp_world_size = group.world_size
    dcp_rank = group.rank_in_group

    # Align to 256-byte boundary for symmetric heap alignment
    aligned_bytes = (requested_bytes + 255) & ~255

    local_view = mori.shmem.mori_shmem_create_tensor(
        (aligned_bytes,), torch.uint8
    )

    peer_views: list[torch.Tensor] = []
    for dcp_r in range(dcp_world_size):
        tp_r = dcp_peers[dcp_r]
        if tp_r == tp_rank:
            peer_views.append(local_view)
        else:
            peer_views.append(
                mori.shmem.symm_mori_shmem_tensor(local_view, tp_r)
            )

    peer_ptrs = torch.tensor(
        [v.data_ptr() for v in peer_views],
        dtype=torch.int64,
        device=device,
    )

    local_view.zero_()
    torch.cuda.synchronize()
    dist.barrier(group=group.cpu_group)

    return MoriPeerBuffer(
        local_view=local_view,
        peer_views=peer_views,
        peer_ptrs=peer_ptrs,
        requested_bytes=requested_bytes,
        rank=dcp_rank,
        world_size=dcp_world_size,
    )


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _trap_if_nonzero(value):
    # PTX trap replaced with no-op for AMD/HIP portability.
    # The spin loop's max_spins limit is the safeguard; a timeout
    # indicates a peer crash, not a recoverable error.
    return value


@triton.jit
def _direct_publish_kernel(
    partial_output,
    partial_lse,
    peer_output_ptrs,
    peer_lse_ptrs,
    output_token_stride,
    output_head_stride,
    output_dim_stride,
    lse_token_stride,
    lse_head_stride,
    peer_source_stride,
    peer_token_stride,
    peer_head_stride,
    peer_dim_stride,
    peer_lse_source_stride,
    peer_lse_token_stride,
    peer_lse_head_stride,
    my_rank: tl.constexpr,
    local_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_items: tl.constexpr,
    head_block_size: tl.constexpr,
):
    """Write local partial output+LSE to one peer's buffer (P2P write).

    Grid: (rows, world_size).  Each program writes to one destination rank.
    The peer buffer pointer is loaded from a pointer table because Mori SHMEM
    peer buffers live at separate addresses (unlike CUDA VMM's contiguous map).
    """
    token_idx = tl.program_id(0).to(tl.int64)
    destination_rank = tl.program_id(1).to(tl.int64)

    # Load peer buffer pointers from pointer table
    raw_out_ptr = tl.load(peer_output_ptrs + destination_rank)
    raw_lse_ptr = tl.load(peer_lse_ptrs + destination_rank)

    # Cast to typed pointers
    peer_outputs = raw_out_ptr.to(tl.pointer_type(tl.bfloat16))
    peer_lses = raw_lse_ptr.to(tl.pointer_type(tl.float32))

    item = tl.arange(0, block_items)
    item_mask = item < local_heads * head_dim
    local_head_idx = item // head_dim
    dim = item % head_dim
    source_head_idx = destination_rank * local_heads + local_head_idx

    # Read from local partial_output
    source_output_offset = (
        token_idx * output_token_stride
        + source_head_idx * output_head_stride
        + dim * output_dim_stride
    )
    value = tl.load(partial_output + source_output_offset, mask=item_mask)

    # Write to peer's buffer at my_rank slot (P2P write)
    destination_output_offset = (
        my_rank * peer_source_stride
        + token_idx * peer_token_stride
        + local_head_idx * peer_head_stride
        + dim * peer_dim_stride
    )
    tl.store(peer_outputs + destination_output_offset, value, mask=item_mask)

    # Same for LSE
    lse_local_head_idx = tl.arange(0, head_block_size)
    lse_mask = lse_local_head_idx < local_heads
    lse_source_head_idx = destination_rank * local_heads + lse_local_head_idx
    source_lse_offset = (
        token_idx * lse_token_stride + lse_source_head_idx * lse_head_stride
    )
    destination_lse_offset = (
        my_rank * peer_lse_source_stride
        + token_idx * peer_lse_token_stride
        + lse_local_head_idx * peer_lse_head_stride
    )
    tl.store(
        peer_lses + destination_lse_offset,
        tl.load(partial_lse + source_lse_offset, mask=lse_mask),
        mask=lse_mask,
    )


@triton.jit
def _direct_signal_kernel(
    local_epoch,
    peer_signal_ptrs,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """Signal all peers via atomic flag write (P2P atomic, scope=sys).

    Replaces shmem_barrier_on_stream with per-destination atomic_xchg.
    The loop over world_size is unrolled at compile time (constexpr).
    """
    epoch = tl.atomic_add(local_epoch, 1, sem="acq_rel", scope="gpu") + 1
    for dest in range(world_size):
        raw_ptr = tl.load(peer_signal_ptrs + dest)
        flag_ptr = raw_ptr.to(tl.pointer_type(tl.int32))
        tl.atomic_xchg(flag_ptr + my_rank, epoch, sem="release", scope="sys")


@triton.jit
def _direct_consumer_merge_kernel(
    local_outputs,
    local_lses,
    local_signals,
    local_epoch,
    merged_output,
    merged_lse,
    local_output_source_stride,
    local_output_token_stride,
    local_output_head_stride,
    local_output_dim_stride,
    local_lse_source_stride,
    local_lse_token_stride,
    local_lse_head_stride,
    merged_output_token_stride,
    merged_output_head_stride,
    merged_output_dim_stride,
    merged_lse_token_stride,
    merged_lse_head_stride,
    world_size: tl.constexpr,
    is_base_e: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
    signal_block_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    """Spin-wait for all peer signals, then LSE-weighted merge (local reads).

    Reads from local_outputs/local_lses (local memory where peers wrote via P2P).
    Reads local_signals (local flags where peers signaled via P2P atomics).
    Unchanged from PR #32851 except _trap_if_nonzero is a no-op on AMD.
    """
    token_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    expected_epoch = tl.atomic_add(local_epoch, 0, sem="acquire", scope="gpu")
    signal_source = tl.arange(0, signal_block_size)
    signal_mask = signal_source < world_size
    observed = tl.atomic_add(
        local_signals + signal_source,
        0,
        mask=signal_mask,
        sem="acquire",
        scope="sys",
    )
    pending = tl.max(tl.where(signal_mask & (observed < expected_epoch), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            local_signals + signal_source,
            0,
            mask=signal_mask,
            sem="acquire",
            scope="sys",
        )
        pending = tl.max(tl.where(signal_mask & (observed < expected_epoch), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)

    source_rank = tl.arange(0, world_size)
    lse_offset = (
        source_rank * local_lse_source_stride
        + token_idx * local_lse_token_stride
        + local_head_idx * local_lse_head_stride
    )
    lse = tl.load(local_lses + lse_offset)
    lse = tl.where(
        (lse != lse) | (lse == float("inf")),
        -float("inf"),
        lse,
    )
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    if is_base_e:
        weights = tl.exp(lse - lse_max)
        weight_sum = tl.sum(weights, axis=0)
        final_lse = tl.log(weight_sum) + lse_max
    else:
        weights = tl.exp2(lse - lse_max)
        weight_sum = tl.sum(weights, axis=0)
        final_lse = tl.log2(weight_sum) + lse_max
    weights = tl.where(weight_sum == 0.0, 0.0, weights / weight_sum)

    dim = tl.arange(0, block_dim)
    dim_mask = dim < head_dim
    output_offset = (
        source_rank[:, None] * local_output_source_stride
        + token_idx * local_output_token_stride
        + local_head_idx * local_output_head_stride
        + dim[None, :] * local_output_dim_stride
    )
    partial_output = tl.load(local_outputs + output_offset, mask=dim_mask[None, :])
    weighted_output = partial_output.to(tl.float32) * weights[:, None]
    weighted_output = tl.where(weights[:, None] == 0.0, 0.0, weighted_output)
    output = tl.sum(weighted_output, axis=0)
    merged_output_offset = (
        token_idx * merged_output_token_stride
        + local_head_idx * merged_output_head_stride
        + dim * merged_output_dim_stride
    )
    tl.store(merged_output + merged_output_offset, output, mask=dim_mask)
    merged_lse_offset = (
        token_idx * merged_lse_token_stride + local_head_idx * merged_lse_head_stride
    )
    tl.store(merged_lse + merged_lse_offset, final_lse)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

@dataclass
class DcpOutputMoriWorkspace:
    rank: int
    world_size: int
    max_rows: int
    total_heads: int
    head_dim: int
    group: GroupCoordinator
    device: torch.device
    allocation: MoriPeerBuffer
    local_partial_output: torch.Tensor
    local_partial_lse: torch.Tensor
    local_merged_output: torch.Tensor
    local_merged_lse: torch.Tensor
    local_flags: torch.Tensor
    local_epoch: torch.Tensor
    peer_output_ptrs: torch.Tensor
    peer_lse_ptrs: torch.Tensor
    peer_signal_ptrs: torch.Tensor

    @property
    def local_heads(self) -> int:
        return self.total_heads // self.world_size

    @property
    def physical_bytes_per_rank(self) -> int:
        return self.allocation.bytes_per_rank

    def merge(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        *,
        is_lse_base_on_e: bool,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.allocation.closed:
            raise RuntimeError("DCP Output/LSE Mori workspace is closed")
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError(
                "DCP Output/LSE Mori current device changed after init: "
                f"workspace={self.device}, current=cuda:{torch.cuda.current_device()}"
            )

        rows = partial_output.shape[0]
        expected_output_shape = (rows, self.total_heads, self.head_dim)
        expected_lse_shape = (rows, self.total_heads)
        if tuple(partial_output.shape) != expected_output_shape:
            raise RuntimeError(
                "DCP Output/LSE Mori partial-output shape mismatch: "
                f"expected {expected_output_shape}, got {tuple(partial_output.shape)}"
            )
        if tuple(partial_lse.shape) != expected_lse_shape:
            raise RuntimeError(
                "DCP Output/LSE Mori LSE shape mismatch: "
                f"expected {expected_lse_shape}, got {tuple(partial_lse.shape)}"
            )
        if rows > self.max_rows:
            raise RuntimeError(
                f"DCP Output/LSE Mori has {self.max_rows} rows, requested {rows}"
            )
        if partial_output.dtype != torch.bfloat16:
            raise RuntimeError(
                "DCP Output/LSE Mori currently requires BF16 attention output, "
                f"got {partial_output.dtype}"
            )
        if partial_lse.dtype != torch.float32:
            raise RuntimeError(
                f"DCP Output/LSE Mori requires FP32 LSE, got {partial_lse.dtype}"
            )
        if partial_output.device != self.device or partial_lse.device != self.device:
            raise RuntimeError(
                f"DCP Output/LSE Mori inputs must be on {self.device}, got "
                f"{partial_output.device} and {partial_lse.device}"
            )
        if not partial_output.is_contiguous() or not partial_lse.is_contiguous():
            raise RuntimeError("DCP Output/LSE Mori inputs must be contiguous")

        if rows not in _logged_rows:
            _logged_rows.add(rows)
            logger.info(
                "Executing producer-direct Mori SHMEM DCP Output/LSE for rows=%d",
                rows,
            )

        with record_function("dcp.output_lse.mori.publish"):
            _direct_publish_kernel[(rows, self.world_size)](
                partial_output,
                partial_lse,
                self.peer_output_ptrs,
                self.peer_lse_ptrs,
                partial_output.stride(0),
                partial_output.stride(1),
                partial_output.stride(2),
                partial_lse.stride(0),
                partial_lse.stride(1),
                self.local_partial_output.stride(0),
                self.local_partial_output.stride(1),
                self.local_partial_output.stride(2),
                self.local_partial_output.stride(3),
                self.local_partial_lse.stride(0),
                self.local_partial_lse.stride(1),
                self.local_partial_lse.stride(2),
                my_rank=self.rank,
                local_heads=self.local_heads,
                head_dim=self.head_dim,
                block_items=triton.next_power_of_2(self.local_heads * self.head_dim),
                head_block_size=triton.next_power_of_2(self.local_heads),
                num_warps=8,
            )
            _direct_signal_kernel[(1,)](
                self.local_epoch,
                self.peer_signal_ptrs,
                my_rank=self.rank,
                world_size=self.world_size,
            )

        output = self.local_merged_output[:rows]
        lse = self.local_merged_lse[:rows]
        with record_function("dcp.output_lse.mori.local_merge"):
            _direct_consumer_merge_kernel[(rows, self.local_heads)](
                self.local_partial_output,
                self.local_partial_lse,
                self.local_flags,
                self.local_epoch,
                output,
                lse,
                self.local_partial_output.stride(0),
                self.local_partial_output.stride(1),
                self.local_partial_output.stride(2),
                self.local_partial_output.stride(3),
                self.local_partial_lse.stride(0),
                self.local_partial_lse.stride(1),
                self.local_partial_lse.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                lse.stride(0),
                lse.stride(1),
                world_size=self.world_size,
                is_base_e=is_lse_base_on_e,
                head_dim=self.head_dim,
                block_dim=min(512, triton.next_power_of_2(self.head_dim)),
                signal_block_size=triton.next_power_of_2(self.world_size),
                max_spins=_MAX_FENCE_SPINS,
                num_warps=4,
            )
        return (output, lse) if return_lse else output

    def close(self) -> None:
        if self.allocation.closed:
            return
        torch.cuda.synchronize()
        del self.peer_signal_ptrs
        del self.peer_lse_ptrs
        del self.peer_output_ptrs
        del self.local_epoch
        del self.local_flags
        del self.local_merged_lse
        del self.local_merged_output
        del self.local_partial_lse
        del self.local_partial_output
        self.allocation.close()


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def create_dcp_output_mori_workspace(
    max_rows: int,
    total_heads: int,
    head_dim: int,
    group: GroupCoordinator,
) -> DcpOutputMoriWorkspace:
    """Collectively create one destination-owned Output/LSE workspace."""
    world_size = group.world_size
    rank = group.rank_in_group
    if world_size <= 1:
        raise RuntimeError("DCP Output/LSE Mori requires dcp_size > 1")
    if total_heads % world_size:
        raise RuntimeError(
            f"total_heads={total_heads} is not divisible by dcp_size={world_size}"
        )
    if max_rows <= 0 or total_heads <= 0 or head_dim <= 0:
        raise ValueError(
            "DCP Output/LSE Mori dimensions must be positive: "
            f"max_rows={max_rows}, total_heads={total_heads}, head_dim={head_dim}"
        )

    local_heads = total_heads // world_size
    partial_output_bytes = max_rows * total_heads * head_dim * torch.bfloat16.itemsize
    partial_lse_bytes = max_rows * total_heads * torch.float32.itemsize
    merged_output_bytes = max_rows * local_heads * head_dim * torch.bfloat16.itemsize
    merged_lse_bytes = max_rows * local_heads * torch.float32.itemsize

    partial_output_offset = _HEADER_BYTES
    partial_lse_offset = _align_up(
        partial_output_offset + partial_output_bytes, torch.float32.itemsize
    )
    merged_output_offset = _align_up(
        partial_lse_offset + partial_lse_bytes, torch.bfloat16.itemsize
    )
    merged_lse_offset = _align_up(
        merged_output_offset + merged_output_bytes, torch.float32.itemsize
    )
    requested_bytes = merged_lse_offset + merged_lse_bytes

    allocation = create_mori_peer_buffer(
        requested_bytes,
        group=group,
        device=group.device,
    )

    # Typed views into the local symmetric buffer
    local_flags = allocation.local_view[: world_size * 4].view(torch.int32)
    local_partial_output = (
        allocation.local_view[
            partial_output_offset : partial_output_offset + partial_output_bytes
        ]
        .view(torch.bfloat16)
        .view(world_size, max_rows, local_heads, head_dim)
    )
    local_partial_lse = (
        allocation.local_view[
            partial_lse_offset : partial_lse_offset + partial_lse_bytes
        ]
        .view(torch.float32)
        .view(world_size, max_rows, local_heads)
    )
    local_merged_output = (
        allocation.local_view[
            merged_output_offset : merged_output_offset + merged_output_bytes
        ]
        .view(torch.bfloat16)
        .view(max_rows, local_heads, head_dim)
    )
    local_merged_lse = (
        allocation.local_view[merged_lse_offset : merged_lse_offset + merged_lse_bytes]
        .view(torch.float32)
        .view(max_rows, local_heads)
    )

    # Pre-compute typed pointer tables for producer-direct P2P writes
    peer_output_ptrs = allocation.make_typed_ptr_table(
        partial_output_offset, torch.bfloat16
    )
    peer_lse_ptrs = allocation.make_typed_ptr_table(
        partial_lse_offset, torch.float32
    )
    peer_signal_ptrs = allocation.make_typed_ptr_table(0, torch.int32)

    return DcpOutputMoriWorkspace(
        rank=rank,
        world_size=world_size,
        max_rows=max_rows,
        total_heads=total_heads,
        head_dim=head_dim,
        group=group,
        device=group.device,
        allocation=allocation,
        local_partial_output=local_partial_output,
        local_partial_lse=local_partial_lse,
        local_merged_output=local_merged_output,
        local_merged_lse=local_merged_lse,
        local_flags=local_flags,
        local_epoch=torch.zeros(1, dtype=torch.int32, device=group.device),
        peer_output_ptrs=peer_output_ptrs,
        peer_lse_ptrs=peer_lse_ptrs,
        peer_signal_ptrs=peer_signal_ptrs,
    )


_workspaces: dict[tuple[int, int], DcpOutputMoriWorkspace] = {}
_workspace_failed = False


def get_dcp_output_mori_workspace(
    max_rows: int,
    total_heads: int,
    head_dim: int,
    group: GroupCoordinator,
    workspace_slot: int,
) -> DcpOutputMoriWorkspace:
    """Create or fetch a workspace slot; initialization failure is terminal."""
    global _workspace_failed
    if workspace_slot < 0:
        raise ValueError("DCP Output/LSE Mori workspace_slot must be nonnegative")
    if _workspace_failed:
        raise RuntimeError("DCP Output/LSE Mori workspace is unavailable")

    key = (id(group), workspace_slot)
    workspace = _workspaces.get(key)
    if workspace is not None:
        actual = (
            workspace.max_rows,
            workspace.total_heads,
            workspace.head_dim,
            workspace.world_size,
            workspace.rank,
            workspace.device,
        )
        requested = (
            max_rows,
            total_heads,
            head_dim,
            group.world_size,
            group.rank_in_group,
            group.device,
        )
        if actual != requested:
            raise RuntimeError(
                "DCP Output/LSE Mori workspace identity changed: "
                f"slot={workspace_slot}, actual={actual}, requested={requested}"
            )
        return workspace

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DCP Output/LSE Mori workspace was not initialized before CUDA graph "
            "capture"
        )
    try:
        workspace = create_dcp_output_mori_workspace(
            max_rows, total_heads, head_dim, group
        )
        _workspaces[key] = workspace
    except Exception as error:
        _workspace_failed = True
        raise RuntimeError(
            "DCP Output/LSE Mori initialization failed; refusing to fall back "
            "after selecting the Mori route"
        ) from error
    logger.info(
        "Initialized DCP Output/LSE Mori slot=%d, max_rows=%d, total_heads=%d, "
        "head_dim=%d, physical_bytes_per_rank=%d",
        workspace_slot,
        max_rows,
        total_heads,
        head_dim,
        workspace.physical_bytes_per_rank,
    )
    return workspace


def init_dcp_output_mori_workspaces(
    group: GroupCoordinator,
    total_heads: int,
    head_dim: int,
    *,
    max_rows: int = DCP_OUTPUT_MORI_MAX_ROWS,
    slots: int = DCP_OUTPUT_MORI_SLOTS,
) -> None:
    for workspace_slot in range(slots):
        get_dcp_output_mori_workspace(
            max_rows, total_heads, head_dim, group, workspace_slot
        )


def dcp_output_mori_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    group: GroupCoordinator,
    *,
    is_lse_base_on_e: bool,
    workspace_slot: int,
) -> torch.Tensor:
    workspace = get_dcp_output_mori_workspace(
        DCP_OUTPUT_MORI_MAX_ROWS,
        cp_attn_out.shape[1],
        cp_attn_out.shape[2],
        group,
        workspace_slot,
    )
    return workspace.merge(
        cp_attn_out,
        cp_attn_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )


def close_dcp_output_mori_workspaces() -> None:
    global _workspace_failed
    for workspace in list(_workspaces.values()):
        workspace.close()
    _workspaces.clear()
    _workspace_failed = False

