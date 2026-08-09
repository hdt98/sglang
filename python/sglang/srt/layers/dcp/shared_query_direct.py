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

"""Consumer-direct fused RoPE/FP8 Query transport over Mori SHMEM (AMD).

Each producer packs its local BF16 Query-head shard into owner-local Mori SHMEM
storage once.  Every consumer then directly loads all peer shards from its local
buffer (peers wrote via P2P) and fuses RoPE plus FP8 conversion into its
complete local Query.

AMD/HIP adaptations from PR #32851:
  - _rotate_pair_fp32: PTX FMA replaced with pure Triton arithmetic
  - _trap_if_nonzero: no-op (spin max_spins is the safeguard)
  - tl.float8e4nv -> tl.float8e4m3fn (standard E4M3 type name)
  - Flag kernels use pointer table + constexpr loop (separate peer addresses)
  - _consumer_direct_query_kernel reads from local buffer (unchanged strides)
  - Buffer includes world_size dimension for all peers' data
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
from sglang.srt.layers.dcp.shared_output import (
    MoriPeerBuffer,
    _trap_if_nonzero,
    create_mori_peer_buffer,
)

logger = logging.getLogger(__name__)
_HEADER_BYTES = 256
_MAX_FENCE_SPINS = 100_000_000
DCP_QUERY_DIRECT_MORI_MAX_ROWS = 512
_logged_rows: set[int] = set()


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _rotate_pair_fp32(first, second, cos, sin):
    """FP32 RoPE rotation (pure Triton, replaces PTX inline ASM).

    Matches the PTX contraction order:
      rotated_first  = cos * first - sin * second
      rotated_second = first * sin + second * cos
    """
    rotated_first = cos * first - sin * second
    rotated_second = first * sin + second * cos
    return rotated_first, rotated_second


@triton.jit
def _wait_writable_kernel(
    local_flags,
    peer_flag_ptrs,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    """Wait for all peers to ack the previous round (P2P flag reads).

    Reads local write_seq, then for each peer reads their ack_seq via pointer
    table.  Spins until all peers' ack_seq >= local write_seq.
    """
    write_seq = tl.atomic_add(local_flags, 0, sem="acquire", scope="sys")
    for peer in range(world_size):
        if peer == my_rank:
            continue
        raw_ptr = tl.load(peer_flag_ptrs + peer)
        flag_ptr = raw_ptr.to(tl.pointer_type(tl.int32))
        ack_seq = tl.atomic_add(flag_ptr + 1, 0, sem="acquire", scope="sys")
        spins = 0
        while (ack_seq < write_seq) & (spins < max_spins):
            ack_seq = tl.atomic_add(flag_ptr + 1, 0, sem="acquire", scope="sys")
            spins += 1


@triton.jit
def _publish_and_wait_kernel(
    local_flags,
    peer_flag_ptrs,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    """Increment local write_seq and wait for all peers to see it.

    Writes local write_seq (release), then for each peer reads their write_seq
    via pointer table.  Spins until all peers' write_seq >= epoch.
    """
    epoch = tl.atomic_add(local_flags, 1, sem="release", scope="sys") + 1
    for peer in range(world_size):
        if peer == my_rank:
            continue
        raw_ptr = tl.load(peer_flag_ptrs + peer)
        flag_ptr = raw_ptr.to(tl.pointer_type(tl.int32))
        write_seq = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="sys")
        spins = 0
        while (write_seq < epoch) & (spins < max_spins):
            write_seq = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="sys")
            spins += 1


@triton.jit
def _ack_kernel(local_flags):
    """Increment local ack_seq (local write, peers read via P2P)."""
    tl.atomic_add(local_flags + 1, 1, sem="release", scope="sys")


@triton.jit
def _pack_local_query_kernel(
    q_nope,
    q_rope,
    local_query,
    q_nope_row_stride,
    q_nope_head_stride,
    q_rope_row_stride,
    q_rope_head_stride,
    local_row_stride,
    local_head_stride,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    """Pack local BF16 Query (nope + rope) into the local buffer slot."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, block_dim)
    nope_mask = dim < nope_dim
    rope_mask = dim < rope_dim
    local_base = local_query + row * local_row_stride + head * local_head_stride
    tl.store(
        local_base + dim,
        tl.load(
            q_nope + row * q_nope_row_stride + head * q_nope_head_stride + dim,
            mask=nope_mask,
        ),
        mask=nope_mask,
    )
    tl.store(
        local_base + nope_dim + dim,
        tl.load(
            q_rope + row * q_rope_row_stride + head * q_rope_head_stride + dim,
            mask=rope_mask,
        ),
        mask=rope_mask,
    )


@triton.jit
def _consumer_direct_query_kernel(
    local_query,
    k_nope,
    k_rope,
    positions,
    cos_sin_cache,
    output,
    k_nope_out,
    k_rope_out,
    owner_stride,
    row_stride,
    head_stride,
    k_nope_row_stride,
    k_rope_row_stride,
    cache_row_stride,
    output_row_stride,
    output_head_stride,
    k_nope_out_row_stride,
    k_rope_out_row_stride,
    local_heads: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    nope_block: tl.constexpr,
    half_rope_block: tl.constexpr,
    is_neox: tl.constexpr,
):
    """Read all peer Query shards from local buffer, fuse RoPE + FP8 conversion.

    Reads from local_query (local memory where peers wrote via P2P).
    Unchanged from PR #32851 except tl.float8e4nv -> tl.float8e4m3fn.
    """
    component = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    global_head = tl.program_id(2).to(tl.int64)
    owner = global_head // local_heads
    local_head = global_head % local_heads
    source = (
        local_query
        + owner * owner_stride
        + row * row_stride
        + local_head * head_stride
    )
    target = output + row * output_row_stride + global_head * output_head_stride

    if component == 0:
        dim = tl.arange(0, nope_block)
        mask = dim < nope_dim
        value = tl.load(source + dim, mask=mask).to(tl.float32)
        tl.store(target + dim, value.to(tl.float8e4m3fn), mask=mask)
        k_mask = mask & (global_head == 0)
        k_value = tl.load(
            k_nope + row * k_nope_row_stride + dim,
            mask=k_mask,
        ).to(tl.float32)
        tl.store(
            k_nope_out + row * k_nope_out_row_stride + dim,
            k_value.to(tl.float8e4m3fn),
            mask=k_mask,
        )
        return

    position = tl.load(positions + row)
    half_dim = rope_dim // 2
    offset = tl.arange(0, half_rope_block)
    mask = offset < half_dim
    cos = tl.load(
        cos_sin_cache + position * cache_row_stride + offset,
        mask=mask,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache + position * cache_row_stride + half_dim + offset,
        mask=mask,
    ).to(tl.float32)
    if is_neox:
        first_offset = offset
        second_offset = half_dim + offset
    else:
        first_offset = offset * 2
        second_offset = offset * 2 + 1
    first = tl.load(source + nope_dim + first_offset, mask=mask).to(tl.float32)
    second = tl.load(source + nope_dim + second_offset, mask=mask).to(tl.float32)
    rotated_first, rotated_second = _rotate_pair_fp32(first, second, cos, sin)
    tl.store(
        target + nope_dim + first_offset,
        rotated_first.to(tl.float8e4m3fn),
        mask=mask,
    )
    tl.store(
        target + nope_dim + second_offset,
        rotated_second.to(tl.float8e4m3fn),
        mask=mask,
    )
    k_mask = mask & (global_head == 0)
    k_first = tl.load(
        k_rope + row * k_rope_row_stride + first_offset,
        mask=k_mask,
    ).to(tl.float32)
    k_second = tl.load(
        k_rope + row * k_rope_row_stride + second_offset,
        mask=k_mask,
    ).to(tl.float32)
    k_rotated_first, k_rotated_second = _rotate_pair_fp32(
        k_first,
        k_second,
        cos,
        sin,
    )
    tl.store(
        k_rope_out + row * k_rope_out_row_stride + first_offset,
        k_rotated_first.to(tl.float8e4m3fn),
        mask=k_mask,
    )
    tl.store(
        k_rope_out + row * k_rope_out_row_stride + second_offset,
        k_rotated_second.to(tl.float8e4m3fn),
        mask=k_mask,
    )


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

@dataclass
class DcpQueryDirectMoriWorkspace:
    rank: int
    world_size: int
    max_rows: int
    local_heads: int
    nope_dim: int
    rope_dim: int
    group: GroupCoordinator
    device: torch.device
    allocation: MoriPeerBuffer
    local_query: torch.Tensor
    local_flags: torch.Tensor
    peer_flag_ptrs: torch.Tensor
    query_output: torch.Tensor
    k_nope_output: torch.Tensor
    k_rope_output: torch.Tensor

    @property
    def total_heads(self) -> int:
        return self.local_heads * self.world_size

    @property
    def query_dim(self) -> int:
        return self.nope_dim + self.rope_dim

    @property
    def physical_bytes_per_rank(self) -> int:
        return self.allocation.bytes_per_rank

    def _validate_live(self) -> None:
        if self.allocation.closed:
            raise RuntimeError("consumer-direct Query Mori workspace is closed")
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError(
                "consumer-direct Query Mori current device changed after "
                f"initialization: workspace={self.device}, "
                f"current=cuda:{torch.cuda.current_device()}"
            )

    def quantize_remote(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        *,
        is_neox: bool,
        pipelined: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_live()
        rows = q_nope.shape[0]
        if rows <= 0 or rows > self.max_rows:
            raise RuntimeError(
                f"consumer-direct Query supports 1..{self.max_rows} rows, got {rows}"
            )
        if q_nope.shape != (rows, self.local_heads, self.nope_dim):
            raise RuntimeError(f"unexpected q_nope shape {tuple(q_nope.shape)}")
        if q_rope.shape != (rows, self.local_heads, self.rope_dim):
            raise RuntimeError(f"unexpected q_rope shape {tuple(q_rope.shape)}")
        if q_nope.dtype != torch.bfloat16 or q_rope.dtype != torch.bfloat16:
            raise RuntimeError("consumer-direct Query requires BF16 inputs")
        if k_nope.dtype != torch.bfloat16 or k_rope.dtype != torch.bfloat16:
            raise RuntimeError("consumer-direct Query requires BF16 K inputs")
        if k_nope.shape != (rows, self.nope_dim) or k_rope.shape != (
            rows,
            self.rope_dim,
        ):
            raise RuntimeError(
                "consumer-direct Query received unexpected K shapes: "
                f"{tuple(k_nope.shape)}, {tuple(k_rope.shape)}"
            )
        inputs = (q_nope, q_rope, k_nope, k_rope, positions, cos_sin_cache)
        if any(t.device != self.device for t in inputs):
            raise RuntimeError(
                "consumer-direct Query inputs must be on the workspace device "
                f"{self.device}, got {[str(t.device) for t in inputs]}"
            )
        if positions.ndim != 1 or positions.shape[0] != rows:
            raise RuntimeError(
                "consumer-direct Query positions must have one entry per row, "
                f"got shape={tuple(positions.shape)}, rows={rows}"
            )
        if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != self.rope_dim:
            raise RuntimeError(
                "consumer-direct Query cos/sin cache shape mismatch: "
                f"expected [positions, {self.rope_dim}], "
                f"got {tuple(cos_sin_cache.shape)}"
            )
        if any(t.stride(-1) != 1 for t in inputs):
            raise RuntimeError(
                "consumer-direct Query inputs must be contiguous in their last "
                f"dimension, got strides={[t.stride() for t in inputs]}"
            )

        if rows not in _logged_rows:
            _logged_rows.add(rows)
            logger.debug(
                "Executing consumer-direct Mori SHMEM DCP Query for rows=%d", rows
            )
        output = self.query_output[:rows]
        k_nope_out = self.k_nope_output[:rows]
        k_rope_out = self.k_rope_output[:rows]
        if not pipelined:
            with record_function("dcp.query_direct.wait_reuse"):
                _wait_writable_kernel[(1,)](
                    self.local_flags,
                    self.peer_flag_ptrs,
                    my_rank=self.rank,
                    world_size=self.world_size,
                    max_spins=_MAX_FENCE_SPINS,
                )
        with record_function("dcp.query_direct.pack_local_bf16"):
            _pack_local_query_kernel[(rows, self.local_heads)](
                q_nope,
                q_rope,
                self.local_query[self.rank],
                q_nope.stride(0),
                q_nope.stride(1),
                q_rope.stride(0),
                q_rope.stride(1),
                self.local_query[self.rank].stride(0),
                self.local_query[self.rank].stride(1),
                nope_dim=self.nope_dim,
                rope_dim=self.rope_dim,
                block_dim=triton.next_power_of_2(max(self.nope_dim, self.rope_dim)),
                num_warps=4,
            )
        with record_function("dcp.query_direct.publish_acquire"):
            _publish_and_wait_kernel[(1,)](
                self.local_flags,
                self.peer_flag_ptrs,
                my_rank=self.rank,
                world_size=self.world_size,
                max_spins=_MAX_FENCE_SPINS,
            )
        with record_function("dcp.query_direct.remote_rope_fp8"):
            _consumer_direct_query_kernel[(2, rows, self.total_heads)](
                self.local_query,
                k_nope,
                k_rope,
                positions,
                cos_sin_cache,
                output,
                k_nope_out,
                k_rope_out,
                self.local_query.stride(0),
                self.local_query.stride(1),
                self.local_query.stride(2),
                k_nope.stride(0),
                k_rope.stride(0),
                cos_sin_cache.stride(0),
                output.stride(0),
                output.stride(1),
                k_nope_out.stride(0),
                k_rope_out.stride(0),
                nope_dim=self.nope_dim,
                rope_dim=self.rope_dim,
                local_heads=self.local_heads,
                nope_block=triton.next_power_of_2(self.nope_dim),
                half_rope_block=triton.next_power_of_2(self.rope_dim // 2),
                is_neox=is_neox,
                num_warps=4,
            )
        if not pipelined:
            with record_function("dcp.query_direct.ack"):
                _ack_kernel[(1,)](
                    self.local_flags,
                )
        return output, k_nope_out, k_rope_out

    def close(self) -> None:
        if self.allocation.closed:
            return
        torch.cuda.synchronize()
        dist.barrier(group=self.group.cpu_group)
        del self.peer_flag_ptrs
        del self.local_flags
        del self.local_query
        del self.query_output
        del self.k_nope_output
        del self.k_rope_output
        self.allocation.close()


def create_dcp_query_direct_mori_workspace(
    max_rows: int,
    local_heads: int,
    nope_dim: int,
    rope_dim: int,
    group: GroupCoordinator,
) -> DcpQueryDirectMoriWorkspace:
    if group.world_size <= 1:
        raise RuntimeError("consumer-direct Query Mori requires dcp_size > 1")
    if min(max_rows, local_heads, nope_dim, rope_dim) <= 0:
        raise ValueError(
            "consumer-direct Query Mori dimensions must be positive: "
            f"max_rows={max_rows}, local_heads={local_heads}, "
            f"nope_dim={nope_dim}, rope_dim={rope_dim}"
        )
    world_size = group.world_size
    query_dim = nope_dim + rope_dim
    # Buffer holds all peers' data: (world_size, max_rows, local_heads, query_dim)
    payload_bytes = (
        world_size * max_rows * local_heads * query_dim * torch.bfloat16.itemsize
    )
    requested_bytes = _HEADER_BYTES + payload_bytes
    allocation = create_mori_peer_buffer(
        requested_bytes,
        group=group,
        device=group.device,
    )
    local_flags = allocation.local_view[: 2 * torch.int32.itemsize].view(torch.int32)
    local_query = (
        allocation.local_view[_HEADER_BYTES : _HEADER_BYTES + payload_bytes]
        .view(torch.bfloat16)
        .view(world_size, max_rows, local_heads, query_dim)
    )
    peer_flag_ptrs = allocation.make_typed_ptr_table(0, torch.int32)
    return DcpQueryDirectMoriWorkspace(
        rank=group.rank_in_group,
        world_size=world_size,
        max_rows=max_rows,
        local_heads=local_heads,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        group=group,
        device=group.device,
        allocation=allocation,
        local_query=local_query,
        local_flags=local_flags,
        peer_flag_ptrs=peer_flag_ptrs,
        query_output=torch.empty(
            max_rows,
            local_heads * world_size,
            query_dim,
            dtype=torch.float8_e4m3fn,
            device=group.device,
        ),
        k_nope_output=torch.empty(
            max_rows,
            nope_dim,
            dtype=torch.float8_e4m3fn,
            device=group.device,
        ),
        k_rope_output=torch.empty(
            max_rows,
            rope_dim,
            dtype=torch.float8_e4m3fn,
            device=group.device,
        ),
    )


_workspaces: dict[int, DcpQueryDirectMoriWorkspace] = {}
_workspace_failed = False


def get_dcp_query_direct_mori_workspace(
    max_rows: int,
    local_heads: int,
    nope_dim: int,
    rope_dim: int,
    group: GroupCoordinator,
    *,
    workspace_slot: int = 0,
) -> DcpQueryDirectMoriWorkspace:
    global _workspace_failed
    if workspace_slot < 0:
        raise ValueError(f"workspace_slot must be non-negative, got {workspace_slot}")
    if _workspace_failed:
        raise RuntimeError("consumer-direct Query Mori workspace is unavailable")
    workspace = _workspaces.get(workspace_slot)
    if workspace is not None:
        actual = (
            workspace.max_rows,
            workspace.local_heads,
            workspace.nope_dim,
            workspace.rope_dim,
            workspace.world_size,
            workspace.rank,
            workspace.device,
        )
        requested = (
            max_rows,
            local_heads,
            nope_dim,
            rope_dim,
            group.world_size,
            group.rank_in_group,
            group.device,
        )
        if actual != requested or workspace.group is not group:
            raise RuntimeError(
                "consumer-direct Query Mori workspace identity changed: "
                f"actual={actual}, requested={requested}, "
                f"same_group={workspace.group is group}, "
                f"workspace_slot={workspace_slot}"
            )
        return workspace
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "consumer-direct Query Mori workspace was not initialized before "
            "CUDA graph capture"
        )
    try:
        workspace = create_dcp_query_direct_mori_workspace(
            max_rows, local_heads, nope_dim, rope_dim, group
        )
        _workspaces[workspace_slot] = workspace
    except Exception as error:
        _workspace_failed = True
        raise RuntimeError(
            "consumer-direct Query Mori workspace initialization failed; "
            "refusing to fall back after selecting the direct route"
        ) from error
    logger.info(
        "Initialized consumer-direct DCP Query Mori workspace: slot=%d, max_rows=%d, "
        "local_heads=%d, total_heads=%d, query_dim=%d, "
        "physical_bytes_per_rank=%d",
        workspace_slot,
        workspace.max_rows,
        workspace.local_heads,
        workspace.total_heads,
        workspace.query_dim,
        workspace.physical_bytes_per_rank,
    )
    return workspace


def init_dcp_query_direct_mori_workspace(
    group: GroupCoordinator,
    local_heads: int,
    nope_dim: int,
    rope_dim: int,
    *,
    max_rows: int = DCP_QUERY_DIRECT_MORI_MAX_ROWS,
    workspace_slots: int = 2,
) -> None:
    if workspace_slots <= 0:
        raise ValueError(f"workspace_slots must be positive, got {workspace_slots}")
    for workspace_slot in range(workspace_slots):
        get_dcp_query_direct_mori_workspace(
            max_rows,
            local_heads,
            nope_dim,
            rope_dim,
            group,
            workspace_slot=workspace_slot,
        )


def close_dcp_query_direct_mori_workspace() -> None:
    global _workspace_failed
    for workspace in _workspaces.values():
        workspace.close()
    _workspaces.clear()
    _workspace_failed = False

