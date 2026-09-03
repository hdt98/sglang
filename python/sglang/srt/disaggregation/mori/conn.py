from __future__ import annotations

import dataclasses
import logging
import os
import struct
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import msgspec
import numpy as np
import numpy.typing as npt
import zmq
from mori.cpp import TransferStatus
from mori.io import (
    BackendType,
    EngineDesc,
    IOEngine,
    IOEngineConfig,
    MemoryDesc,
    MemoryLocationType,
    PollCqMode,
    RdmaBackendConfig,
    StatusCode,
    XgmiBackendConfig,
)

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
    KVTransferError,
)
from sglang.srt.disaggregation.common.utils import (
    AuxDataCodec,
    FastQueue,
    TransferKVChunk,
    group_concurrent_contiguous,
    pack_int_lists,
    unpack_int_lists,
)
from sglang.srt.disaggregation.common.staging_buffer import (
    StagingAllocator,
    staging_grid_tokens,
)
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingContext,
    DecodeStagingHandler,
    PrefillStagingContext,
    StagingTransferInfo,
    STAGING_WATERMARK_WAIT_S,
    handle_staging_rsp,
    handle_watermark_msg,
    is_watermark_ready,
    prefetch_staging_reqs,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    build_dsa_tail_transfer_blocks,
    pair_mamba_state_indices,
    slice_dsa_tail_dst_ptrs_for_pp,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_schedule
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)
MORI_GUARD = b"MoriMsgGuard"
_TAG_ABORT = b"ABORT"
_MORI_XGMI_ONLY_FALLBACK_PORT = 1
# Temporary single-run fault isolation; keep empty in production recipes.
_MORI_DIAGNOSTIC_SKIP_STATE_TYPES = frozenset()
_MORI_DIAGNOSTIC_SKIP_KV = False


def _create_configured_xgmi_backend(engine: IOEngine) -> bool:
    """Create XGMI before RDMA fallback when its pool sizes are overridden."""
    raw_num_streams = os.environ.get("SGLANG_MORI_XGMI_NUM_STREAMS")
    raw_num_events = os.environ.get("SGLANG_MORI_XGMI_NUM_EVENTS")
    if raw_num_streams is None and raw_num_events is None:
        return False

    num_streams = int(raw_num_streams or "64")
    num_events = int(raw_num_events or "64")
    if num_streams <= 0 or num_events <= 0:
        raise ValueError(
            "SGLANG_MORI_XGMI_NUM_STREAMS and "
            "SGLANG_MORI_XGMI_NUM_EVENTS must be positive integers"
        )

    engine.create_backend(
        BackendType.XGMI,
        XgmiBackendConfig(num_streams=num_streams, num_events=num_events),
    )
    logger.info(
        "Created configured Mori XGMI backend (streams=%s, events=%s)",
        num_streams,
        num_events,
    )
    return True


def _ensure_xgmi_fallback_kernels(engine: IOEngine, actual_port: int) -> bool:
    """Load Mori's XGMI kernels after its RDMA backend falls back to XGMI."""
    if os.environ.get("SGLANG_MORI_ENABLE_XGMI_FALLBACK_KERNELS", "1") != "1":
        logger.info(
            "Mori XGMI fallback kernels are disabled; fragmented transfers "
            "will use peer copies"
        )
        return False

    # Mori reserves port 1 as kXgmiOnlyFallbackPlaceholderPort. Requesting the
    # already-created XGMI backend is idempotent in Mori and makes its Python
    # wrapper load the scatter/gather module as it does for explicit XGMI.
    if actual_port != _MORI_XGMI_ONLY_FALLBACK_PORT:
        return False

    xgmi_backend = getattr(BackendType, "XGMI", None)
    if xgmi_backend is None:
        logger.warning(
            "Mori XGMI-only fallback is active, but this Mori version does not "
            "expose BackendType.XGMI; fragmented transfers will use peer copies"
        )
        return False

    try:
        engine.create_backend(xgmi_backend)
    except Exception:
        logger.warning(
            "Failed to load Mori XGMI fallback kernels; fragmented transfers "
            "will use peer copies",
            exc_info=True,
        )
        return False

    logger.info("Loaded Mori XGMI fallback kernels")
    return True


def _normalize_state_indices_per_component(
    state_indices: Optional[List],
) -> Optional[List[Optional[npt.NDArray[np.int32]]]]:
    if state_indices is None:
        return None
    out: List[Optional[npt.NDArray[np.int32]]] = []
    for entry in state_indices:
        if entry is None:
            out.append(None)
        else:
            out.append(np.asarray(entry, dtype=np.int32).ravel())
    return out


def _pack_state_indices(
    state_indices: Optional[List[Optional[npt.NDArray[np.int32]]]],
) -> bytes:
    if not state_indices:
        return b""
    lists = [(arr.tolist() if arr is not None else []) for arr in state_indices]
    return pack_int_lists(lists, "i")


def _unpack_state_indices(buf: bytes) -> List[npt.NDArray[np.int32]]:
    if not buf:
        return []
    return [np.asarray(lst, dtype=np.int32) for lst in unpack_int_lists(buf, "i")]


def _pack_mem_desc_list(mems: List[MemoryDesc]) -> bytes:
    if not mems:
        return b""
    packed_descs = [mem.pack() for mem in mems]
    return msgspec.msgpack.encode(packed_descs)


def _unpack_mem_desc_list(blob: bytes) -> List[MemoryDesc]:
    if not blob:
        return []
    desc_blobs = msgspec.msgpack.decode(blob)
    return [MemoryDesc.unpack(b) for b in desc_blobs]


def _pack_mem_desc_lists(mems_per_comp: List[List[MemoryDesc]]) -> bytes:
    if not mems_per_comp:
        return b""
    return msgspec.msgpack.encode(
        [[mem.pack() for mem in comp] for comp in mems_per_comp]
    )


def _unpack_mem_desc_lists(blob: bytes) -> List[List[MemoryDesc]]:
    if not blob:
        return []
    nested = msgspec.msgpack.decode(blob)
    return [[MemoryDesc.unpack(b) for b in comp] for comp in nested]


@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    engine_key: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    dst_state_indices: List[npt.NDArray[np.int32]]
    required_dst_info_num: int
    is_dummy: bool
    staging: Optional[StagingTransferInfo] = None
    # Number of tokens decode already holds in its radix cache; prefill should
    # only send pages beyond this prefix. None means the receiver did not
    # populate this field (older receiver or radix-cache feature off) -> treat
    # as 0 (no prefix hit, full send) for backward compatibility.
    decode_prefix_len: Optional[int] = None

    @classmethod
    def from_zmq(cls, payload: List[bytes]) -> TransferInfo:
        room = int(payload[0].decode("ascii"))
        endpoint = payload[1].decode("ascii")
        dst_port = int(payload[2].decode("ascii"))
        engine_key = payload[3].decode("ascii")

        if payload[4]:
            dst_kv_indices = np.frombuffer(payload[4], dtype=np.int32)
        else:
            dst_kv_indices = np.array([], dtype=np.int32)

        if payload[5]:
            dst_aux_index = int(payload[5].decode("ascii"))
        else:
            dst_aux_index = -1

        if len(payload) > 6 and payload[6]:
            dst_state_indices = _unpack_state_indices(payload[6])
        else:
            dst_state_indices = []

        required_dst_info_num = (
            int(payload[7].decode("ascii")) if len(payload) > 7 else 1
        )

        if len(payload) > 8 and payload[8]:
            decode_prefix_len: Optional[int] = int(payload[8].decode("ascii"))
        else:
            decode_prefix_len = None

        # A transfer is "dummy" only when the receiver does not need any
        # kv/aux/state delivered. When decode_prefix_len > 0 and the delta is
        # exactly zero (full prefix hit), dst_kv_indices is empty but aux is
        # still needed -> not dummy.
        is_dummy = (
            dst_kv_indices.size == 0 and dst_aux_index < 0 and not decode_prefix_len
        )
        return cls(
            room=room,
            endpoint=endpoint,
            dst_port=dst_port,
            engine_key=engine_key,
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_state_indices=dst_state_indices,
            required_dst_info_num=required_dst_info_num,
            is_dummy=is_dummy,
            decode_prefix_len=decode_prefix_len,
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    endpoint: str
    dst_port: int
    engine_desc: EngineDesc
    dst_kv_mem_descs: List[MemoryDesc]
    dst_aux_mem_descs: List[MemoryDesc]
    dst_state_mem_descs: List[List[MemoryDesc]]
    gpu_id: int
    decode_tp_size: int
    decode_tp_rank: int
    dst_kv_item_len: int
    dst_kv_item_lens: List[int]
    dst_state_item_lens: List[List[int]]
    dst_state_dim_per_tensor: List[List[int]]
    dst_state_slot_strides: List[List[int]] = dataclasses.field(default_factory=list)
    dst_state_mem_desc_offsets: List[List[int]] = dataclasses.field(
        default_factory=list
    )
    staging_mem_desc: Optional[MemoryDesc] = None
    dst_num_target_kv_entries: int = 0

    @property
    def engine_key(self) -> str:
        return self.engine_desc.key

    @classmethod
    def from_zmq(cls, payload: List[bytes]) -> KVArgsRegisterInfo:
        endpoint = payload[1].decode("ascii")
        dst_port = int(payload[2].decode("ascii"))
        engine_desc = EngineDesc.unpack(payload[3])
        dst_kv_mem_descs = _unpack_mem_desc_list(payload[4])
        dst_aux_mem_descs = _unpack_mem_desc_list(payload[5])
        dst_state_mem_descs = _unpack_mem_desc_lists(payload[6])
        gpu_id = int(payload[7].decode("ascii"))
        decode_tp_size = int(payload[8].decode("ascii"))
        decode_tp_rank = int(payload[9].decode("ascii"))
        dst_kv_item_len = int(payload[10].decode("ascii"))
        dst_state_item_lens = (
            unpack_int_lists(payload[11], "I")
            if len(payload) > 11 and payload[11]
            else []
        )
        dst_state_dim_per_tensor = (
            unpack_int_lists(payload[12], "I")
            if len(payload) > 12 and payload[12]
            else []
        )
        dst_kv_item_lens = (
            list(struct.unpack(f"{len(payload[13]) // 8}Q", payload[13]))
            if len(payload) > 13 and payload[13]
            else [dst_kv_item_len] * len(dst_kv_mem_descs)
        )
        if len(dst_kv_item_lens) != len(dst_kv_mem_descs):
            raise ValueError(
                "dst_kv_item_lens length mismatch: "
                f"got {len(dst_kv_item_lens)}, expected {len(dst_kv_mem_descs)}"
            )
        dst_state_slot_strides = (
            unpack_int_lists(payload[14], "Q")
            if len(payload) > 14 and payload[14]
            else [list(component) for component in dst_state_item_lens]
        )
        dst_state_mem_desc_offsets = (
            unpack_int_lists(payload[15], "Q")
            if len(payload) > 15 and payload[15]
            else [[0] * len(component) for component in dst_state_mem_descs]
        )
        staging_mem_descs = (
            _unpack_mem_desc_list(payload[16])
            if len(payload) > 16 and payload[16]
            else []
        )
        if len(staging_mem_descs) > 1:
            raise ValueError(
                "Mori staging descriptor count mismatch: expected at most one"
            )
        dst_num_target_kv_entries = (
            int(payload[17].decode("ascii"))
            if len(payload) > 17 and payload[17]
            else 0
        )
        return cls(
            endpoint=endpoint,
            dst_port=dst_port,
            engine_desc=engine_desc,
            dst_kv_mem_descs=dst_kv_mem_descs,
            dst_aux_mem_descs=dst_aux_mem_descs,
            dst_state_mem_descs=dst_state_mem_descs,
            gpu_id=gpu_id,
            decode_tp_size=decode_tp_size,
            decode_tp_rank=decode_tp_rank,
            dst_kv_item_len=dst_kv_item_len,
            dst_kv_item_lens=dst_kv_item_lens,
            dst_state_item_lens=dst_state_item_lens,
            dst_state_slot_strides=dst_state_slot_strides,
            dst_state_mem_desc_offsets=dst_state_mem_desc_offsets,
            dst_state_dim_per_tensor=dst_state_dim_per_tensor,
            staging_mem_desc=staging_mem_descs[0] if staging_mem_descs else None,
            dst_num_target_kv_entries=dst_num_target_kv_entries,
        )


@dataclasses.dataclass
class TPSliceConfig:
    page_size: int
    src_item_len: int
    dst_item_len: int
    bytes_per_token_src: int
    bytes_per_token_dst: int
    src_head_slice_offset: int
    dst_head_slice_offset: int
    heads_bytes_per_token_to_send: int


@dataclasses.dataclass(frozen=True)
class GroupedIndexPlan:
    src_starts: List[int]
    dst_starts: List[int]
    counts: List[int]

    @classmethod
    def from_groups(
        cls, src_groups: List[List[int]], dst_groups: List[List[int]]
    ) -> GroupedIndexPlan:
        if len(src_groups) != len(dst_groups):
            raise ValueError("Source and destination groups must have the same length")
        return cls(
            src_starts=[int(group[0]) for group in src_groups],
            dst_starts=[int(group[0]) for group in dst_groups],
            counts=[len(group) for group in src_groups],
        )

    def materialize(self, item_len: int) -> BatchTransferPlan:
        return BatchTransferPlan(
            local_offsets=[start * item_len for start in self.src_starts],
            remote_offsets=[start * item_len for start in self.dst_starts],
            sizes=[count * item_len for count in self.counts],
        )


@dataclasses.dataclass(frozen=True)
class BatchTransferPlan:
    local_offsets: List[int]
    remote_offsets: List[int]
    sizes: List[int]

    def empty(self) -> bool:
        return not self.sizes


@dataclasses.dataclass(frozen=True)
class TransferTarget:
    info: TransferInfo
    peer_info: KVArgsRegisterInfo


def _map_views_to_registered_regions(
    view_ptrs: List[int],
    view_lens: List[int],
    registration_ptrs: List[int],
    registration_lens: List[int],
    registration_descs: List[MemoryDesc],
) -> Tuple[List[MemoryDesc], List[int]]:
    """Map logical tensor views onto their registered backing allocations."""
    if len(view_ptrs) != len(view_lens):
        raise ValueError(
            "Mori state view metadata mismatch: "
            f"ptrs={len(view_ptrs)}, lens={len(view_lens)}"
        )
    if not (
        len(registration_ptrs)
        == len(registration_lens)
        == len(registration_descs)
    ):
        raise ValueError(
            "Mori state registration metadata mismatch: "
            f"ptrs={len(registration_ptrs)}, lens={len(registration_lens)}, "
            f"descs={len(registration_descs)}"
        )

    mapped_descs: List[MemoryDesc] = []
    mapped_offsets: List[int] = []
    for view_ptr, view_len in zip(view_ptrs, view_lens):
        view_end = view_ptr + view_len
        for registration_ptr, registration_len, registration_desc in zip(
            registration_ptrs, registration_lens, registration_descs
        ):
            registration_end = registration_ptr + registration_len
            if registration_ptr <= view_ptr and view_end <= registration_end:
                mapped_descs.append(registration_desc)
                mapped_offsets.append(view_ptr - registration_ptr)
                break
        else:
            raise ValueError(
                "Mori state tensor view is outside registered backing memory: "
                f"view=[{view_ptr}, {view_end}), "
                f"regions={list(zip(registration_ptrs, registration_lens))}"
            )
    return mapped_descs, mapped_offsets


class MoriKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.engine = self._init_engine()
        self.engine_desc = self.engine.get_engine_desc()
        self.kv_mem_descs: List[MemoryDesc] = []
        self.aux_mem_descs: List[MemoryDesc] = []
        self.state_mem_descs: List[List[MemoryDesc]] = []
        self.state_mem_desc_offsets: List[List[int]] = []
        self._diagnostic_log_ranges = (
            os.environ.get("SGLANG_MORI_DIAGNOSTIC_LOG_RANGES", "0") == "1"
        )
        self._diagnostic_plan_limit = int(
            os.environ.get("SGLANG_MORI_DIAGNOSTIC_PLAN_LIMIT", "4096")
        )
        self._diagnostic_plan_count = 0
        self._synchronous_chunk_transfer = (
            os.environ.get("SGLANG_MORI_SYNCHRONOUS_CHUNK_TRANSFER", "0") == "1"
        )
        self._release_xgmi_mappings_after_chunk = (
            os.environ.get(
                "SGLANG_MORI_RELEASE_XGMI_MAPPINGS_AFTER_CHUNK", "0"
            )
            == "1"
        )
        if (
            self._release_xgmi_mappings_after_chunk
            and not self._synchronous_chunk_transfer
        ):
            raise ValueError(
                "SGLANG_MORI_RELEASE_XGMI_MAPPINGS_AFTER_CHUNK requires "
                "SGLANG_MORI_SYNCHRONOUS_CHUNK_TRANSFER=1"
            )
        if self._synchronous_chunk_transfer:
            logger.warning(
                "Mori synchronous chunk transfer is enabled; prefill compute "
                "will not overlap KV handoff"
            )
        if self._release_xgmi_mappings_after_chunk:
            logger.warning(
                "Mori will close remote XGMI IPC mappings after each completed "
                "chunk; this is a gfx950 fault-isolation control"
            )
        self.transfer_lock = threading.Lock()
        self._zmq_ctx = zmq.Context()
        self._socket_local = threading.local()
        self._send_aux_rdma = envs.SGLANG_MORI_SEND_AUX_RDMA.get()
        self.enable_staging = envs.SGLANG_MORI_STAGING_BUFFER.get()
        self.staging_mem_desc: Optional[MemoryDesc] = None
        self._wait_poll_ms = envs.SGLANG_MORI_WAIT_POLL_MS.get()
        self._transfer_timeout_ms = envs.SGLANG_MORI_TRANSFER_TIMEOUT_MS.get()
        self._register_local_buffers()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._num_shards = max(1, envs.SGLANG_MORI_TRANSFER_SHARDS.get())
            self._transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(self._num_shards)
            ]
            self._staging_ctx = PrefillStagingContext() if self.enable_staging else None
            self._staging_full_chunk_pages = (
                staging_grid_tokens(
                    get_schedule().chunked_prefill_size,
                    self.kv_args.page_size,
                )
                // self.kv_args.page_size
                if self.enable_staging
                else 0
            )
            self._room_status_notified: Dict[int, bool] = {}
            self._room_notify_lock = threading.Lock()
            for shard, queue in enumerate(self._transfer_queues):
                threading.Thread(
                    target=self._transfer_worker,
                    args=(queue,),
                    daemon=True,
                    name=(
                        f"mori-xfer-dp{self.system_dp_rank}-"
                        f"tp{self.attn_tp_rank}-s{shard}"
                    ),
                ).start()
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.room_to_bootstrap_addr: Dict[int, str] = {}
            self._staging_ctx = DecodeStagingContext() if self.enable_staging else None
            self._staging_handler = None
            if self.enable_staging:
                self._init_staging_allocator()
            self._start_decode_thread()
            self._start_heartbeat_checker_thread()

    def _num_target_kv_entries(self) -> int:
        num_target = getattr(self.kv_args, "num_target_kv_entries", 0)
        if num_target <= 0:
            num_target = len(self.kv_mem_descs)
        if num_target > len(self.kv_mem_descs):
            raise ValueError(
                "Mori staging target descriptor count exceeds local descriptors: "
                f"target={num_target}, local={len(self.kv_mem_descs)}"
            )
        return num_target

    def _init_staging_allocator(self) -> None:
        pool_size_mb = envs.SGLANG_MORI_STAGING_POOL_SIZE_MB.get()
        pool_size_bytes = pool_size_mb * 1024 * 1024
        device = f"cuda:{self.kv_args.gpu_id}"
        allocator = StagingAllocator(
            pool_size_bytes,
            device,
            self.kv_args.gpu_id,
        )
        self.staging_mem_desc = self.engine.register_memory(
            allocator.get_base_ptr(),
            allocator.get_total_size(),
            self.kv_args.gpu_id,
            MemoryLocationType.GPU,
        )
        self._staging_ctx.allocator = allocator

    def register_staging_room_bootstrap(self, room, bootstrap_infos, receiver) -> None:
        self._staging_ctx.room_bootstrap[room] = bootstrap_infos
        self._staging_ctx.room_receivers[room] = receiver

    def create_staging_handler(self, scheduler, tp_rank):
        return MoriDecodeStagingHandler.create(
            self,
            scheduler,
            tp_rank,
        )

    def _init_engine(self) -> IOEngine:
        if self.kv_args.ib_device:
            os.environ["MORI_RDMA_DEVICES"] = self.kv_args.ib_device

        self.local_ip = get_local_ip_auto()
        config = IOEngineConfig(host=self.local_ip, port=0)

        engine_key = (
            f"io-{self.disaggregation_mode.value}-"
            f"dp{self.system_dp_rank}-tp{self.attn_tp_rank}-"
            f"pid{os.getpid()}-{self.local_ip}-"
            f"{uuid.uuid4().hex[:8]}"
        )

        engine = IOEngine(engine_key, config)
        _create_configured_xgmi_backend(engine)
        poll_mode = PollCqMode.POLLING

        qp_per_transfer = envs.SGLANG_MORI_QP_PER_TRANSFER.get()
        post_batch_size = envs.SGLANG_MORI_POST_BATCH_SIZE.get()
        num_worker_threads = envs.SGLANG_MORI_NUM_WORKERS.get()

        rdma_cfg = RdmaBackendConfig(
            qp_per_transfer,
            post_batch_size,
            num_worker_threads,
            poll_mode,
            False,
        )
        engine.create_backend(BackendType.RDMA, rdma_cfg)
        actual_port = engine.get_engine_desc().port
        assert actual_port > 0, f"Failed to bind port for engine {engine_key}"
        _ensure_xgmi_fallback_kernels(engine, actual_port)
        logger.debug(
            "Initialized Mori IOEngine %s at %s:%s (qp_per_transfer=%s, workers=%s, poll_mode=%s)",
            engine_key,
            self.local_ip,
            actual_port,
            qp_per_transfer,
            num_worker_threads,
            poll_mode.name,
        )
        return engine

    def _register_local_buffers(self) -> None:
        for ptr, length in zip(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens):
            mem_desc = self.engine.register_memory(
                ptr,
                length,
                self.kv_args.gpu_id,
                MemoryLocationType.GPU,
            )
            self.kv_mem_descs.append(mem_desc)
        for ptr, length in zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens):
            desc = self.engine.register_memory(
                ptr,
                length,
                -1,
                MemoryLocationType.CPU,
            )
            self.aux_mem_descs.append(desc)
        state_data_lens = getattr(self.kv_args, "state_data_lens", [])
        registration_ptrs = getattr(
            self.kv_args, "state_registration_ptrs", self.kv_args.state_data_ptrs
        )
        registration_lens = getattr(
            self.kv_args, "state_registration_lens", state_data_lens
        )
        for component_idx, (component_ptrs, component_lens) in enumerate(
            zip(self.kv_args.state_data_ptrs, state_data_lens)
        ):
            component_registration_ptrs = registration_ptrs[component_idx]
            component_registration_lens = registration_lens[component_idx]
            registered_descs = [
                self.engine.register_memory(
                    ptr,
                    length,
                    self.kv_args.gpu_id,
                    MemoryLocationType.GPU,
                )
                for ptr, length in zip(
                    component_registration_ptrs, component_registration_lens
                )
            ]
            component_descs, component_offsets = _map_views_to_registered_regions(
                component_ptrs,
                component_lens,
                component_registration_ptrs,
                component_registration_lens,
                registered_descs,
            )
            self.state_mem_descs.append(component_descs)
            self.state_mem_desc_offsets.append(component_offsets)

        if self._diagnostic_log_ranges:
            self._log_registered_ranges()

    @staticmethod
    def _memory_desc_fields(desc: MemoryDesc) -> str:
        return ", ".join(
            f"{name}={getattr(desc, name, 'n/a')}"
            for name in (
                "id",
                "data",
                "size",
                "device_id",
                "deviceId",
                "device_bus_id",
                "deviceBusId",
                "ipc_offset",
                "ipcOffset",
            )
        )

    def _log_registered_ranges(self) -> None:
        for index, (ptr, length, desc) in enumerate(
            zip(
                self.kv_args.kv_data_ptrs,
                self.kv_args.kv_data_lens,
                self.kv_mem_descs,
            )
        ):
            logger.info(
                "Mori diagnostic KV registration: rank=%s index=%s "
                "ptr=%#x end=%#x length=%s desc=(%s)",
                self.attn_tp_rank,
                index,
                ptr,
                ptr + length,
                length,
                self._memory_desc_fields(desc),
            )
        for component, (ptrs, lens, descs, desc_offsets) in enumerate(
            zip(
                self.kv_args.state_data_ptrs,
                getattr(self.kv_args, "state_data_lens", []),
                self.state_mem_descs,
                self.state_mem_desc_offsets,
            )
        ):
            for index, (ptr, length, desc, desc_offset) in enumerate(
                zip(ptrs, lens, descs, desc_offsets)
            ):
                logger.info(
                    "Mori diagnostic state registration: rank=%s component=%s "
                    "index=%s ptr=%#x end=%#x length=%s desc_offset=%s desc=(%s)",
                    self.attn_tp_rank,
                    component,
                    index,
                    ptr,
                    ptr + length,
                    length,
                    desc_offset,
                    self._memory_desc_fields(desc),
                )

    def update_status(self, bootstrap_room: int, status: KVPoll):
        current = self.request_status.get(bootstrap_room)
        if current is None:
            # Room not yet created or already cleared.
            # Only allow initial creation: Bootstrapping (normal) or
            # WaitingForInput (dummy CP rank, see CommonKVSender.__init__).
            if status not in (KVPoll.Bootstrapping, KVPoll.WaitingForInput):
                return
        elif current == KVPoll.Failed and status != KVPoll.Failed:
            # Failed is terminal — never overwrite with non-Failed.
            return
        super().update_status(bootstrap_room, status)

    def _transfer_worker(self, queue: FastQueue) -> None:
        while True:
            kv_chunk = queue.get()
            try:
                self._process_transfer_chunk(kv_chunk, queue)
            except Exception as exc:
                failure_reason = f"transfer worker raised: {exc!r}"
                try:
                    logger.exception(
                        "Mori transfer worker failed for room %s",
                        kv_chunk.room,
                    )
                except Exception:
                    pass
                try:
                    self._conclude_room_failure(kv_chunk.room, failure_reason)
                except Exception:
                    try:
                        logger.exception(
                            "Mori transfer worker failover failed for room %s",
                            kv_chunk.room,
                        )
                    except Exception:
                        pass

    def _process_transfer_chunk(
        self, kv_chunk: TransferKVChunk, queue: Optional[FastQueue] = None
    ) -> None:
        room = kv_chunk.room
        if self._should_skip_transfer(room):
            return

        if kv_chunk.wait_event is not None:
            kv_chunk.wait_event.synchronize()

        if self._should_skip_transfer(room):
            return

        if self.enable_staging and not self._staging_room_ready(
            room, kv_chunk.index_slice
        ):
            with self._staging_ctx.watermark_cv:
                self._staging_ctx.watermark_cv.wait(
                    STAGING_WATERMARK_WAIT_S
                )
            if queue is not None:
                queue.put(kv_chunk)
            else:
                self._dispatch_transfer_chunk(
                    room % self._num_shards,
                    kv_chunk,
                )
            return

        statuses, target_infos = self._submit_kv_transfer(
            room,
            kv_chunk.prefill_kv_indices,
            kv_chunk.index_slice,
            kv_chunk.is_last_chunk,
            aux_index=kv_chunk.prefill_aux_index,
            state_indices=kv_chunk.state_indices,
        )

        if self._should_skip_transfer(room):
            return

        failure_reason = self._wait_transfer_completion(statuses)
        if failure_reason is None:
            self._release_completed_chunk_mappings()
        if self._should_skip_transfer(room):
            return
        if failure_reason is not None:
            self._conclude_room_failure(room, failure_reason)
            return

        if self.enable_staging and target_infos:
            self._send_chunk_ready(
                [info for info in target_infos if not info.is_dummy],
                room,
                kv_chunk.index_slice,
                len(kv_chunk.prefill_kv_indices),
            )

        if kv_chunk.is_last_chunk:
            self._notify_decode_for_room(
                room, KVPoll.Success, target_infos=target_infos
            )
            self.update_status(room, KVPoll.Success)

    def _should_skip_transfer(self, room: int) -> bool:
        if room not in self.request_status or self.check_status(room) == KVPoll.Failed:
            logger.debug(
                "Skipping chunk for room %s because it has already failed or been aborted",
                room,
            )
            return True
        return False

    def _staging_chunk_ready(
        self, info: TransferInfo, index_slice: slice
    ) -> Tuple[bool, int, int]:
        if info.staging is None:
            return False, 0, -1

        chunk_idx = (
            index_slice.start // self._staging_full_chunk_pages
            if self._staging_full_chunk_pages > 0
            else 0
        )
        if chunk_idx >= len(info.staging.offsets):
            return False, chunk_idx, -1

        offset = info.staging.offsets[chunk_idx]
        if offset == StagingAllocator.ALLOC_OVERSIZED:
            raise RuntimeError(
                f"Mori staging chunk {chunk_idx} is larger than the decode pool"
            )
        if offset < 0:
            return False, chunk_idx, offset

        alloc_round = info.staging.rounds[chunk_idx]
        alloc_end = info.staging.ends[chunk_idx]
        ready = is_watermark_ready(
            self._staging_ctx,
            info.engine_key,
            alloc_round,
            alloc_end,
        )
        return ready, chunk_idx, offset

    def _staging_room_ready(self, room: int, index_slice: slice) -> bool:
        with self.transfer_lock:
            transfer_infos = self.transfer_infos.get(room, {})
            for info in transfer_infos.values():
                if info.is_dummy:
                    continue
                ready, _, _ = self._staging_chunk_ready(info, index_slice)
                if not ready:
                    return False
        return True

    def _send_chunk_ready(
        self,
        target_infos: List[TransferInfo],
        room: int,
        index_slice: slice,
        num_pages: int,
    ) -> None:
        chunk_idx = (
            index_slice.start // self._staging_full_chunk_pages
            if self._staging_full_chunk_pages > 0
            else 0
        )
        writer_id = str(self._compute_prefill_unique_rank()).encode("ascii")
        for info in target_infos:
            payload = [
                b"CHUNK_READY",
                str(room).encode("ascii"),
                str(chunk_idx).encode("ascii"),
                str(index_slice.start).encode("ascii"),
                str(num_pages).encode("ascii"),
                info.engine_key.encode("ascii"),
                writer_id,
            ]
            try:
                na = NetworkAddress(info.endpoint, info.dst_port)
                socket = self._connect_threadsafe(na.to_tcp(), is_ipv6=na.is_ipv6)
                socket.send_multipart(payload)
            except Exception:
                logger.exception(
                    "Failed to send Mori CHUNK_READY for room=%s chunk=%s",
                    room,
                    chunk_idx,
                )

    def _prefetch_staging_reqs(self, room: int) -> None:
        if not self.enable_staging:
            return
        prefetch_staging_reqs(
            room,
            self.transfer_infos,
            {"page_size": self.kv_args.page_size},
            get_schedule().chunked_prefill_size,
            self._staging_ctx.prefetch_requested,
            self._staging_ctx.prefetch_sockets,
            requester_pp_rank=self.pp_rank,
        )

    def _release_completed_chunk_mappings(self) -> None:
        if not self._release_xgmi_mappings_after_chunk:
            return
        released = self.engine.release_xgmi_remote_mappings()
        logger.info(
            "Released %s Mori XGMI remote mapping(s) after completed chunk",
            released,
        )

    def _wait_transfer_completion(
        self, statuses: List[TransferStatus]
    ) -> Optional[str]:
        if not statuses:
            return None

        start = time.perf_counter()
        sla_ms = self._transfer_timeout_ms

        while True:
            rc = self.engine.wait_all(statuses, timeout_ms=self._wait_poll_ms)
            if rc != StatusCode.IN_PROGRESS:
                if rc == StatusCode.SUCCESS:
                    return None
                return self._collect_transfer_failure_reason(statuses)
            if sla_ms > 0 and (time.perf_counter() - start) * 1000 >= sla_ms:
                return f"KV transfer exceeded SLA {sla_ms}ms"

    @staticmethod
    def _collect_transfer_failure_reason(statuses: List[TransferStatus]) -> str:
        for status in statuses:
            if status.Failed():
                return f"KV transfer failed: {status.Message()}"
        return "KV transfer failed due to unknown reason"

    def _notify_decode_for_room(
        self,
        room: int,
        status: KVPoll,
        failure_reason: Optional[str] = None,
        target_infos: Optional[List[TransferInfo]] = None,
    ) -> None:
        with self._room_notify_lock:
            if room not in self.request_status or self._room_status_notified.get(room):
                return

            emitted_status = status
            emitted_reason = failure_reason

            if emitted_status == KVPoll.Success:
                with self.failure_lock:
                    recorded = self.failure_records.get(room)
                if recorded is not None:
                    emitted_status = KVPoll.Failed
                    emitted_reason = recorded
                elif self.request_status.get(room) == KVPoll.Failed:
                    emitted_status = KVPoll.Failed
                    emitted_reason = (
                        emitted_reason or "request marked Failed before notify"
                    )

            if emitted_status == KVPoll.Failed:
                with self.failure_lock:
                    self.failure_records.setdefault(
                        room, emitted_reason or "KV transfer failed"
                    )
                self.update_status(room, KVPoll.Failed)

            infos = target_infos
            if infos is None:
                with self.transfer_lock:
                    room_infos = self.transfer_infos.get(room)
                    infos = (
                        list(room_infos.values()) if room_infos is not None else None
                    )

            self._room_status_notified[room] = True

        if infos:
            self.notify_decode_status(infos, room, emitted_status, emitted_reason)

    def _conclude_room_failure(
        self, room: int, failure_reason: Optional[str] = None
    ) -> None:
        if failure_reason is None:
            with self.failure_lock:
                failure_reason = self.failure_records.get(room, "KV transfer failed")
        self._notify_decode_for_room(room, KVPoll.Failed, failure_reason)

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
        wait_event: Optional[object] = None,
    ) -> None:
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last_chunk or (is_last_chunk and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            return

        shard_idx = bootstrap_room % self._num_shards
        chunk = TransferKVChunk(
            room=bootstrap_room,
            prefill_kv_indices=kv_indices,
            index_slice=index_slice,
            is_last_chunk=is_last_chunk,
            prefill_aux_index=aux_index,
            state_indices=state_indices,
            num_kv_tokens=num_kv_tokens,
            wait_event=wait_event,
        )
        self._dispatch_transfer_chunk(shard_idx, chunk)

    def _dispatch_transfer_chunk(
        self, shard_idx: int, chunk: TransferKVChunk
    ) -> None:
        if self._synchronous_chunk_transfer:
            self._process_transfer_chunk(chunk, self._transfer_queues[shard_idx])
        else:
            self._transfer_queues[shard_idx].put(chunk)

    def _connect_threadsafe(self, endpoint: str, is_ipv6: bool = False):
        """Thread-local ZMQ socket cache with shared Context.

        Each worker thread gets its own PUSH socket (ZMQ sockets are not
        thread-safe), but all sockets share a single process-level Context
        to avoid creating excessive I/O threads and TCP connections.
        """
        cache = getattr(self._socket_local, "socket_cache", None)
        if cache is None:
            cache = {}
            self._socket_local.socket_cache = cache
        if endpoint not in cache:
            sock = self._zmq_ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.SNDHWM, 0)
            sock.setsockopt(zmq.SNDTIMEO, 5000)
            sock.setsockopt(zmq.LINGER, 0)
            if is_ipv6:
                sock.setsockopt(zmq.IPV6, 1)
            sock.connect(endpoint)
            cache[endpoint] = sock
        return cache[endpoint]

    def _handle_register_message(self, payload: List[bytes]) -> None:
        try:
            register_info = KVArgsRegisterInfo.from_zmq(payload)
            self._add_remote_peer(register_info)
        except Exception:
            logger.exception("Failed to register remote peer")

    def _handle_transfer_message(self, payload: List[bytes]) -> None:
        try:
            transfer_info = TransferInfo.from_zmq(payload)
            with self.transfer_lock:
                # Accept metadata when room is not yet created (None) or
                # in Bootstrapping. Reject for active/terminal states where
                # the worker may already be using transfer_infos.
                # None is allowed because metadata can arrive from decode
                # before the prefill scheduler creates the MoriKVSender.
                current = self.request_status.get(transfer_info.room)
                if current is not None and current != KVPoll.Bootstrapping:
                    logger.debug(
                        "Ignoring stale transfer info for room %s (status=%s)",
                        transfer_info.room,
                        current,
                    )
                    return
                infos = self.transfer_infos.setdefault(transfer_info.room, {})
                infos[transfer_info.engine_key] = transfer_info

                if len(infos) >= transfer_info.required_dst_info_num:
                    self.resolve_kv_replica_factor(infos)
                    # All decode peers reported their dst metadata; pick a
                    # non-None decode_prefix_len if any peer set it (they
                    # should all agree, but be defensive). 0 means "no
                    # prefix hit", which is the same as "feature off".
                    chosen_prefix_len = next(
                        (
                            info.decode_prefix_len
                            for info in infos.values()
                            if info.decode_prefix_len is not None
                        ),
                        0,
                    )
                    self.req_to_decode_prefix_len[transfer_info.room] = (
                        chosen_prefix_len
                    )
                    if chosen_prefix_len > 0:
                        # Surface incremental KV transfer at INFO so it's
                        # visible without bumping the global log level.
                        logger.info(
                            "MoriKV incremental: room=%s prefix_len=%s peers=%s",
                            transfer_info.room,
                            chosen_prefix_len,
                            len(infos),
                        )
                    else:
                        logger.debug(
                            "Bootstrap room %s got enough transfer info (%s), "
                            "decode_prefix_len=0",
                            transfer_info.room,
                            len(infos),
                        )
                    self.update_status(transfer_info.room, KVPoll.WaitingForInput)
        except Exception:
            logger.exception("Failed to parse transfer info message")

    def _validate_message(self, msg: List[bytes]) -> Optional[List[bytes]]:
        if not msg or msg[0] != MORI_GUARD:
            logger.warning("Received malformed bootstrap message")
            return None
        payload = msg[1:]
        if not payload:
            return None
        return payload

    def _handle_abort_message(self, msg: List[bytes]) -> None:
        """Handle best-effort ABORT notifications from the decode side."""
        if len(msg) < 2:
            logger.warning("Malformed ABORT message: too few frames (%d)", len(msg))
            return

        try:
            bootstrap_room = int(msg[1].decode("ascii"))
        except (ValueError, UnicodeDecodeError):
            logger.warning("Malformed ABORT message: invalid room field %r", msg[1])
            return

        with self.transfer_lock:
            current = self.request_status.get(bootstrap_room)
            if current is None:
                logger.debug(
                    "ABORT for room %s is not tracked; ignoring",
                    bootstrap_room,
                )
                return
            if current == KVPoll.Success:
                logger.debug(
                    "ABORT for room %s already succeeded; ignoring",
                    bootstrap_room,
                )
                return
            if current == KVPoll.Failed:
                return

            self.update_status(bootstrap_room, KVPoll.Failed)

        logger.debug("Room %s marked Failed via ABORT from decode", bootstrap_room)

    def _start_bootstrap_thread(self) -> None:
        def bootstrap_worker():
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    if not msg:
                        continue

                    tag = msg[0]
                    if tag == _TAG_ABORT:
                        self._handle_abort_message(msg)
                        continue
                    if tag == b"STAGING_RSP":
                        if self.enable_staging:
                            with self.transfer_lock:
                                handle_staging_rsp(msg, self.transfer_infos)
                        else:
                            logger.warning(
                                "Mori STAGING_RSP received while staging is disabled"
                            )
                        continue
                    if tag == b"WATERMARK":
                        if self.enable_staging:
                            handle_watermark_msg(self._staging_ctx, msg)
                        else:
                            logger.warning(
                                "Mori WATERMARK received while staging is disabled"
                            )
                        continue

                    payload = self._validate_message(msg)
                    if payload is None:
                        continue
                    room = payload[0].decode("ascii")

                    if room == "None":
                        self._handle_register_message(payload)
                    else:
                        self._handle_transfer_message(payload)
                except Exception:
                    logger.exception("Bootstrap worker failed")

        threading.Thread(target=bootstrap_worker, daemon=True).start()

    def _cleanup_room_tracking(self, bootstrap_room: int) -> None:
        bootstrap_addr = self.room_to_bootstrap_addr.pop(bootstrap_room, None)
        if bootstrap_addr is not None:
            rooms = self.addr_to_rooms_tracker.get(bootstrap_addr)
            if rooms is not None:
                rooms.discard(bootstrap_room)
                if not rooms:
                    self.addr_to_rooms_tracker.pop(bootstrap_addr, None)

    def _start_decode_thread(self) -> None:
        def decode_worker():
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    if msg and msg[0] == MoriKVManager.AUX_DATA_HEADER:
                        self._handle_aux_data(msg)
                        continue
                    if msg and msg[0] == b"STAGING_REQ":
                        self._handle_staging_req(msg)
                        continue
                    if msg and msg[0] == b"CHUNK_READY":
                        if not self.enable_staging or self._staging_handler is None:
                            logger.warning(
                                "Mori CHUNK_READY received while staging is unavailable"
                            )
                            continue
                        room = int(msg[1].decode("ascii"))
                        chunk_idx = int(msg[2].decode("ascii"))
                        page_start = int(msg[3].decode("ascii"))
                        num_pages = int(msg[4].decode("ascii"))
                        writer_id = (
                            msg[6].decode("ascii")
                            if len(msg) > 6
                            else "unknown"
                        )
                        self._staging_handler.handle_chunk_arrived(
                            room,
                            chunk_idx,
                            page_start,
                            num_pages,
                            writer_id,
                        )
                        continue

                    if not msg or msg[0] != MORI_GUARD:
                        logger.warning(
                            "Received malformed status message on decode worker"
                        )
                        continue
                    payload = msg[1:]
                    if len(payload) < 3:
                        logger.warning("Incomplete status payload received")
                        continue
                    bootstrap_room = int(payload[0].decode("ascii"))
                    if bootstrap_room not in self.request_status:
                        logger.debug(
                            "Dropping late status for cleared room %s",
                            bootstrap_room,
                        )
                        continue
                    status_code = int(payload[1].decode("ascii"))
                    prefill_rank = int(payload[2].decode("ascii"))
                    failure_reason = (
                        payload[3].decode("utf-8")
                        if len(payload) > 3 and payload[3]
                        else None
                    )

                    if status_code == KVPoll.Success:
                        tracker = self.prefill_response_tracker[bootstrap_room]
                        tracker.add(prefill_rank)
                        expected = self.required_prefill_response_num_table.get(
                            bootstrap_room, 1
                        )
                        if len(tracker) >= expected:
                            self.prefill_response_tracker.pop(bootstrap_room, None)
                            self.update_status(bootstrap_room, KVPoll.Success)
                            self._cleanup_room_tracking(bootstrap_room)
                    elif status_code == KVPoll.Failed:
                        if failure_reason:
                            self.record_failure(bootstrap_room, failure_reason)
                        self.prefill_response_tracker.pop(bootstrap_room, None)
                        self.update_status(bootstrap_room, KVPoll.Failed)
                        self._cleanup_room_tracking(bootstrap_room)
                    else:
                        logger.warning(
                            "Unknown status code %s received for room %s",
                            status_code,
                            bootstrap_room,
                        )
                except Exception:
                    logger.exception("Decode status worker failed")

        threading.Thread(target=decode_worker, daemon=True).start()

    def _compute_prefill_unique_rank(self) -> int:
        """Unique id per prefill sender, encoding TP/PP/CP ranks.
        Must match Mooncake's formula so decode's response set size matches
        expected_response_num when multiple CP ranks participate."""
        return (
            self.attn_tp_rank * (self.pp_size * self.attn_cp_size)
            + self.pp_rank * self.attn_cp_size
            + self.attn_cp_rank
        )

    def notify_decode_status(
        self,
        infos: List[TransferInfo],
        bootstrap_room: int,
        status: KVPoll,
        failure_reason: Optional[str] = None,
    ) -> None:
        if not infos:
            return
        payload = [
            MORI_GUARD,
            str(bootstrap_room).encode("ascii"),
            str(int(status)).encode("ascii"),
            str(self._compute_prefill_unique_rank()).encode("ascii"),
            failure_reason.encode("utf-8") if failure_reason else b"",
        ]
        for info in infos:
            try:
                na = NetworkAddress(info.endpoint, info.dst_port)
                socket = self._connect_threadsafe(na.to_tcp(), is_ipv6=na.is_ipv6)
                socket.send_multipart(payload)
            except Exception:
                logger.exception(
                    "Failed to sync status %s to decode endpoint %s:%s for room %s",
                    status,
                    info.endpoint,
                    info.dst_port,
                    bootstrap_room,
                )

    def _add_remote_peer(self, register_info: KVArgsRegisterInfo) -> None:
        engine_key = register_info.engine_key
        if engine_key in self.decode_kv_args_table:
            logger.debug("Remote peer %s already registered. Skipping.", engine_key)
            return
        self.engine.register_remote_engine(register_info.engine_desc)
        self.decode_kv_args_table[engine_key] = register_info
        logger.debug(
            "Registered decode peer %s (%s:%s)",
            engine_key,
            register_info.endpoint,
            register_info.dst_port,
        )

    def _get_mha_mem_desc_slices(
        self, dst_mem_descs: List[MemoryDesc]
    ) -> tuple[
        List[MemoryDesc], List[MemoryDesc], List[MemoryDesc], List[MemoryDesc], int
    ]:
        src_descs = self.kv_mem_descs
        if not src_descs:
            raise RuntimeError("KV memory descriptors are empty on prefill side")

        num_local_layers = len(src_descs) // 2
        src_k_descs = src_descs[:num_local_layers]
        src_v_descs = src_descs[num_local_layers:]

        # Both peers expose the same PP-local layout. Their descriptor indices
        # are already aligned, so applying the Prefill rank's global layer
        # offset would incorrectly index into a local list.
        if len(src_descs) == len(dst_mem_descs):
            dst_k_descs = dst_mem_descs[:num_local_layers]
            dst_v_descs = dst_mem_descs[num_local_layers:]
            return (
                src_k_descs,
                src_v_descs,
                dst_k_descs,
                dst_v_descs,
                num_local_layers,
            )

        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + num_local_layers
        dst_total_layers = len(dst_mem_descs) // 2
        if len(dst_mem_descs) < 2 or end_layer > dst_total_layers:
            raise ValueError(
                "Destination KV descriptors do not match prefill pp configuration"
            )
        dst_k_descs = dst_mem_descs[start_layer:end_layer]
        if (
            num_local_layers < dst_total_layers
            and dst_total_layers % num_local_layers != 0
        ):
            # Decode has draft-model KV while Prefill has target-model KV only:
            # [K_main..., V_main..., draft_K..., draft_V...].
            multiplier_ratio = dst_total_layers // num_local_layers
            dst_v_offset = num_local_layers * multiplier_ratio
        else:
            dst_v_offset = dst_total_layers
        dst_v_descs = dst_mem_descs[
            dst_v_offset + start_layer : dst_v_offset + end_layer
        ]
        return src_k_descs, src_v_descs, dst_k_descs, dst_v_descs, num_local_layers

    def _get_mla_mem_desc_slices(
        self, dst_mem_descs: List[MemoryDesc]
    ) -> tuple[List[MemoryDesc], List[MemoryDesc], int]:
        src_descs = self.kv_mem_descs
        num_local_layers = len(src_descs)
        # Same-PP peers register matching local descriptor lists.
        if len(src_descs) == len(dst_mem_descs):
            return src_descs, dst_mem_descs, num_local_layers

        start_layer = self.kv_args.prefill_start_layer
        end_layer = getattr(self.kv_args, "prefill_end_layer", None)
        if end_layer is None:
            end_layer = start_layer + len(src_descs)
        num_target_descs = end_layer - start_layer
        num_draft_descs = len(src_descs) - num_target_descs
        if num_target_descs < 0 or num_draft_descs < 0:
            raise ValueError(
                "Invalid prefill MLA descriptor geometry: "
                f"start={start_layer}, end={end_layer}, src={len(src_descs)}"
            )
        if end_layer > len(dst_mem_descs) - num_draft_descs:
            raise ValueError(
                "Destination MLA KV descriptors do not match prefill pp configuration"
            )
        dst_slice = list(dst_mem_descs[start_layer:end_layer])
        if num_draft_descs:
            dst_slice.extend(dst_mem_descs[-num_draft_descs:])
        if len(dst_slice) != len(src_descs):
            raise ValueError(
                "Destination MLA KV descriptor count mismatch: "
                f"src={len(src_descs)}, dst={len(dst_slice)}"
            )
        return src_descs, dst_slice, num_target_descs

    @staticmethod
    def _validate_batch_transfer_plan(
        src_desc: MemoryDesc,
        dst_desc: MemoryDesc,
        plan: BatchTransferPlan,
        *,
        context: str,
    ) -> None:
        counts = (len(plan.local_offsets), len(plan.remote_offsets), len(plan.sizes))
        if len(set(counts)) != 1:
            raise ValueError(
                f"{context} transfer plan length mismatch: "
                f"local={counts[0]}, remote={counts[1]}, sizes={counts[2]}"
            )
        if not plan.sizes:
            return
        if min(plan.local_offsets) < 0 or min(plan.remote_offsets) < 0:
            raise ValueError(f"{context} transfer plan contains a negative offset")
        if min(plan.sizes) <= 0:
            raise ValueError(f"{context} transfer plan contains a non-positive size")

        local_end = max(
            offset + size for offset, size in zip(plan.local_offsets, plan.sizes)
        )
        remote_end = max(
            offset + size for offset, size in zip(plan.remote_offsets, plan.sizes)
        )
        src_size = int(src_desc.size)
        dst_size = int(dst_desc.size)
        if local_end > src_size or remote_end > dst_size:
            raise ValueError(
                f"{context} transfer exceeds registered memory: "
                f"local_end={local_end}, src_size={src_size}, "
                f"remote_end={remote_end}, dst_size={dst_size}"
            )

    def _submit_batch_transfer_plan(
        self,
        src_desc: MemoryDesc,
        dst_desc: MemoryDesc,
        plan: BatchTransferPlan,
        *,
        context: str = "Mori",
    ) -> List[TransferStatus]:
        if plan.empty():
            return []

        self._validate_batch_transfer_plan(
            src_desc,
            dst_desc,
            plan,
            context=context,
        )

        if (
            getattr(self, "_diagnostic_log_ranges", False)
            and self._diagnostic_plan_count < self._diagnostic_plan_limit
        ):
            logger.info(
                "Mori diagnostic transfer plan: rank=%s sequence=%s context=%s "
                "segments=%s local_min=%s local_end=%s remote_min=%s "
                "remote_end=%s bytes=%s src=(%s) dst=(%s)",
                self.attn_tp_rank,
                self._diagnostic_plan_count,
                context,
                len(plan.sizes),
                min(plan.local_offsets),
                max(
                    offset + size
                    for offset, size in zip(plan.local_offsets, plan.sizes)
                ),
                min(plan.remote_offsets),
                max(
                    offset + size
                    for offset, size in zip(plan.remote_offsets, plan.sizes)
                ),
                sum(plan.sizes),
                self._memory_desc_fields(src_desc),
                self._memory_desc_fields(dst_desc),
            )
            self._diagnostic_plan_count += 1

        transfer_uid = self.engine.allocate_transfer_uid()

        statuses = self.engine.batch_write(
            [src_desc],
            [plan.local_offsets],
            [dst_desc],
            [plan.remote_offsets],
            [plan.sizes],
            [transfer_uid],
        )
        return statuses

    def _build_contiguous_transfer_plan(
        self, grouped_plan: GroupedIndexPlan, item_len: int
    ) -> BatchTransferPlan:
        # Reuse grouped indices across all layers/tensors that share the same item length.
        return grouped_plan.materialize(item_len)

    def _build_tp_slice_config(self, peer_info: KVArgsRegisterInfo) -> TPSliceConfig:
        page_size = self.kv_args.page_size

        src_item_len = self.kv_args.kv_item_lens[0]
        dst_item_len = peer_info.dst_kv_item_len

        bytes_per_token_src = src_item_len // page_size
        bytes_per_token_dst = dst_item_len // page_size

        prefill_tp_size = self.attn_tp_size
        decode_tp_size = peer_info.decode_tp_size

        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * prefill_tp_size

        src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // decode_tp_size)

        bytes_per_head_slice = bytes_per_token_dst // dst_heads_per_rank
        if bytes_per_head_slice == 0:
            raise ValueError("Head slice size evaluates to zero")

        src_replication = max(1, prefill_tp_size // total_kv_heads)

        local_tp_rank = self.kv_args.engine_rank % prefill_tp_size
        dst_tp_rank = peer_info.decode_tp_rank % decode_tp_size

        if prefill_tp_size > decode_tp_size:
            src_head_start = 0
            num_heads_to_send = src_heads_per_rank
            unique_head_idx = local_tp_rank // src_replication
            dst_head_start = (unique_head_idx * src_heads_per_rank) % dst_heads_per_rank
        else:
            src_head_start = (dst_tp_rank * dst_heads_per_rank) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start = 0

        src_head_slice_offset = src_head_start * bytes_per_head_slice
        dst_head_slice_offset = dst_head_start * bytes_per_head_slice
        heads_bytes_per_token = num_heads_to_send * bytes_per_head_slice

        if heads_bytes_per_token > bytes_per_token_dst:
            raise ValueError(
                "Slice size exceeds destination token capacity for TP slice transfer"
            )

        return TPSliceConfig(
            page_size=page_size,
            src_item_len=src_item_len,
            dst_item_len=dst_item_len,
            bytes_per_token_src=bytes_per_token_src,
            bytes_per_token_dst=bytes_per_token_dst,
            src_head_slice_offset=src_head_slice_offset,
            dst_head_slice_offset=dst_head_slice_offset,
            heads_bytes_per_token_to_send=heads_bytes_per_token,
        )

    def _build_tp_slice_transfer_plan(
        self,
        kv_indices: npt.NDArray[np.int32],
        dst_indices: npt.NDArray[np.int32],
        tp_cfg: TPSliceConfig,
    ) -> BatchTransferPlan:
        if kv_indices.size == 0 or dst_indices.size == 0:
            return BatchTransferPlan([], [], [])

        limit = min(kv_indices.size, dst_indices.size)
        if not limit:
            return BatchTransferPlan([], [], [])

        src_pages = kv_indices[:limit].astype(np.int64)
        dst_pages = dst_indices[:limit].astype(np.int64)
        token_slots = np.arange(tp_cfg.page_size, dtype=np.int64)

        src_page_bases = src_pages * tp_cfg.src_item_len
        dst_page_bases = dst_pages * tp_cfg.dst_item_len

        src_token_offsets = token_slots * tp_cfg.bytes_per_token_src
        dst_token_offsets = token_slots * tp_cfg.bytes_per_token_dst

        local_offsets = (
            (
                src_page_bases[:, np.newaxis]
                + src_token_offsets
                + tp_cfg.src_head_slice_offset
            )
            .flatten()
            .tolist()
        )
        remote_offsets = (
            (
                dst_page_bases[:, np.newaxis]
                + dst_token_offsets
                + tp_cfg.dst_head_slice_offset
            )
            .flatten()
            .tolist()
        )

        num_transfers = limit * tp_cfg.page_size
        sizes = [tp_cfg.heads_bytes_per_token_to_send] * num_transfers

        if not local_offsets:
            return BatchTransferPlan([], [], [])

        return BatchTransferPlan(
            local_offsets=local_offsets,
            remote_offsets=remote_offsets,
            sizes=sizes,
        )

    def send_kvcache(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
        staging_offset: Optional[int] = None,
    ) -> List[TransferStatus]:
        if _MORI_DIAGNOSTIC_SKIP_KV:
            logger.warning(
                "Mori diagnostic: skipping KV payload for %d prompt slots",
                len(prefill_kv_indices),
            )
            return []

        if (
            self.enable_staging
            and staging_offset is not None
            and staging_offset >= 0
        ):
            if peer_info.staging_mem_desc is None:
                raise RuntimeError(
                    "Mori staged transfer is missing the decode staging descriptor"
                )
            return self._send_staged_kvcache(
                peer_info,
                prefill_kv_indices,
                staging_offset,
            )

        grouped_plan = GroupedIndexPlan.from_groups(
            *group_concurrent_contiguous(
                prefill_kv_indices,
                dst_kv_indices,
            )
        )
        statuses: List[TransferStatus] = []
        kv_item_len = self.kv_args.kv_item_lens[0]

        if self.is_mla_backend or self.is_hybrid_mla_backend:
            src_descs, dst_descs, target_desc_count = (
                self._get_mla_mem_desc_slices(peer_info.dst_kv_mem_descs)
            )
            draft_desc_count = len(src_descs) - target_desc_count
            dst_item_lens = list(
                peer_info.dst_kv_item_lens[
                    self.kv_args.prefill_start_layer : self.kv_args.prefill_start_layer
                    + target_desc_count
                ]
            )
            if draft_desc_count:
                dst_item_lens.extend(peer_info.dst_kv_item_lens[-draft_desc_count:])
            if len(dst_item_lens) != len(src_descs):
                raise ValueError(
                    "Destination MLA KV item-length count mismatch: "
                    f"src={len(src_descs)}, dst={len(dst_item_lens)}"
                )

            # EAGLE's prompt-side draft state is handed off through the state
            # components below. Its appended draft KV descriptors must not be
            # populated with the full target prompt. That old MHA-style path
            # mixed target/draft descriptor layouts and could submit OOB Mori
            # plans on hybrid MLA models such as GLM-5.3 Flash.
            for layer_id in range(target_desc_count):
                src_item_len = self.kv_args.kv_item_lens[layer_id]
                dst_item_len = dst_item_lens[layer_id]
                if src_item_len != dst_item_len:
                    raise ValueError(
                        "Mori MLA source/destination item length mismatch at "
                        f"descriptor {layer_id}: src={src_item_len}, "
                        f"dst={dst_item_len}"
                    )
                layer_plan = self._build_contiguous_transfer_plan(
                    grouped_plan, src_item_len
                )
                statuses.extend(
                    self._submit_batch_transfer_plan(
                        src_descs[layer_id],
                        dst_descs[layer_id],
                        layer_plan,
                        context=f"Mori MLA KV descriptor {layer_id}",
                    )
                )
            return statuses

        (
            src_k_descs,
            src_v_descs,
            dst_k_descs,
            dst_v_descs,
            layers_current_pp_stage,
        ) = self._get_mha_mem_desc_slices(peer_info.dst_kv_mem_descs)

        if peer_info.decode_tp_size != self.attn_tp_size:
            tp_cfg = self._build_tp_slice_config(peer_info)
            slice_plan = self._build_tp_slice_transfer_plan(
                prefill_kv_indices, dst_kv_indices, tp_cfg
            )
            for layer_id in range(layers_current_pp_stage):
                statuses.extend(
                    self._submit_batch_transfer_plan(
                        src_k_descs[layer_id],
                        dst_k_descs[layer_id],
                        slice_plan,
                    )
                )
                statuses.extend(
                    self._submit_batch_transfer_plan(
                        src_v_descs[layer_id],
                        dst_v_descs[layer_id],
                        slice_plan,
                    )
                )
            return statuses

        layer_plan = self._build_contiguous_transfer_plan(grouped_plan, kv_item_len)
        for layer_id in range(layers_current_pp_stage):
            statuses.extend(
                self._submit_batch_transfer_plan(
                    src_k_descs[layer_id],
                    dst_k_descs[layer_id],
                    layer_plan,
                )
            )
            statuses.extend(
                self._submit_batch_transfer_plan(
                    src_v_descs[layer_id],
                    dst_v_descs[layer_id],
                    layer_plan,
                )
            )
        return statuses

    def _send_staged_kvcache(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_kv_indices: npt.NDArray[np.int32],
        staging_offset: int,
    ) -> List[TransferStatus]:
        num_target = self._num_target_kv_entries()
        if peer_info.dst_num_target_kv_entries != num_target:
            raise ValueError(
                "Mori staged transfer target descriptor count mismatch: "
                f"prefill={num_target}, decode="
                f"{peer_info.dst_num_target_kv_entries}"
            )
        if list(self.kv_args.kv_item_lens[:num_target]) != list(
            peer_info.dst_kv_item_lens[:num_target]
        ):
            raise ValueError(
                "Mori staged transfer target item lengths do not match "
                "between prefill and decode"
            )
        if len(self.kv_mem_descs) < num_target:
            raise ValueError(
                "Mori staged transfer source descriptor count is too small: "
                f"target={num_target}, local={len(self.kv_mem_descs)}"
            )
        if peer_info.staging_mem_desc is None:
            raise RuntimeError("Mori staging descriptor is missing")

        dst_indices = np.arange(len(prefill_kv_indices), dtype=np.int32)
        grouped_plan = GroupedIndexPlan.from_groups(
            *group_concurrent_contiguous(
                prefill_kv_indices,
                dst_indices,
            )
        )

        statuses: List[TransferStatus] = []
        layer_offset = 0
        for layer_id in range(num_target):
            item_len = self.kv_args.kv_item_lens[layer_id]
            layer_plan = grouped_plan.materialize(item_len)
            layer_plan = BatchTransferPlan(
                local_offsets=layer_plan.local_offsets,
                remote_offsets=[
                    offset + staging_offset + layer_offset
                    for offset in layer_plan.remote_offsets
                ],
                sizes=layer_plan.sizes,
            )
            statuses.extend(
                self._submit_batch_transfer_plan(
                    self.kv_mem_descs[layer_id],
                    peer_info.staging_mem_desc,
                    layer_plan,
                    context=f"Mori staged KV descriptor {layer_id}",
                )
            )
            layer_offset += item_len
        return statuses

    def copy_staged_kv_to_pool(
        self,
        staging_offset: int,
        dst_kv_indices: npt.NDArray[np.int32],
    ) -> None:
        if self.staging_mem_desc is None:
            raise RuntimeError("Mori local staging descriptor is missing")

        num_target = self._num_target_kv_entries()
        src_indices = np.arange(len(dst_kv_indices), dtype=np.int32)
        grouped_plan = GroupedIndexPlan.from_groups(
            *group_concurrent_contiguous(
                src_indices,
                dst_kv_indices,
            )
        )

        statuses: List[TransferStatus] = []
        layer_offset = 0
        for layer_id in range(num_target):
            item_len = self.kv_args.kv_item_lens[layer_id]
            layer_plan = grouped_plan.materialize(item_len)
            layer_plan = BatchTransferPlan(
                local_offsets=[
                    offset + staging_offset + layer_offset
                    for offset in layer_plan.local_offsets
                ],
                remote_offsets=layer_plan.remote_offsets,
                sizes=layer_plan.sizes,
            )
            statuses.extend(
                self._submit_batch_transfer_plan(
                    self.staging_mem_desc,
                    self.kv_mem_descs[layer_id],
                    layer_plan,
                    context=f"Mori staging-to-KV descriptor {layer_id}",
                )
            )
            layer_offset += item_len

        failure_reason = self._wait_transfer_completion(statuses)
        if failure_reason is not None:
            raise RuntimeError(f"Mori staging-to-KV copy failed: {failure_reason}")

    def send_aux(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
    ) -> List[TransferStatus]:
        if self._send_aux_rdma:
            return self.send_aux_rdma(peer_info, prefill_aux_index, dst_aux_index, room)
        return self.send_aux_tcp(peer_info, prefill_aux_index, dst_aux_index, room)

    def send_aux_rdma(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
    ) -> List[TransferStatus]:
        if not self.aux_mem_descs or len(self.aux_mem_descs) != len(
            peer_info.dst_aux_mem_descs
        ):
            return self.send_aux_tcp(peer_info, prefill_aux_index, dst_aux_index, room)

        src_descs: List[MemoryDesc] = []
        dst_descs: List[MemoryDesc] = []
        local_offsets: List[List[int]] = []
        remote_offsets: List[List[int]] = []
        sizes: List[List[int]] = []
        uids = []
        for i in range(len(self.aux_mem_descs)):
            item_len = self.kv_args.aux_item_lens[i]
            src_descs.append(self.aux_mem_descs[i])
            dst_descs.append(peer_info.dst_aux_mem_descs[i])
            local_offsets.append([prefill_aux_index * item_len])
            remote_offsets.append([dst_aux_index * item_len])
            sizes.append([item_len])
            uids.append(self.engine.allocate_transfer_uid())
        return list(
            self.engine.batch_write(
                src_descs, local_offsets, dst_descs, remote_offsets, sizes, uids
            )
        )

    def send_aux_tcp(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
    ) -> List[TransferStatus]:
        for i in range(len(self.kv_args.aux_data_ptrs)):
            length = self.kv_args.aux_item_lens[i]
            src_addr = self.kv_args.aux_data_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)
            self._send_aux_data_to_endpoint(
                remote=peer_info.endpoint,
                dst_port=peer_info.dst_port,
                room=room,
                buffer_index=i,
                aux_index=dst_aux_index,
                data=data,
            )
        return []  # TCP path has no TransferStatus to poll

    def _send_aux_data_to_endpoint(
        self, remote, dst_port, room, buffer_index, aux_index, data
    ):
        na = NetworkAddress(remote, dst_port)
        socket = self._connect_threadsafe(na.to_tcp(), is_ipv6=na.is_ipv6)
        socket.send_multipart(
            [
                MoriKVManager.AUX_DATA_HEADER,
                str(room).encode("ascii"),
                str(buffer_index).encode("ascii"),
                str(aux_index).encode("ascii"),
                struct.pack(">I", len(data)),
                data,
            ]
        )

    def send_state(
        self,
        peer_info: KVArgsRegisterInfo,
        src_state_indices: List[npt.NDArray[np.int32]],
        dst_state_indices: List[npt.NDArray[np.int32]],
    ) -> List[TransferStatus]:
        # Guard: no local state tensors -> no-op (e.g. SWA layers=0 on this PP rank)
        if not self.state_mem_descs:
            return []

        state_types = self.kv_args.state_types
        if not state_types:
            raise RuntimeError(
                "PD state transfer failed: kv_args.state_types is empty but "
                "state_indices were provided"
            )

        if len(peer_info.dst_state_mem_descs) != len(self.state_mem_descs):
            raise RuntimeError(
                f"PD state transfer failed: state component count mismatch "
                f"(local={len(self.state_mem_descs)}, "
                f"remote={len(peer_info.dst_state_mem_descs)})"
            )

        src_state_item_lens = self.kv_args.state_item_lens
        src_state_slot_strides = self.kv_args.state_slot_strides
        src_state_dim_per_tensor = self.kv_args.state_dim_per_tensor

        statuses: List[TransferStatus] = []
        for i, st in enumerate(state_types):
            if st in _MORI_DIAGNOSTIC_SKIP_STATE_TYPES:
                continue
            src_indices = src_state_indices[i] if i < len(src_state_indices) else None
            dst_indices = dst_state_indices[i] if i < len(dst_state_indices) else None
            if st == "dsa_tail":
                if src_indices is None or dst_indices is None:
                    continue
                if src_indices.size == 0 and dst_indices.size == 0:
                    continue
                src_descs = self.state_mem_descs[i]
                dst_descs = peer_info.dst_state_mem_descs[i]
                src_lens = (
                    src_state_item_lens[i] if i < len(src_state_item_lens) else []
                )
                dst_lens = (
                    peer_info.dst_state_item_lens[i]
                    if i < len(peer_info.dst_state_item_lens)
                    else []
                )
                statuses.extend(
                    self._send_dsa_tail_state(
                        src_indices,
                        dst_indices,
                        src_descs,
                        dst_descs,
                        src_lens,
                        dst_lens,
                    )
                )
                continue
            if src_indices is None or src_indices.size == 0:
                continue
            if dst_indices is None or dst_indices.size == 0:
                continue

            src_descs = self.state_mem_descs[i]
            dst_descs = peer_info.dst_state_mem_descs[i]
            src_desc_offsets = self.state_mem_desc_offsets[i]
            peer_desc_offsets = getattr(
                peer_info, "dst_state_mem_desc_offsets", []
            )
            dst_desc_offsets = (
                peer_desc_offsets[i]
                if i < len(peer_desc_offsets)
                else [0] * len(dst_descs)
            )
            src_lens = src_state_item_lens[i] if i < len(src_state_item_lens) else []
            src_strides = (
                src_state_slot_strides[i]
                if i < len(src_state_slot_strides)
                else src_lens
            )
            dst_lens = (
                peer_info.dst_state_item_lens[i]
                if i < len(peer_info.dst_state_item_lens)
                else []
            )
            peer_slot_strides = getattr(peer_info, "dst_state_slot_strides", [])
            dst_strides = (
                peer_slot_strides[i] if i < len(peer_slot_strides) else dst_lens
            )
            src_dims = (
                src_state_dim_per_tensor[i] if i < len(src_state_dim_per_tensor) else []
            )
            dst_dims = (
                peer_info.dst_state_dim_per_tensor[i]
                if i < len(peer_info.dst_state_dim_per_tensor)
                else []
            )

            if st == "mamba":
                statuses.extend(
                    self._send_mamba_state(
                        peer_info,
                        src_indices,
                        dst_indices,
                        src_descs,
                        dst_descs,
                        src_desc_offsets,
                        dst_desc_offsets,
                        src_lens,
                        dst_lens,
                        src_strides,
                        dst_strides,
                        src_dims,
                        dst_dims,
                    )
                )
            elif st in ("swa", "dsa", "swa_ring", "c128_state", "minimax_index_k"):
                statuses.extend(
                    self._send_swa_dsa_state(
                        peer_info,
                        src_indices,
                        dst_indices,
                        src_descs,
                        src_lens,
                        dst_descs,
                        st,
                    )
                )
            else:
                raise RuntimeError(f"PD state transfer failed: unknown state_type={st}")

        return statuses

    def _send_dsa_tail_state(
        self,
        src_state_indices: npt.NDArray[np.int32],
        dst_state_indices: npt.NDArray[np.int32],
        src_state_mem_descs: List[MemoryDesc],
        dst_state_mem_descs: List[MemoryDesc],
        src_state_item_lens: List[int],
        dst_state_item_lens: List[int],
    ) -> List[TransferStatus]:
        dst_state_mem_descs = slice_dsa_tail_dst_ptrs_for_pp(
            src_state_mem_descs,
            dst_state_mem_descs,
            self.kv_args.prefill_start_layer,
            getattr(self.kv_args, "prefill_end_layer", None),
        )
        dst_state_item_lens = slice_dsa_tail_dst_ptrs_for_pp(
            src_state_mem_descs,
            dst_state_item_lens,
            self.kv_args.prefill_start_layer,
            getattr(self.kv_args, "prefill_end_layer", None),
        )

        if not (
            len(src_state_mem_descs)
            == len(dst_state_mem_descs)
            == len(src_state_item_lens)
            == len(dst_state_item_lens)
        ):
            raise RuntimeError(
                "PD state transfer failed: DSA tail descriptor metadata mismatch "
                f"(src_descs={len(src_state_mem_descs)}, "
                f"dst_descs={len(dst_state_mem_descs)}, "
                f"src_item_lens={len(src_state_item_lens)}, "
                f"dst_item_lens={len(dst_state_item_lens)})"
            )

        src_indices = src_state_indices.tolist()
        dst_indices = dst_state_indices.tolist()
        statuses: List[TransferStatus] = []
        for src_desc, dst_desc, src_item_len, dst_item_len in zip(
            src_state_mem_descs,
            dst_state_mem_descs,
            src_state_item_lens,
            dst_state_item_lens,
        ):
            blocks = build_dsa_tail_transfer_blocks(
                [0],
                [src_item_len],
                [0],
                src_indices,
                dst_indices,
                [dst_item_len],
            )
            statuses.extend(
                self._submit_batch_transfer_plan(
                    src_desc,
                    dst_desc,
                    BatchTransferPlan(
                        local_offsets=[src for src, _, _ in blocks],
                        remote_offsets=[dst for _, dst, _ in blocks],
                        sizes=[size for _, _, size in blocks],
                    ),
                )
            )
        return statuses

    def _send_mamba_state(
        self,
        peer_info: KVArgsRegisterInfo,
        src_state_indices: npt.NDArray[np.int32],
        dst_state_indices: npt.NDArray[np.int32],
        src_state_mem_descs: List[MemoryDesc],
        dst_state_mem_descs: List[MemoryDesc],
        src_state_mem_desc_offsets: List[int],
        dst_state_mem_desc_offsets: List[int],
        src_state_item_lens: List[int],
        dst_state_item_lens: List[int],
        src_state_slot_strides: List[int],
        dst_state_slot_strides: List[int],
        src_state_dim_per_tensor: List[int],
        dst_state_dim_per_tensor: List[int],
    ) -> List[TransferStatus]:
        state_pairs = pair_mamba_state_indices(
            src_state_indices, dst_state_indices
        )

        tp_mismatch = peer_info.decode_tp_size != self.attn_tp_size

        # If dim info missing, silently degrade to whole-item copy (Mooncake compat)
        if tp_mismatch and (
            not src_state_dim_per_tensor or not dst_state_dim_per_tensor
        ):
            tp_mismatch = False

        if tp_mismatch:
            logger.warning_once(
                "Using Mamba state slice transfer for different TP sizes between prefill and decode. "
                f"Prefill attn_tp_size={self.attn_tp_size}, Decode attn_tp_size={peer_info.decode_tp_size}. "
                "Performance may be affected."
            )

        statuses: List[TransferStatus] = []

        local_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank = peer_info.decode_tp_rank % peer_info.decode_tp_size

        for i, src_desc in enumerate(src_state_mem_descs):
            dst_desc = dst_state_mem_descs[i]
            src_desc_offset = src_state_mem_desc_offsets[i]
            dst_desc_offset = dst_state_mem_desc_offsets[i]
            src_item_len = src_state_item_lens[i]
            dst_item_len = dst_state_item_lens[i]
            src_slot_stride = src_state_slot_strides[i]
            dst_slot_stride = dst_state_slot_strides[i]

            local_offsets: List[int] = []
            remote_offsets: List[int] = []
            sizes: List[int] = []
            for src_idx, dst_idx in state_pairs:
                if not tp_mismatch:
                    # same-TP: whole item copy
                    src_offset = src_desc_offset + src_idx * src_slot_stride
                    dst_offset = dst_desc_offset + dst_idx * dst_slot_stride
                    size = src_item_len
                else:
                    # TP mismatch slice copy
                    src_dim = src_state_dim_per_tensor[i]
                    dst_dim = dst_state_dim_per_tensor[i]

                    src_bytes_per_dim = src_item_len // src_dim

                    if self.attn_tp_size > peer_info.decode_tp_size:
                        src_dim_start = 0
                        num_dims_to_send = src_dim
                        writers_per_decode = (
                            self.attn_tp_size // peer_info.decode_tp_size
                        )
                        local_writer_idx = local_tp_rank % writers_per_decode
                        dst_dim_start = local_writer_idx * src_dim
                    else:
                        src_dim_start = (dst_tp_rank * dst_dim) % src_dim
                        num_dims_to_send = dst_dim
                        dst_dim_start = 0

                    dst_bytes_per_dim = dst_item_len // dst_dim
                    src_dim_offset = src_dim_start * src_bytes_per_dim
                    dst_dim_offset = dst_dim_start * dst_bytes_per_dim
                    bytes_to_send = num_dims_to_send * src_bytes_per_dim

                    src_offset = (
                        src_desc_offset + src_idx * src_slot_stride + src_dim_offset
                    )
                    dst_offset = (
                        dst_desc_offset + dst_idx * dst_slot_stride + dst_dim_offset
                    )
                    size = bytes_to_send

                local_offsets.append(src_offset)
                remote_offsets.append(dst_offset)
                sizes.append(size)

            statuses.extend(
                self._submit_batch_transfer_plan(
                    src_desc,
                    dst_desc,
                    BatchTransferPlan(
                        local_offsets=local_offsets,
                        remote_offsets=remote_offsets,
                        sizes=sizes,
                    ),
                    context=f"Mori Mamba state tensor {i}",
                )
            )

        return statuses

    def _send_swa_dsa_state(
        self,
        peer_info: KVArgsRegisterInfo,
        src_state_indices: npt.NDArray[np.int32],
        dst_state_indices: npt.NDArray[np.int32],
        src_state_mem_descs: List[MemoryDesc],
        src_state_item_lens: List[int],
        dst_state_mem_descs: List[MemoryDesc],
        state_type: str,
    ) -> List[TransferStatus]:
        # TP mismatch check for non-MLA SWA
        if (
            state_type == "swa"
            and not self.is_mla_backend
            and peer_info.decode_tp_size != self.attn_tp_size
        ):
            raise RuntimeError(
                f"PD state transfer does not support TP-mismatched non-MLA SWA models "
                f"(prefill_tp_size={self.attn_tp_size}, decode_tp_size={peer_info.decode_tp_size})"
            )
        if state_type == "minimax_index_k":
            if self.pp_size is not None and self.pp_size > 1:
                raise RuntimeError(
                    "PD disagg: PP>1 not supported for MiniMax sparse index yet."
                )
            if peer_info.decode_tp_size != self.attn_tp_size:
                raise RuntimeError(
                    "PD disagg: heterogeneous TP not supported for MiniMax sparse index yet."
                )

        common_len = min(src_state_indices.size, dst_state_indices.size)
        if (
            state_type == "c128_state"
            and common_len == 0
            and src_state_indices.size == 0
            and dst_state_indices.size == 0
        ):
            return []
        if common_len == 0 and max(src_state_indices.size, dst_state_indices.size) > 0:
            raise RuntimeError(
                f"No overlapping state indices for state_type={state_type}"
            )
        if src_state_indices.size != dst_state_indices.size:
            # These components are position- or request-indexed: truncating
            # silently misaligns rows and corrupts KV. Paged swa/dsa tolerate
            # a 1-page drift -> keep truncation.
            if state_type in ("swa_ring", "c128_state"):
                raise RuntimeError(
                    f"{state_type.upper()} state index length mismatch: "
                    f"src={src_state_indices.size}, dst={dst_state_indices.size}"
                )
            logger.warning(
                "State index length mismatch for %s: src=%d dst=%d; truncating to common prefix=%d",
                state_type,
                src_state_indices.size,
                dst_state_indices.size,
                common_len,
            )
            src_state_indices = src_state_indices[:common_len]
            dst_state_indices = dst_state_indices[:common_len]

        # Group contiguous indices and issue per-tensor transfers
        grouped_plan = GroupedIndexPlan.from_groups(
            *group_concurrent_contiguous(src_state_indices, dst_state_indices)
        )

        statuses: List[TransferStatus] = []
        for i, src_desc in enumerate(src_state_mem_descs):
            dst_desc = dst_state_mem_descs[i]
            state_item_len = src_state_item_lens[i]

            statuses.extend(
                self._submit_batch_transfer_plan(
                    src_desc,
                    dst_desc,
                    self._build_contiguous_transfer_plan(grouped_plan, state_item_len),
                )
            )

        return statuses

    def _handle_aux_data(self, msg: List[bytes]):
        """Handle AUX_DATA messages received by the decode thread (legacy TCP path)."""
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}")
            return

        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )

    def _handle_staging_req(self, msg: List[bytes]) -> None:
        if not self.enable_staging or self._staging_handler is None:
            logger.warning(
                "Mori STAGING_REQ received while staging is unavailable; ignoring"
            )
            return

        room = int(msg[1].decode("ascii"))
        chunk_idx = int(msg[2].decode("ascii"))
        chunk_num_pages = int(msg[3].decode("ascii"))
        session_id = msg[4].decode("ascii")
        requester_pp_rank = (
            int(msg[5].decode("ascii")) if len(msg) > 5 else None
        )

        receiver = self._staging_ctx.room_receivers.get(room)
        if receiver is None:
            logger.warning(
                "Mori STAGING_REQ has no registered receiver: room=%s chunk=%s",
                room,
                chunk_idx,
            )
            return

        infos = receiver.chunk_staging_infos
        if (
            chunk_idx < len(infos)
            and infos[chunk_idx][0] >= 0
        ):
            alloc_id, offset, alloc_round, alloc_end, _ = infos[chunk_idx]
        elif (
            chunk_idx < len(infos)
            and infos[chunk_idx][1] == StagingAllocator.ALLOC_OVERSIZED
        ):
            alloc_id = -1
            offset = StagingAllocator.ALLOC_OVERSIZED
            alloc_round = 0
            alloc_end = -1
        else:
            num_target = self._num_target_kv_entries()
            required = chunk_num_pages * sum(
                self.kv_args.kv_item_lens[:num_target]
            )
            result = self._staging_ctx.allocator.assign(required)
            if result is None:
                logger.error(
                    "Mori staging allocation failed: room=%s chunk=%s "
                    "required=%s total=%s",
                    room,
                    chunk_idx,
                    required,
                    self._staging_ctx.allocator.total_size,
                )
                alloc_id = -1
                offset = StagingAllocator.ALLOC_OVERSIZED
                alloc_round = 0
                alloc_end = -1
            else:
                alloc_id, offset, alloc_round = result
                alloc_end = offset + required

            while len(infos) <= chunk_idx:
                infos.append((-1, -1, 0, -1, 0))
            infos[chunk_idx] = (
                alloc_id,
                offset,
                alloc_round,
                alloc_end,
                chunk_num_pages,
            )

        self._send_staging_rsp(
            receiver,
            room,
            chunk_idx,
            offset,
            alloc_round,
            alloc_end,
            session_id,
            requester_pp_rank,
        )
        self._staging_handler.register_wm_subscriber(receiver, session_id)

    def _send_staging_rsp(
        self,
        receiver,
        room: int,
        chunk_idx: int,
        offset: int,
        alloc_round: int,
        alloc_end: int,
        session_id: str,
        requester_pp_rank: Optional[int],
    ) -> None:
        bootstrap_infos = self._staging_ctx.room_bootstrap.get(room, [])
        for bootstrap_info in bootstrap_infos:
            if (
                requester_pp_rank is not None
                and bootstrap_info.get("pp_rank") != requester_pp_rank
            ):
                continue
            try:
                sock, lock = receiver._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    sock.send_multipart(
                        [
                            b"STAGING_RSP",
                            str(room).encode("ascii"),
                            str(chunk_idx).encode("ascii"),
                            str(offset).encode("ascii"),
                            str(alloc_round).encode("ascii"),
                            str(alloc_end).encode("ascii"),
                            session_id.encode("ascii"),
                        ]
                    )
            except Exception:
                logger.exception(
                    "Failed to send Mori STAGING_RSP: room=%s chunk=%s",
                    room,
                    chunk_idx,
                )

    def _submit_kv_transfer(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[npt.NDArray[np.int32]]] = None,
    ) -> Tuple[List[TransferStatus], Optional[List[TransferInfo]]]:
        assert self.disaggregation_mode == DisaggregationMode.PREFILL

        if (
            bootstrap_room not in self.request_status
            or self.request_status.get(bootstrap_room) == KVPoll.Failed
        ):
            return [], None

        targets: List[TransferTarget] = []
        target_infos_snapshot: Optional[List[TransferInfo]] = None
        with self.transfer_lock:
            current = self.request_status.get(bootstrap_room)
            if current is None or current == KVPoll.Failed:
                return [], None

            transfer_infos = self.transfer_infos.get(bootstrap_room)
            if not transfer_infos:
                raise RuntimeError(
                    f"No transfer info found for bootstrap_room={bootstrap_room}"
                )

            self.update_status(bootstrap_room, KVPoll.Transferring)
            for info in transfer_infos.values():
                peer_info = self.decode_kv_args_table.get(info.engine_key)
                if not peer_info:
                    raise RuntimeError(
                        f"Peer info missing for engine {info.engine_key}"
                    )
                targets.append(TransferTarget(info=info, peer_info=peer_info))
            target_infos_snapshot = list(transfer_infos.values())

        result_statuses: List[TransferStatus] = []
        try:
            for target in targets:
                info = target.info
                peer_info = target.peer_info

                if not info.is_dummy:
                    dst_indices_chunk = info.dst_kv_indices[index_slice]
                    staging_offset = None
                    if self.enable_staging:
                        _, _, staging_offset = self._staging_chunk_ready(
                            info, index_slice
                        )
                    result_statuses.extend(
                        self.send_kvcache(
                            peer_info,
                            kv_indices,
                            dst_indices_chunk,
                            staging_offset=staging_offset,
                        )
                    )

                if (
                    is_last_chunk
                    and state_indices is not None
                    and not info.is_dummy
                    and self.state_mem_descs
                ):
                    result_statuses.extend(
                        self.send_state(
                            peer_info, state_indices, info.dst_state_indices
                        )
                    )

                if (
                    is_last_chunk
                    and aux_index is not None
                    and info.dst_aux_index >= 0
                    and self.pp_group.is_last_rank
                ):
                    result_statuses.extend(
                        self.send_aux(
                            peer_info, aux_index, info.dst_aux_index, bootstrap_room
                        )
                    )
        except Exception as e:
            logger.exception(
                "Mori KV transfer submission failed for bootstrap_room=%s",
                bootstrap_room,
            )
            raise RuntimeError(f"Transfer submission failed: {e}") from e

        return result_statuses, target_infos_snapshot


class MoriDecodeStagingHandler(DecodeStagingHandler):
    """Mori-specific staging handler for contiguous target-KV descriptors."""

    @classmethod
    def create(cls, kv_manager, scheduler, tp_rank: int):
        allocator = kv_manager._staging_ctx.allocator
        if allocator is None:
            raise RuntimeError(
                "Mori staging is enabled but its decode allocator is missing"
            )
        return cls(
            kv_manager=kv_manager,
            staging_allocator=allocator,
            kv_buffer_info={"page_size": kv_manager.kv_args.page_size},
            decode_tp=kv_manager.attn_tp_size,
            total_kv_heads=0,
            tp_rank=tp_rank,
            scheduler=scheduler,
        )

    def _scatter_region(
        self,
        staging_offset: int,
        page_start: int,
        num_pages: int,
        decode_req,
        receiver,
    ) -> bool:
        page_size = self.kv_buffer_info["page_size"]
        prefix_tokens = decode_req.req.kv.cache_protected_len
        token_start = prefix_tokens + page_start * page_size
        token_end = token_start + num_pages * page_size
        req_pool_idx = decode_req.req.req_pool_idx
        kv_indices = self.scheduler.req_to_token_pool.req_to_token[
            req_pool_idx, token_start:token_end
        ]
        if page_size > 1:
            kv_indices = kv_indices[::page_size] // page_size
        dst_kv_indices = kv_indices.detach().cpu().numpy().astype(np.int32)
        self.kv_manager.copy_staged_kv_to_pool(staging_offset, dst_kv_indices)
        return True

    def submit_chunk_scatter(
        self, room: int, chunk_idx: int, page_start: int, num_pages: int
    ) -> bool:
        decode_req = self._room_to_decode_req.get(room)
        if decode_req is None:
            logger.warning(
                "Mori staging chunk arrived for unregistered room=%s chunk=%s",
                room,
                chunk_idx,
            )
            return False

        receiver = self._room_to_receiver.get(room)
        chunk_infos = receiver.chunk_staging_infos if receiver is not None else []
        if chunk_idx >= len(chunk_infos):
            return False
        alloc_id, staging_offset, _, _, _ = chunk_infos[chunk_idx]
        if staging_offset < 0 or alloc_id < 0:
            return False

        try:
            ok = self._scatter_region(
                staging_offset,
                page_start,
                num_pages,
                decode_req,
                receiver,
            )
        except Exception:
            logger.exception(
                "Mori staging-to-KV copy failed: room=%s chunk=%s",
                room,
                chunk_idx,
            )
            decode_req._staging_failed = True
            ok = False

        self._free_and_send_watermark(alloc_id, decode_req)
        chunk_infos[chunk_idx] = (-1, -1, 0, -1, 0)
        return ok


class MoriKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        super().__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            dest_tp_ranks,
            pp_rank,
            req_has_disagg_prefill_dp_rank,
        )
        self.conclude_state: Optional[KVPoll] = None
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
    ):
        kv_indices, index_slice, is_last_chunk, should_skip = (
            self._prepare_send_indices(kv_indices, state_indices)
        )
        if should_skip:
            return

        transfer_state_indices = (
            None
            if self.kv_mgr._should_skip_cp_replicated_state_transfer()
            else state_indices
        )
        normalized_state = (
            _normalize_state_indices_per_component(transfer_state_indices)
            if is_last_chunk
            else None
        )
        self._record_transfer_indices(kv_indices, transfer_state_indices)
        wait_event = getattr(self, "_early_send_wait_event", None)
        self._early_send_wait_event = None

        if not is_last_chunk:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
                num_kv_tokens=num_kv_tokens,
                wait_event=wait_event,
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
                state_indices=normalized_state,
                num_kv_tokens=num_kv_tokens,
                wait_event=wait_event,
            )

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        if self.bootstrap_room not in self.kv_mgr.request_status:
            self.conclude_state = KVPoll.Failed
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status == KVPoll.Bootstrapping:
            timeout_result = self._check_bootstrap_timeout()
            if timeout_result is not None:
                self.conclude_state = timeout_result
                return timeout_result
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        return status

    def clear(self) -> None:
        super().clear()
        with self.kv_mgr._room_notify_lock:
            self.kv_mgr._room_status_notified.pop(self.bootstrap_room, None)

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "KV transfer failed"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )


class MoriKVReceiver(CommonKVReceiver):

    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self.init_time: Optional[float] = None

    def init(
        self,
        prefill_dp_rank: int,
    ):
        super().init(prefill_dp_rank)
        if self.bootstrap_room is None:
            return
        if self.conclude_state == KVPoll.Failed:
            return
        if self.kv_mgr.enable_staging:
            self.require_staging = True
            if self.prefill_info.attn_tp_size != self.kv_mgr.attn_tp_size:
                raise RuntimeError(
                    "Mori staging intake currently requires matching prefill and "
                    f"decode attention TP sizes (prefill="
                    f"{self.prefill_info.attn_tp_size}, decode="
                        f"{self.kv_mgr.attn_tp_size})"
                )
            if self.prefill_info.pp_size != self.kv_mgr.pp_size:
                raise RuntimeError(
                    "Mori staging intake currently requires matching prefill and "
                    f"decode pipeline sizes (prefill="
                    f"{self.prefill_info.pp_size}, decode={self.kv_mgr.pp_size})"
                )
        self.kv_mgr.room_to_bootstrap_addr[self.bootstrap_room] = self.bootstrap_addr

    def _register_kv_args(self) -> bool:
        if self.bootstrap_infos is None:
            return False
        engine_desc_blob = self.kv_mgr.engine_desc.pack()
        packed_kv_descs = _pack_mem_desc_list(self.kv_mgr.kv_mem_descs)
        packed_aux_descs = _pack_mem_desc_list(self.kv_mgr.aux_mem_descs)
        packed_state_descs = _pack_mem_desc_lists(self.kv_mgr.state_mem_descs)
        gpu_id = str(self.kv_mgr.kv_args.gpu_id).encode("ascii")
        decode_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
        decode_tp_rank = str(self.kv_mgr.kv_args.engine_rank).encode("ascii")
        kv_item_len = str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii")
        packed_kv_item_lens = struct.pack(
            f"{len(self.kv_mgr.kv_args.kv_item_lens)}Q",
            *self.kv_mgr.kv_args.kv_item_lens,
        )
        packed_state_item_lens = pack_int_lists(
            self.kv_mgr.kv_args.state_item_lens, "I"
        )
        packed_state_slot_strides = pack_int_lists(
            self.kv_mgr.kv_args.state_slot_strides, "Q"
        )
        packed_state_mem_desc_offsets = pack_int_lists(
            self.kv_mgr.state_mem_desc_offsets, "Q"
        )
        packed_state_dim_per_tensor = pack_int_lists(
            self.kv_mgr.kv_args.state_dim_per_tensor, "I"
        )
        packed_staging_descs = (
            _pack_mem_desc_list([self.kv_mgr.staging_mem_desc])
            if self.kv_mgr.staging_mem_desc is not None
            else b""
        )
        packed_num_target_kv_entries = str(
            self.kv_mgr._num_target_kv_entries()
        ).encode("ascii")

        for bootstrap_info in self.bootstrap_infos:
            try:
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    sock.send_multipart(
                        [
                            MORI_GUARD,
                            "None".encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                            engine_desc_blob,
                            packed_kv_descs,
                            packed_aux_descs,
                            packed_state_descs,
                            gpu_id,
                            decode_tp_size,
                            decode_tp_rank,
                            kv_item_len,
                            packed_state_item_lens,
                            packed_state_dim_per_tensor,
                            packed_kv_item_lens,
                            packed_state_slot_strides,
                            packed_state_mem_desc_offsets,
                            packed_staging_descs,
                            packed_num_target_kv_entries,
                        ]
                    )
            except zmq.ZMQError:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"_register_kv_args to prefill {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return False
        return True

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if self.bootstrap_infos is None or self.bootstrap_room is None:
            return

        kv_indices_bytes = (
            np.asarray(kv_indices, dtype=np.int32).tobytes() if kv_indices.size else b""
        )
        aux_bytes = str(aux_index).encode("ascii") if aux_index is not None else b""
        normalized_state = _normalize_state_indices_per_component(state_indices)

        decode_prefix_bytes = (
            str(int(decode_prefix_len)).encode("ascii")
            if decode_prefix_len is not None and decode_prefix_len > 0
            else b""
        )
        if self.kv_mgr.enable_staging:
            self.chunk_staging_infos = []
            self.kv_mgr.register_staging_room_bootstrap(
                self.bootstrap_room,
                self.bootstrap_infos,
                self,
            )

        for bootstrap_info in self.bootstrap_infos:
            is_dummy = bootstrap_info.get("is_dummy", False)
            if not is_dummy and normalized_state is not None:
                state_bytes = _pack_state_indices(normalized_state)
            else:
                state_bytes = b""
            try:
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    sock.send_multipart(
                        [
                            MORI_GUARD,
                            str(self.bootstrap_room).encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                            self.kv_mgr.engine_desc.key.encode("ascii"),
                            kv_indices_bytes if not is_dummy else b"",
                            aux_bytes if not is_dummy else b"",
                            state_bytes,
                            str(self.required_dst_info_num).encode("ascii"),
                            decode_prefix_bytes,
                        ]
                    )
            except zmq.ZMQError:
                self.invalidate_cached_bootstrap_infos()
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"send_metadata to prefill {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status

        if status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result

        return status

    def clear(self) -> None:
        if self.bootstrap_room is None:
            return
        super().clear()
        self.kv_mgr._cleanup_room_tracking(self.bootstrap_room)

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "KV transfer failed"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )

    def abort(self):
        if self.bootstrap_room is None:
            return
        bootstrap_room = self.bootstrap_room
        super().abort()
        self.clear()
        with self.kv_mgr.failure_lock:
            self.kv_mgr.failure_records.pop(bootstrap_room, None)


class MoriKVBootstrapServer(CommonKVBootstrapServer):
    pass
