#!/usr/bin/env bash
set -euo pipefail

# PD Prefill-Decode disaggregation launch script for GLM-5.3 Flash FP8
# on gfx942 (MI325X) with upstream PR #36607 kpool/ROCm fixes.
#
# Config: TP4 prefill + TP4 decode, MoRI XGMI transfer,
#         CP-interleave on prefill, EAGLE/NEXTN spec decode (5 steps),
#         CUDA graph full on decode, INT4 QuickReduce on both roles.
#
# Usage: docker exec -e RUN_STAMP=<tag> \
#   -e PYTHONPATH=/tmp/sglang-src/python \
#   container bash /tmp/sglang-src/scripts/pd/launch_pd_glm53_flash_pr36607.sh
#
# Performance experiments are opt-in through role-specific environment
# variables below; defaults preserve the measured reference configuration.

: "${RUN_STAMP:?set RUN_STAMP}"
readonly S=${RUN_STAMP}
readonly SRC=/tmp/sglang-src
readonly OUT="/out/pd_glm53_pr36607_${S}"
readonly P_PORT=31100 D_PORT=31200 P_NCCL=31300 D_NCCL=31400 BOOT=31500 RTR=31000

readonly PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-4}"
readonly PREFILL_PP_SIZE="${PREFILL_PP_SIZE:-1}"
readonly DECODE_TP_SIZE="${DECODE_TP_SIZE:-4}"
readonly DECODE_PP_SIZE="${DECODE_PP_SIZE:-1}"
readonly PREFILL_GPU_IDS="${PREFILL_GPU_IDS:-0,1,2,3}"
readonly DECODE_GPU_IDS="${DECODE_GPU_IDS:-4,5,6,7}"
readonly PREFILL_PP_LAYER_PARTITION="${PREFILL_PP_LAYER_PARTITION:-}"
readonly DECODE_PP_LAYER_PARTITION="${DECODE_PP_LAYER_PARTITION:-}"
readonly PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-1}"
readonly DECODE_DP_SIZE="${DECODE_DP_SIZE:-1}"
readonly PREFILL_EP_SIZE="${PREFILL_EP_SIZE:-1}"
readonly DECODE_EP_SIZE="${DECODE_EP_SIZE:-1}"
readonly PREFILL_MOE_A2A_BACKEND="${PREFILL_MOE_A2A_BACKEND:-none}"
readonly DECODE_MOE_A2A_BACKEND="${DECODE_MOE_A2A_BACKEND:-none}"
readonly PREFILL_MOE_DISPATCH_DTYPE="${PREFILL_MOE_DISPATCH_DTYPE:-auto}"
readonly DECODE_MOE_DISPATCH_DTYPE="${DECODE_MOE_DISPATCH_DTYPE:-auto}"
readonly PREFILL_MORI_DISPATCH_DTYPE="${PREFILL_MORI_DISPATCH_DTYPE:-auto}"
readonly DECODE_MORI_DISPATCH_DTYPE="${DECODE_MORI_DISPATCH_DTYPE:-auto}"
readonly PREFILL_MORI_COMBINE_DTYPE="${PREFILL_MORI_COMBINE_DTYPE:-auto}"
readonly DECODE_MORI_COMBINE_DTYPE="${DECODE_MORI_COMBINE_DTYPE:-auto}"
readonly PREFILL_SHARED_EXPERTS_FUSION="${PREFILL_SHARED_EXPERTS_FUSION:-disable}"
readonly DECODE_SHARED_EXPERTS_FUSION="${DECODE_SHARED_EXPERTS_FUSION:-disable}"
readonly PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-8192}"
readonly PREFILL_CP_STRATEGY="${PREFILL_CP_STRATEGY:-interleave}"
readonly PREFILL_OVERLAP_SCHEDULE="${PREFILL_OVERLAP_SCHEDULE:-0}"
readonly DECODE_OVERLAP_SCHEDULE="${DECODE_OVERLAP_SCHEDULE:-0}"
readonly DISAGGREGATION_TRANSFER_BACKEND="${DISAGGREGATION_TRANSFER_BACKEND:-mori}"
readonly CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
readonly PREFILL_MEM_FRACTION_STATIC="${PREFILL_MEM_FRACTION_STATIC:-0.85}"
readonly DECODE_MEM_FRACTION_STATIC="${DECODE_MEM_FRACTION_STATIC:-0.80}"
readonly PREFILL_MAX_RUNNING_REQUESTS="${PREFILL_MAX_RUNNING_REQUESTS:-16}"
readonly DECODE_MAX_RUNNING_REQUESTS="${DECODE_MAX_RUNNING_REQUESTS:-120}"
readonly PREFILL_MAMBA_FULL_MEMORY_RATIO="${PREFILL_MAMBA_FULL_MEMORY_RATIO:-0.9}"
readonly DECODE_MAMBA_FULL_MEMORY_RATIO="${DECODE_MAMBA_FULL_MEMORY_RATIO:-0.9}"
readonly PREFILL_MAX_MAMBA_CACHE_SIZE="${PREFILL_MAX_MAMBA_CACHE_SIZE:-}"
readonly DECODE_MAX_MAMBA_CACHE_SIZE="${DECODE_MAX_MAMBA_CACHE_SIZE:-}"
readonly DECODE_CUDA_GRAPH_BACKEND="${DECODE_CUDA_GRAPH_BACKEND:-full}"
readonly DECODE_CUDA_GRAPH_MAX_BS="${DECODE_CUDA_GRAPH_MAX_BS:-64}"
readonly ENABLE_DECODE_RADIX_CACHE="${ENABLE_DECODE_RADIX_CACHE:-1}"
readonly JSON_MODEL_OVERRIDE_ARGS="${JSON_MODEL_OVERRIDE_ARGS:-}"
readonly SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-5}"
readonly SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-6}"
readonly ENABLE_SPECULATIVE_DECODING="${ENABLE_SPECULATIVE_DECODING:-1}"
readonly ENABLE_SPECULATIVE_ADAPTIVE="${ENABLE_SPECULATIVE_ADAPTIVE:-0}"
readonly PREFILL_AITER_ALLREDUCE_FUSION="${PREFILL_AITER_ALLREDUCE_FUSION:-1}"
readonly DECODE_AITER_ALLREDUCE_FUSION="${DECODE_AITER_ALLREDUCE_FUSION:-1}"
readonly PREFILL_QUICK_REDUCE_QUANTIZATION="${PREFILL_QUICK_REDUCE_QUANTIZATION:-INT4}"
readonly DECODE_QUICK_REDUCE_QUANTIZATION="${DECODE_QUICK_REDUCE_QUANTIZATION:-INT4}"
readonly MORI_QP_PER_TRANSFER="${MORI_QP_PER_TRANSFER:-${SGLANG_MORI_QP_PER_TRANSFER:-8}}"
readonly MORI_TRANSFER_SHARDS="${MORI_TRANSFER_SHARDS:-${SGLANG_MORI_TRANSFER_SHARDS:-24}}"
readonly PREFILL_ROCPROF="${PREFILL_ROCPROF:-0}"
readonly DECODE_ROCPROF="${DECODE_ROCPROF:-0}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

validate_gpu_ids() {
  local role=$1 ids=$2 output_var=$3
  if [[ ! "${ids}" =~ ^[0-7](,[0-7])*$ ]]; then
    echo "${role}_GPU_IDS must be a comma-separated subset of GPU IDs 0-7; got ${ids}" >&2; exit 2
  fi
  local -a parsed_ids=(); local -A seen_ids=(); local id
  IFS=, read -r -a parsed_ids <<< "${ids}"
  for id in "${parsed_ids[@]}"; do
    if [[ -n "${seen_ids[${id}]:-}" ]]; then
      echo "${role}_GPU_IDS contains duplicate GPU ID ${id}: ${ids}" >&2; exit 2
    fi
    seen_ids[${id}]=1
  done
  printf -v "${output_var}" '%d' "${#parsed_ids[@]}"
}

PREFILL_GPU_COUNT=0; DECODE_GPU_COUNT=0
validate_gpu_ids PREFILL "${PREFILL_GPU_IDS}" PREFILL_GPU_COUNT
validate_gpu_ids DECODE "${DECODE_GPU_IDS}" DECODE_GPU_COUNT
readonly PREFILL_GPU_COUNT DECODE_GPU_COUNT
readonly PREFILL_LOGICAL_GPU_IDS="$(seq -s, 0 $((PREFILL_GPU_COUNT - 1)))"
readonly DECODE_LOGICAL_GPU_IDS="$(seq -s, 0 $((DECODE_GPU_COUNT - 1)))"

IFS=, read -r -a _pg_ids <<< "${PREFILL_GPU_IDS}"
IFS=, read -r -a _dg_ids <<< "${DECODE_GPU_IDS}"
for _pg in "${_pg_ids[@]}"; do
  for _dg in "${_dg_ids[@]}"; do
    if [[ "${_pg}" == "${_dg}" ]]; then
      echo "PREFILL_GPU_IDS and DECODE_GPU_IDS overlap on GPU ${_pg}" >&2; exit 2
    fi
  done
done
unset _pg_ids _dg_ids _pg _dg

validate_role_parallelism() {
  local role=$1 tp=$2 pp=$3 count=$4
  if ((tp < 1 || pp < 1 || tp * pp != count)); then
    echo "${role}_TP_SIZE * ${role}_PP_SIZE must equal ${count} GPUs; got ${tp} * ${pp}" >&2; exit 2
  fi
}
validate_role_parallelism PREFILL "${PREFILL_TP_SIZE}" "${PREFILL_PP_SIZE}" "${PREFILL_GPU_COUNT}"
validate_role_parallelism DECODE "${DECODE_TP_SIZE}" "${DECODE_PP_SIZE}" "${DECODE_GPU_COUNT}"

readonly PREFILL_READY_PATH="$([[ ${PREFILL_PP_SIZE} -gt 1 ]] && echo model_info || echo health)"
readonly DECODE_READY_PATH="$([[ ${DECODE_PP_SIZE} -gt 1 ]] && echo model_info || echo health)"

if [[ "${PREFILL_CP_STRATEGY}" != "none" && "${PREFILL_CP_STRATEGY}" != "interleave" ]]; then
  echo "PREFILL_CP_STRATEGY must be none or interleave; got ${PREFILL_CP_STRATEGY}" >&2; exit 2
fi
if [[ "${PREFILL_CP_STRATEGY}" == "interleave" ]] && ((PREFILL_DP_SIZE != 1)); then
  echo "PREFILL_CP_STRATEGY=interleave requires PREFILL_DP_SIZE=1" >&2; exit 2
fi

case "${DISAGGREGATION_TRANSFER_BACKEND}" in
  mori|mooncake|mooncake_tcp) ;;
  *) echo "DISAGGREGATION_TRANSFER_BACKEND must be mori, mooncake, or mooncake_tcp; got ${DISAGGREGATION_TRANSFER_BACKEND}" >&2; exit 2 ;;
esac

if [[ ! "${SPECULATIVE_NUM_DRAFT_TOKENS}" =~ ^[0-9]+$ ]] || ((SPECULATIVE_NUM_DRAFT_TOKENS != SPECULATIVE_NUM_STEPS + 1)); then
  echo "SPECULATIVE_NUM_DRAFT_TOKENS must equal SPECULATIVE_NUM_STEPS + 1; got steps=${SPECULATIVE_NUM_STEPS}, draft_tokens=${SPECULATIVE_NUM_DRAFT_TOKENS}" >&2; exit 2
fi

# ---------------------------------------------------------------------------
# Speculative decoding args
# ---------------------------------------------------------------------------

declare -a SPECULATIVE_ARGS=()
if [[ "${ENABLE_SPECULATIVE_DECODING}" == "1" ]]; then
  SPECULATIVE_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
    --speculative-attention-mode prefill
    --speculative-accept-threshold-single 1.0
    --speculative-accept-threshold-acc 1.0
    --speculative-draft-model-quantization unquant
  )
fi

if [[ "${ENABLE_SPECULATIVE_ADAPTIVE}" == "1" ]]; then
  SPECULATIVE_ARGS+=(--speculative-adaptive)
elif [[ "${ENABLE_SPECULATIVE_ADAPTIVE}" != "0" ]]; then
  echo "ENABLE_SPECULATIVE_ADAPTIVE must be 0 or 1; got ${ENABLE_SPECULATIVE_ADAPTIVE}" >&2
  exit 2
fi

for _role_profiler in PREFILL_ROCPROF DECODE_ROCPROF; do
  if [[ "${!_role_profiler}" != "0" && "${!_role_profiler}" != "1" ]]; then
    echo "${_role_profiler} must be 0 or 1; got ${!_role_profiler}" >&2
    exit 2
  fi
done
unset _role_profiler

for _mori_setting in MORI_QP_PER_TRANSFER MORI_TRANSFER_SHARDS; do
  if [[ ! "${!_mori_setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${_mori_setting} must be a positive integer; got ${!_mori_setting}" >&2
    exit 2
  fi
done
unset _mori_setting

for _request_limit in PREFILL_MAX_RUNNING_REQUESTS DECODE_MAX_RUNNING_REQUESTS; do
  if [[ ! "${!_request_limit}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${_request_limit} must be a positive integer; got ${!_request_limit}" >&2
    exit 2
  fi
done
unset _request_limit

for _mamba_limit in PREFILL_MAX_MAMBA_CACHE_SIZE DECODE_MAX_MAMBA_CACHE_SIZE; do
  if [[ -n "${!_mamba_limit}" && ! "${!_mamba_limit}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${_mamba_limit} must be empty or a positive integer; got ${!_mamba_limit}" >&2
    exit 2
  fi
done
unset _mamba_limit

declare -a MODEL_OVERRIDE_ARGS=()
if [[ -n "${JSON_MODEL_OVERRIDE_ARGS}" ]]; then
  MODEL_OVERRIDE_ARGS=(--json-model-override-args "${JSON_MODEL_OVERRIDE_ARGS}")
fi

declare -a DECODE_RADIX_ARGS=()
if [[ "${ENABLE_DECODE_RADIX_CACHE}" == "1" ]]; then
  DECODE_RADIX_ARGS=(--disaggregation-decode-enable-radix-cache)
elif [[ "${ENABLE_DECODE_RADIX_CACHE}" != "0" ]]; then
  echo "ENABLE_DECODE_RADIX_CACHE must be 0 or 1; got ${ENABLE_DECODE_RADIX_CACHE}" >&2
  exit 2
fi

declare -a PREFILL_MAMBA_ARGS=(--mamba-full-memory-ratio "${PREFILL_MAMBA_FULL_MEMORY_RATIO}")
if [[ -n "${PREFILL_MAX_MAMBA_CACHE_SIZE}" ]]; then
  PREFILL_MAMBA_ARGS+=(--max-mamba-cache-size "${PREFILL_MAX_MAMBA_CACHE_SIZE}")
fi
declare -a DECODE_MAMBA_ARGS=(--mamba-full-memory-ratio "${DECODE_MAMBA_FULL_MEMORY_RATIO}")
if [[ -n "${DECODE_MAX_MAMBA_CACHE_SIZE}" ]]; then
  DECODE_MAMBA_ARGS+=(--max-mamba-cache-size "${DECODE_MAX_MAMBA_CACHE_SIZE}")
fi

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------

readonly PREFILL_CHUNKED_PREFILL_SIZE=$((PREFILL_CHUNK_TOKENS * PREFILL_DP_SIZE))
readonly DECODE_CHUNKED_PREFILL_SIZE=$((65536 * DECODE_DP_SIZE))
readonly PREFILL_MORI_MAX_DISPATCH_TOKENS="${PREFILL_MORI_MAX_DISPATCH_TOKENS:-${PREFILL_CHUNKED_PREFILL_SIZE}}"

declare -a PREFILL_SHARED_EXPERTS_ARGS=()
case "${PREFILL_SHARED_EXPERTS_FUSION}" in
  disable) PREFILL_SHARED_EXPERTS_ARGS=(--disable-shared-experts-fusion) ;;
  enforce) PREFILL_SHARED_EXPERTS_ARGS=(--enforce-shared-experts-fusion) ;;
esac
declare -a DECODE_SHARED_EXPERTS_ARGS=()
case "${DECODE_SHARED_EXPERTS_FUSION}" in
  disable) DECODE_SHARED_EXPERTS_ARGS=(--disable-shared-experts-fusion) ;;
  enforce) DECODE_SHARED_EXPERTS_ARGS=(--enforce-shared-experts-fusion) ;;
esac

declare -a PREFILL_CP_ARGS=()
declare -a PREFILL_ALLREDUCE_ARGS=(--enable-aiter-allreduce-fusion)
if [[ "${PREFILL_CP_STRATEGY}" == "interleave" ]]; then
  PREFILL_CP_ARGS=(--enable-prefill-cp --cp-strategy interleave)
  PREFILL_ALLREDUCE_ARGS=()
fi

declare -a DECODE_ALLREDUCE_ARGS=(--enable-aiter-allreduce-fusion)
if ((DECODE_EP_SIZE > 1)); then DECODE_ALLREDUCE_ARGS=(); fi

declare -a PREFILL_OVERLAP_ARGS=(--disable-overlap-schedule)
if [[ "${PREFILL_OVERLAP_SCHEDULE}" == "1" ]]; then PREFILL_OVERLAP_ARGS=(); fi
declare -a DECODE_OVERLAP_ARGS=(--disable-overlap-schedule)
if [[ "${DECODE_OVERLAP_SCHEDULE}" == "1" ]]; then DECODE_OVERLAP_ARGS=(); fi

declare -a PREFILL_DP_ARGS=()
if ((PREFILL_DP_SIZE > 1)); then
  PREFILL_DP_ARGS=(--dp-size "${PREFILL_DP_SIZE}" --enable-dp-attention --enable-metrics-for-all-schedulers)
fi
declare -a DECODE_DP_ARGS=()
if ((DECODE_DP_SIZE > 1)); then
  DECODE_DP_ARGS=(--dp-size "${DECODE_DP_SIZE}" --enable-dp-attention --enable-metrics-for-all-schedulers)
fi

declare -a PREFILL_MOE_ARGS=(--moe-a2a-backend none --deepep-dispatcher-output-dtype "${PREFILL_MOE_DISPATCH_DTYPE}")
if ((PREFILL_EP_SIZE > 1)); then
  PREFILL_MOE_ARGS=(--ep-size "${PREFILL_EP_SIZE}" --moe-a2a-backend "${PREFILL_MOE_A2A_BACKEND}" --deepep-dispatcher-output-dtype "${PREFILL_MOE_DISPATCH_DTYPE}")
fi
declare -a DECODE_MOE_ARGS=(--moe-a2a-backend none --deepep-dispatcher-output-dtype "${DECODE_MOE_DISPATCH_DTYPE}")
if ((DECODE_EP_SIZE > 1)); then
  DECODE_MOE_ARGS=(--ep-size "${DECODE_EP_SIZE}" --moe-a2a-backend "${DECODE_MOE_A2A_BACKEND}" --deepep-dispatcher-output-dtype "${DECODE_MOE_DISPATCH_DTYPE}")
fi

declare -a ROUTER_SCHEDULING_ARGS=(--mini-lb)
if ((PREFILL_DP_SIZE > 1 || DECODE_DP_SIZE > 1)); then
  ROUTER_SCHEDULING_ARGS=(--dp-aware --policy manual --prefill-policy manual --decode-policy manual --assignment-mode min_load)
fi

mkdir -p "${OUT}" "${OUT}/profiles/prefill" "${OUT}/profiles/decode"

declare -a PREFILL_LAUNCH_PREFIX=()
if [[ "${PREFILL_ROCPROF}" == "1" ]]; then
  mkdir -p "${OUT}/profiles/rocprof/prefill"
  PREFILL_LAUNCH_PREFIX=(
    /opt/rocm/bin/rocprofv3
    --kernel-trace
    --rccl-trace
    --output-format csv
    --output-directory "${OUT}/profiles/rocprof/prefill"
    --output-file 'prefill-%pid%'
    --
  )
fi

declare -a DECODE_LAUNCH_PREFIX=()
if [[ "${DECODE_ROCPROF}" == "1" ]]; then
  mkdir -p "${OUT}/profiles/rocprof/decode"
  DECODE_LAUNCH_PREFIX=(
    /opt/rocm/bin/rocprofv3
    --kernel-trace
    --rccl-trace
    --output-format csv
    --output-directory "${OUT}/profiles/rocprof/decode"
    --output-file 'decode-%pid%'
    --
  )
fi

# ---------------------------------------------------------------------------
# Log configuration
# ---------------------------------------------------------------------------

echo "[$(date -u +%FT%TZ)] GLM-5.3 Flash FP8 PD -- stamp=${S}"
echo "[$(date -u +%FT%TZ)] Prefill: TP=${PREFILL_TP_SIZE} PP=${PREFILL_PP_SIZE} DP=${PREFILL_DP_SIZE} EP=${PREFILL_EP_SIZE} GPUs=${PREFILL_GPU_IDS}"
echo "[$(date -u +%FT%TZ)] Decode:  TP=${DECODE_TP_SIZE} PP=${DECODE_PP_SIZE} DP=${DECODE_DP_SIZE} EP=${DECODE_EP_SIZE} GPUs=${DECODE_GPU_IDS}"
echo "[$(date -u +%FT%TZ)] Transfer: ${DISAGGREGATION_TRANSFER_BACKEND} (XGMI enabled), CP=${PREFILL_CP_STRATEGY}"
echo "[$(date -u +%FT%TZ)] Spec decode: enabled=${ENABLE_SPECULATIVE_DECODING} adaptive=${ENABLE_SPECULATIVE_ADAPTIVE} steps=${SPECULATIVE_NUM_STEPS} draft=${SPECULATIVE_NUM_DRAFT_TOKENS}"
echo "[$(date -u +%FT%TZ)] CUDA graph decode: ${DECODE_CUDA_GRAPH_BACKEND} max_bs=${DECODE_CUDA_GRAPH_MAX_BS}"
echo "[$(date -u +%FT%TZ)] Decode radix cache: ${ENABLE_DECODE_RADIX_CACHE}"
echo "[$(date -u +%FT%TZ)] Model overrides: ${JSON_MODEL_OVERRIDE_ARGS:-none}"
echo "[$(date -u +%FT%TZ)] Mem fraction: prefill=${PREFILL_MEM_FRACTION_STATIC} decode=${DECODE_MEM_FRACTION_STATIC}"
echo "[$(date -u +%FT%TZ)] Running requests: prefill=${PREFILL_MAX_RUNNING_REQUESTS} decode=${DECODE_MAX_RUNNING_REQUESTS}"
echo "[$(date -u +%FT%TZ)] KDA/Mamba: prefill_ratio=${PREFILL_MAMBA_FULL_MEMORY_RATIO} prefill_cap=${PREFILL_MAX_MAMBA_CACHE_SIZE:-auto} decode_ratio=${DECODE_MAMBA_FULL_MEMORY_RATIO} decode_cap=${DECODE_MAX_MAMBA_CACHE_SIZE:-auto}"
echo "[$(date -u +%FT%TZ)] QuickReduce: prefill=${PREFILL_QUICK_REDUCE_QUANTIZATION} decode=${DECODE_QUICK_REDUCE_QUANTIZATION}"
echo "[$(date -u +%FT%TZ)] Shared experts fusion: prefill=${PREFILL_SHARED_EXPERTS_FUSION} decode=${DECODE_SHARED_EXPERTS_FUSION}"
echo "[$(date -u +%FT%TZ)] MoRI: qp_per_transfer=${MORI_QP_PER_TRANSFER} transfer_shards=${MORI_TRANSFER_SHARDS}"
echo "[$(date -u +%FT%TZ)] rocprof: prefill=${PREFILL_ROCPROF} decode=${DECODE_ROCPROF}"

# ---------------------------------------------------------------------------
# Prefill server
# ---------------------------------------------------------------------------

env -u MC_FORCE_TCP -u MOONCAKE_PROTOCOL -u SGLANG_PP_LAYER_PARTITION \
  PYTHONPATH="${SRC}/python" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${PREFILL_LOGICAL_GPU_IDS}" \
  SGLANG_TORCH_PROFILER_DIR="${OUT}/profiles/prefill" \
  SGLANG_USE_AITER=1 \
  ROCM_QUICK_REDUCE_QUANTIZATION="${PREFILL_QUICK_REDUCE_QUANTIZATION}" \
  ROCR_VISIBLE_DEVICES="${PREFILL_GPU_IDS}" HIP_VISIBLE_DEVICES="${PREFILL_LOGICAL_GPU_IDS}" \
  SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 SGLANG_OPT_USE_FLYDSL_FP8_MQA_LOGITS=1 \
  SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${PREFILL_MORI_MAX_DISPATCH_TOKENS}" \
  SGLANG_MORI_DISPATCH_DTYPE="${PREFILL_MORI_DISPATCH_DTYPE}" \
  SGLANG_MORI_COMBINE_DTYPE="${PREFILL_MORI_COMBINE_DTYPE}" \
  SGLANG_MORI_QP_PER_TRANSFER="${MORI_QP_PER_TRANSFER}" \
  SGLANG_MORI_TRANSFER_SHARDS="${MORI_TRANSFER_SHARDS}" \
  MORI_DISABLE_AUTO_XGMI=0 MC_TE_METRIC=1 \
  "${PREFILL_LAUNCH_PREFIX[@]}" \
  python3 -m sglang.launch_server \
    --model-path /model --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port ${P_PORT} --base-gpu-id 0 \
    --tp-size ${PREFILL_TP_SIZE} --pp-size ${PREFILL_PP_SIZE} \
    "${PREFILL_DP_ARGS[@]}" "${PREFILL_CP_ARGS[@]}" \
    "${PREFILL_MOE_ARGS[@]}" "${PREFILL_SHARED_EXPERTS_ARGS[@]}" \
    --quantization fp8 --trust-remote-code --kv-cache-dtype fp8_e4m3 \
    "${MODEL_OVERRIDE_ARGS[@]}" \
    --attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --linear-attn-backend triton --moe-runner-backend aiter \
    "${PREFILL_ALLREDUCE_ARGS[@]}" \
    --mem-fraction-static ${PREFILL_MEM_FRACTION_STATIC} --max-running-requests "${PREFILL_MAX_RUNNING_REQUESTS}" \
    "${PREFILL_MAMBA_ARGS[@]}" \
    --schedule-policy lpm --context-length "${CONTEXT_LENGTH}" --page-size 64 \
    --chunked-prefill-size ${PREFILL_CHUNKED_PREFILL_SIZE} --max-prefill-tokens 16384 \
    --prefill-max-requests 16 "${PREFILL_OVERLAP_ARGS[@]}" \
    --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
    "${SPECULATIVE_ARGS[@]}" \
    --enable-session-radix-cache \
    --disaggregation-mode prefill --disaggregation-transfer-backend ${DISAGGREGATION_TRANSFER_BACKEND} \
    --disaggregation-bootstrap-port ${BOOT} --nccl-port ${P_NCCL} \
    --enable-metrics --enable-cache-report --enable-request-time-stats-logging \
    --random-seed 0 --watchdog-timeout 3600 \
    > "${OUT}/prefill.log" 2>&1 &
P_PID=$!
echo "${P_PID}" > "${OUT}/prefill.pid"
echo "[$(date -u +%FT%TZ)] Prefill PID=${P_PID}"
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${P_PORT}/${PREFILL_READY_PATH}" >/dev/null 2>&1 && break
  kill -0 ${P_PID} 2>/dev/null || { echo PREFILL_DIED; tail -30 "${OUT}/prefill.log"; exit 1; }
  sleep 10
done
echo "[$(date -u +%FT%TZ)] Prefill healthy"

# ---------------------------------------------------------------------------
# Decode server
# ---------------------------------------------------------------------------

env -u MC_FORCE_TCP -u MOONCAKE_PROTOCOL -u SGLANG_PP_LAYER_PARTITION \
  PYTHONPATH="${SRC}/python" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${DECODE_LOGICAL_GPU_IDS}" \
  SGLANG_TORCH_PROFILER_DIR="${OUT}/profiles/decode" \
  SGLANG_USE_AITER=1 \
  ROCM_QUICK_REDUCE_QUANTIZATION="${DECODE_QUICK_REDUCE_QUANTIZATION}" \
  ROCR_VISIBLE_DEVICES="${DECODE_GPU_IDS}" HIP_VISIBLE_DEVICES="${DECODE_LOGICAL_GPU_IDS}" \
  SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
  SGLANG_MORI_DISPATCH_DTYPE="${DECODE_MORI_DISPATCH_DTYPE}" \
  SGLANG_MORI_COMBINE_DTYPE="${DECODE_MORI_COMBINE_DTYPE}" \
  SGLANG_MORI_QP_PER_TRANSFER="${MORI_QP_PER_TRANSFER}" \
  SGLANG_MORI_TRANSFER_SHARDS="${MORI_TRANSFER_SHARDS}" \
  MORI_DISABLE_AUTO_XGMI=0 MC_TE_METRIC=1 \
  "${DECODE_LAUNCH_PREFIX[@]}" \
  python3 -m sglang.launch_server \
    --model-path /model --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port ${D_PORT} --base-gpu-id 0 \
    --tp-size ${DECODE_TP_SIZE} --pp-size ${DECODE_PP_SIZE} \
    "${DECODE_DP_ARGS[@]}" \
    "${DECODE_MOE_ARGS[@]}" "${DECODE_SHARED_EXPERTS_ARGS[@]}" \
    --quantization fp8 --trust-remote-code --kv-cache-dtype fp8_e4m3 \
    "${MODEL_OVERRIDE_ARGS[@]}" \
    --attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --linear-attn-backend triton --moe-runner-backend aiter \
    "${DECODE_ALLREDUCE_ARGS[@]}" \
    --mem-fraction-static ${DECODE_MEM_FRACTION_STATIC} --max-running-requests "${DECODE_MAX_RUNNING_REQUESTS}" \
    "${DECODE_MAMBA_ARGS[@]}" \
    --context-length "${CONTEXT_LENGTH}" --page-size 64 \
    --chunked-prefill-size ${DECODE_CHUNKED_PREFILL_SIZE} --max-prefill-tokens 65536 \
    "${DECODE_OVERLAP_ARGS[@]}" \
    --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode "${DECODE_CUDA_GRAPH_BACKEND}" \
    --cuda-graph-max-bs-decode "${DECODE_CUDA_GRAPH_MAX_BS}" \
    "${SPECULATIVE_ARGS[@]}" \
    --num-reserved-decode-tokens 1024 \
    --disaggregation-mode decode --disaggregation-transfer-backend ${DISAGGREGATION_TRANSFER_BACKEND} \
    "${DECODE_RADIX_ARGS[@]}" \
    --disaggregation-bootstrap-port ${BOOT} --nccl-port ${D_NCCL} \
    --enable-metrics --enable-cache-report --enable-request-time-stats-logging \
    --random-seed 0 --watchdog-timeout 3600 \
    > "${OUT}/decode.log" 2>&1 &
D_PID=$!
echo "${D_PID}" > "${OUT}/decode.pid"
echo "[$(date -u +%FT%TZ)] Decode PID=${D_PID}"
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${D_PORT}/${DECODE_READY_PATH}" >/dev/null 2>&1 && break
  kill -0 ${D_PID} 2>/dev/null || { echo DECODE_DIED; tail -30 "${OUT}/decode.log"; exit 1; }
  sleep 10
done
echo "[$(date -u +%FT%TZ)] Decode healthy"

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PYTHONPATH="${SRC}/python" python3 -m sglang_router.launch_router \
  --host 0.0.0.0 --port ${RTR} \
  --pd-disaggregation "${ROUTER_SCHEDULING_ARGS[@]}" \
  --prefill http://127.0.0.1:${P_PORT} ${BOOT} \
  --decode http://127.0.0.1:${D_PORT} \
  > "${OUT}/router.log" 2>&1 &
R_PID=$!
echo "${R_PID}" > "${OUT}/router.pid"
echo "[$(date -u +%FT%TZ)] Router PID=${R_PID}"
sleep 5
curl -sf "http://127.0.0.1:${RTR}/health" >/dev/null 2>&1 || { echo ROUTER_NOT_HEALTHY; tail -30 "${OUT}/router.log"; exit 1; }
echo "[$(date -u +%FT%TZ)] Router healthy"
echo ALL_ROLES_READY
wait $P_PID $D_PID $R_PID
