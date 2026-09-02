#!/usr/bin/env bash
set -euo pipefail

# PD Prefill-Decode disaggregation launch script for GLM-5.3 Flash FP8
# on gfx942 (MI325X) with upstream PR #36607 kpool/ROCm fixes.
#
# Config: TP4 prefill + TP4 decode, MoRI XGMI transfer,
#         CP-interleave on prefill, matched EAGLE draft-state prefill,
#         EAGLE speculative token decode (5 steps) on the decode role,
#         CUDA graph full on decode, INT4 QuickReduce on both roles.
#
# Usage: docker exec -e RUN_STAMP=<tag> \
#   -e PYTHONPATH=/tmp/sglang-src/python \
#   container bash /tmp/sglang-src/scripts/pd/launch_pd_glm53_flash_pr36607.sh
# Set SGLANG_SOURCE_ROOT when validating an isolated staged source tree.
#
# Performance experiments are opt-in through role-specific environment
# variables below; defaults preserve the measured reference configuration.

: "${RUN_STAMP:?set RUN_STAMP}"
readonly S=${RUN_STAMP}
readonly SRC="${SGLANG_SOURCE_ROOT:-/tmp/sglang-src}"
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
readonly PREFILL_REQUEST_CHUNK_TOKENS="${PREFILL_REQUEST_CHUNK_TOKENS:-}"
readonly PREFILL_REQUEST_CHUNK_QUEUE_THRESHOLD="${PREFILL_REQUEST_CHUNK_QUEUE_THRESHOLD:-}"
readonly PREFILL_CP_STRATEGY="${PREFILL_CP_STRATEGY:-interleave}"
readonly PREFILL_OVERLAP_SCHEDULE="${PREFILL_OVERLAP_SCHEDULE:-0}"
readonly DECODE_OVERLAP_SCHEDULE="${DECODE_OVERLAP_SCHEDULE:-0}"
readonly DISAGGREGATION_TRANSFER_BACKEND="${DISAGGREGATION_TRANSFER_BACKEND:-mori}"
readonly CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
readonly PREFILL_MEM_FRACTION_STATIC="${PREFILL_MEM_FRACTION_STATIC:-0.85}"
readonly DECODE_MEM_FRACTION_STATIC="${DECODE_MEM_FRACTION_STATIC:-0.80}"
readonly PREFILL_MAX_RUNNING_REQUESTS="${PREFILL_MAX_RUNNING_REQUESTS:-16}"
readonly PREFILL_MAX_INFLIGHT_TRANSFERS="${PREFILL_MAX_INFLIGHT_TRANSFERS:-0}"
readonly DECODE_MAX_RUNNING_REQUESTS="${DECODE_MAX_RUNNING_REQUESTS:-120}"
readonly PREFILL_RADIX_EVICTION_POLICY="${PREFILL_RADIX_EVICTION_POLICY:-lru}"
readonly DECODE_RADIX_EVICTION_POLICY="${DECODE_RADIX_EVICTION_POLICY:-lru}"
readonly ENABLE_PREFILL_HICACHE="${ENABLE_PREFILL_HICACHE:-0}"
readonly PREFILL_HICACHE_SIZE_GB="${PREFILL_HICACHE_SIZE_GB:-32}"
readonly PREFILL_HICACHE_WRITE_POLICY="${PREFILL_HICACHE_WRITE_POLICY:-write_through}"
readonly PREFILL_HICACHE_IO_BACKEND="${PREFILL_HICACHE_IO_BACKEND:-kernel}"
readonly PREFILL_HICACHE_MEM_LAYOUT="${PREFILL_HICACHE_MEM_LAYOUT:-page_first}"
readonly PREFILL_MAMBA_FULL_MEMORY_RATIO="${PREFILL_MAMBA_FULL_MEMORY_RATIO:-0.9}"
readonly DECODE_MAMBA_FULL_MEMORY_RATIO="${DECODE_MAMBA_FULL_MEMORY_RATIO:-0.9}"
readonly PREFILL_MAX_MAMBA_CACHE_SIZE="${PREFILL_MAX_MAMBA_CACHE_SIZE:-}"
readonly DECODE_MAX_MAMBA_CACHE_SIZE="${DECODE_MAX_MAMBA_CACHE_SIZE:-}"
readonly PREFILL_MAMBA_SSM_DTYPE="${PREFILL_MAMBA_SSM_DTYPE:-}"
readonly DECODE_MAMBA_SSM_DTYPE="${DECODE_MAMBA_SSM_DTYPE:-}"
readonly DECODE_MAMBA_SKIP_DECODE_LOCK="${DECODE_MAMBA_SKIP_DECODE_LOCK:-0}"
readonly DECODE_CUDA_GRAPH_BACKEND="${DECODE_CUDA_GRAPH_BACKEND:-full}"
readonly DECODE_CUDA_GRAPH_MAX_BS="${DECODE_CUDA_GRAPH_MAX_BS:-64}"
readonly ENABLE_DECODE_RADIX_CACHE="${ENABLE_DECODE_RADIX_CACHE:-1}"
readonly REASONING_PARSER="${REASONING_PARSER:-}"
readonly TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
readonly TOOL_STRICT_LEVEL="${TOOL_STRICT_LEVEL:-0}"
readonly ENABLE_STRICT_THINKING="${ENABLE_STRICT_THINKING:-0}"
readonly ENABLE_REQUEST_LOGGING="${ENABLE_REQUEST_LOGGING:-0}"
readonly REQUEST_LOG_LEVEL="${REQUEST_LOG_LEVEL:-2}"
readonly JSON_MODEL_OVERRIDE_ARGS="${JSON_MODEL_OVERRIDE_ARGS:-}"
readonly SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-EAGLE}"
readonly SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-5}"
readonly SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-6}"
readonly SPECULATIVE_ADAPTIVE_CONFIG="${SPECULATIVE_ADAPTIVE_CONFIG:-}"
readonly SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-}"
readonly SPECULATIVE_DRAFT_ATTENTION_BACKEND="${SPECULATIVE_DRAFT_ATTENTION_BACKEND:-}"
readonly SPECULATIVE_DRAFT_WINDOW_SIZE="${SPECULATIVE_DRAFT_WINDOW_SIZE:-}"
readonly SPECULATIVE_DFLASH_BLOCK_SIZE="${SPECULATIVE_DFLASH_BLOCK_SIZE:-}"
readonly ENABLE_SPECULATIVE_DECODING="${ENABLE_SPECULATIVE_DECODING:-1}"
readonly ENABLE_SPECULATIVE_ADAPTIVE="${ENABLE_SPECULATIVE_ADAPTIVE:-1}"
readonly ENABLE_PREFILL_DRAFT_STATE="${ENABLE_PREFILL_DRAFT_STATE:-1}"
readonly PREFILL_AITER_ALLREDUCE_FUSION="${PREFILL_AITER_ALLREDUCE_FUSION:-1}"
readonly DECODE_AITER_ALLREDUCE_FUSION="${DECODE_AITER_ALLREDUCE_FUSION:-1}"
readonly PREFILL_AITER_CONFIG_FMOE="${PREFILL_AITER_CONFIG_FMOE:-}"
readonly DECODE_AITER_CONFIG_FMOE="${DECODE_AITER_CONFIG_FMOE:-}"
readonly PREFILL_FLYDSL_FP8_MQA_LOGITS="${PREFILL_FLYDSL_FP8_MQA_LOGITS:-1}"
readonly PREFILL_DSA_KPOOL_CACHE_TRIM_THRESHOLD="${PREFILL_DSA_KPOOL_CACHE_TRIM_THRESHOLD:-0}"
readonly PREFILL_DSA_REUSE_RAGGED_LOGITS="${PREFILL_DSA_REUSE_RAGGED_LOGITS:-0}"
readonly PREFILL_DSA_SEGMENT_RAGGED_LOGITS="${PREFILL_DSA_SEGMENT_RAGGED_LOGITS:-0}"
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

case "${SPECULATIVE_ALGORITHM}" in
  EAGLE|DFLASH) ;;
  *) echo "SPECULATIVE_ALGORITHM must be EAGLE or DFLASH; got ${SPECULATIVE_ALGORITHM}" >&2; exit 2 ;;
esac

if [[ "${SPECULATIVE_ALGORITHM}" == "EAGLE" ]]; then
  if [[ ! "${SPECULATIVE_NUM_DRAFT_TOKENS}" =~ ^[0-9]+$ ]] || ((SPECULATIVE_NUM_DRAFT_TOKENS != SPECULATIVE_NUM_STEPS + 1)); then
    echo "SPECULATIVE_NUM_DRAFT_TOKENS must equal SPECULATIVE_NUM_STEPS + 1 for EAGLE; got steps=${SPECULATIVE_NUM_STEPS}, draft_tokens=${SPECULATIVE_NUM_DRAFT_TOKENS}" >&2
    exit 2
  fi
elif [[ -n "${SPECULATIVE_DFLASH_BLOCK_SIZE}" && ! "${SPECULATIVE_DFLASH_BLOCK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SPECULATIVE_DFLASH_BLOCK_SIZE must be empty or a positive integer; got ${SPECULATIVE_DFLASH_BLOCK_SIZE}" >&2
  exit 2
fi
if [[ -n "${SPECULATIVE_DRAFT_WINDOW_SIZE}" && ! "${SPECULATIVE_DRAFT_WINDOW_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SPECULATIVE_DRAFT_WINDOW_SIZE must be empty or a positive integer; got ${SPECULATIVE_DRAFT_WINDOW_SIZE}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Speculative decode and prefill draft-state args
# ---------------------------------------------------------------------------

for _bool_setting in ENABLE_SPECULATIVE_DECODING ENABLE_SPECULATIVE_ADAPTIVE ENABLE_PREFILL_DRAFT_STATE; do
  if [[ "${!_bool_setting}" != "0" && "${!_bool_setting}" != "1" ]]; then
    echo "${_bool_setting} must be 0 or 1; got ${!_bool_setting}" >&2
    exit 2
  fi
done
unset _bool_setting

if [[ "${ENABLE_SPECULATIVE_DECODING}" == "0" && "${ENABLE_SPECULATIVE_ADAPTIVE}" == "1" ]]; then
  echo "ENABLE_SPECULATIVE_ADAPTIVE=1 requires ENABLE_SPECULATIVE_DECODING=1" >&2
  exit 2
fi
if [[ -n "${SPECULATIVE_ADAPTIVE_CONFIG}" && "${ENABLE_SPECULATIVE_ADAPTIVE}" != "1" ]]; then
  echo "SPECULATIVE_ADAPTIVE_CONFIG requires ENABLE_SPECULATIVE_ADAPTIVE=1" >&2
  exit 2
fi
if [[ -n "${SPECULATIVE_ADAPTIVE_CONFIG}" && ! -r "${SPECULATIVE_ADAPTIVE_CONFIG}" ]]; then
  echo "SPECULATIVE_ADAPTIVE_CONFIG is not readable: ${SPECULATIVE_ADAPTIVE_CONFIG}" >&2
  exit 2
fi
if [[ "${ENABLE_SPECULATIVE_DECODING}" == "0" && "${ENABLE_PREFILL_DRAFT_STATE}" == "1" ]]; then
  echo "ENABLE_PREFILL_DRAFT_STATE=1 requires ENABLE_SPECULATIVE_DECODING=1" >&2
  exit 2
fi
if [[ "${ENABLE_SPECULATIVE_DECODING}" == "1" && "${SPECULATIVE_ALGORITHM}" == "DFLASH" ]]; then
  if [[ -z "${SPECULATIVE_DRAFT_MODEL_PATH}" || ! -r "${SPECULATIVE_DRAFT_MODEL_PATH}/config.json" ]]; then
    echo "DFLASH requires SPECULATIVE_DRAFT_MODEL_PATH with a readable config.json; got ${SPECULATIVE_DRAFT_MODEL_PATH:-<empty>}" >&2
    exit 2
  fi
  if [[ "${ENABLE_SPECULATIVE_ADAPTIVE}" == "1" ]]; then
    echo "ENABLE_SPECULATIVE_ADAPTIVE is EAGLE-only; set it to 0 for DFLASH" >&2
    exit 2
  fi
  if [[ "${ENABLE_PREFILL_DRAFT_STATE}" == "0" ]]; then
    echo "PD DFLASH requires ENABLE_PREFILL_DRAFT_STATE=1 to materialize and transfer the draft KV cache" >&2
    exit 2
  fi
  if [[ "${ENABLE_DECODE_RADIX_CACHE}" == "1" ]]; then
    echo "PD decode radix cache currently supports EAGLE/NEXTN only; set ENABLE_DECODE_RADIX_CACHE=0 for the DFLASH feasibility cell" >&2
    exit 2
  fi
fi

declare -a DECODE_SPECULATIVE_ARGS=()
if [[ "${ENABLE_SPECULATIVE_DECODING}" == "1" ]]; then
  if [[ "${SPECULATIVE_ALGORITHM}" == "EAGLE" ]]; then
    DECODE_SPECULATIVE_ARGS=(
      --speculative-algorithm EAGLE
      --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
      --speculative-attention-mode prefill
      --speculative-accept-threshold-single 1.0
      --speculative-accept-threshold-acc 1.0
      --speculative-draft-model-quantization unquant
    )
    if [[ "${ENABLE_SPECULATIVE_ADAPTIVE}" == "1" ]]; then
      DECODE_SPECULATIVE_ARGS+=(--speculative-adaptive)
      if [[ -n "${SPECULATIVE_ADAPTIVE_CONFIG}" ]]; then
        DECODE_SPECULATIVE_ARGS+=(
          --speculative-adaptive-config "${SPECULATIVE_ADAPTIVE_CONFIG}"
        )
      fi
    fi
  else
    DECODE_SPECULATIVE_ARGS=(
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL_PATH}"
      --speculative-attention-mode prefill
      --speculative-draft-model-quantization unquant
    )
    if [[ -n "${SPECULATIVE_DRAFT_ATTENTION_BACKEND}" ]]; then
      DECODE_SPECULATIVE_ARGS+=(
        --speculative-draft-attention-backend "${SPECULATIVE_DRAFT_ATTENTION_BACKEND}"
      )
    fi
    if [[ -n "${SPECULATIVE_DFLASH_BLOCK_SIZE}" ]]; then
      DECODE_SPECULATIVE_ARGS+=(
        --speculative-dflash-block-size "${SPECULATIVE_DFLASH_BLOCK_SIZE}"
      )
    fi
    if [[ -n "${SPECULATIVE_DRAFT_WINDOW_SIZE}" ]]; then
      DECODE_SPECULATIVE_ARGS+=(
        --speculative-draft-window-size "${SPECULATIVE_DRAFT_WINDOW_SIZE}"
      )
    fi
  fi
fi

declare -a PREFILL_DRAFT_STATE_ARGS=()
if [[ "${ENABLE_PREFILL_DRAFT_STATE}" == "1" ]]; then
  PREFILL_DRAFT_STATE_ARGS=("${DECODE_SPECULATIVE_ARGS[@]}")
fi

for _bool_setting in \
  PREFILL_AITER_ALLREDUCE_FUSION DECODE_AITER_ALLREDUCE_FUSION \
  ENABLE_PREFILL_HICACHE \
  PREFILL_ROCPROF DECODE_ROCPROF PREFILL_FLYDSL_FP8_MQA_LOGITS \
  PREFILL_DSA_REUSE_RAGGED_LOGITS PREFILL_DSA_SEGMENT_RAGGED_LOGITS; do
  if [[ "${!_bool_setting}" != "0" && "${!_bool_setting}" != "1" ]]; then
    echo "${_bool_setting} must be 0 or 1; got ${!_bool_setting}" >&2
    exit 2
  fi
done
unset _bool_setting

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

if [[ ! "${PREFILL_MAX_INFLIGHT_TRANSFERS}" =~ ^[0-9]+$ ]]; then
  echo "PREFILL_MAX_INFLIGHT_TRANSFERS must be a non-negative integer; got ${PREFILL_MAX_INFLIGHT_TRANSFERS}" >&2
  exit 2
fi

for _radix_policy in PREFILL_RADIX_EVICTION_POLICY DECODE_RADIX_EVICTION_POLICY; do
  case "${!_radix_policy}" in
    lru|lfu|slru|priority) ;;
    *) echo "${_radix_policy} must be lru, lfu, slru, or priority; got ${!_radix_policy}" >&2; exit 2 ;;
  esac
done
unset _radix_policy

if [[ ! "${PREFILL_HICACHE_SIZE_GB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PREFILL_HICACHE_SIZE_GB must be a positive integer; got ${PREFILL_HICACHE_SIZE_GB}" >&2
  exit 2
fi

case "${PREFILL_HICACHE_WRITE_POLICY}" in
  write_back|write_through|write_through_selective) ;;
  *) echo "PREFILL_HICACHE_WRITE_POLICY must be write_back, write_through, or write_through_selective; got ${PREFILL_HICACHE_WRITE_POLICY}" >&2; exit 2 ;;
esac
case "${PREFILL_HICACHE_IO_BACKEND}:${PREFILL_HICACHE_MEM_LAYOUT}" in
  kernel:page_first|direct:page_first_direct) ;;
  *) echo "PREFILL_HICACHE_IO_BACKEND/PREFILL_HICACHE_MEM_LAYOUT must be kernel/page_first or direct/page_first_direct; got ${PREFILL_HICACHE_IO_BACKEND}/${PREFILL_HICACHE_MEM_LAYOUT}" >&2; exit 2 ;;
esac

for _mamba_limit in PREFILL_MAX_MAMBA_CACHE_SIZE DECODE_MAX_MAMBA_CACHE_SIZE; do
  if [[ -n "${!_mamba_limit}" && ! "${!_mamba_limit}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${_mamba_limit} must be empty or a positive integer; got ${!_mamba_limit}" >&2
    exit 2
  fi
done
unset _mamba_limit

for _mamba_dtype in PREFILL_MAMBA_SSM_DTYPE DECODE_MAMBA_SSM_DTYPE; do
  case "${!_mamba_dtype}" in
    ""|float32|bfloat16|float16) ;;
    *) echo "${_mamba_dtype} must be empty, float32, bfloat16, or float16; got ${!_mamba_dtype}" >&2; exit 2 ;;
  esac
done
unset _mamba_dtype

if [[ "${DECODE_MAMBA_SKIP_DECODE_LOCK}" != "0" && "${DECODE_MAMBA_SKIP_DECODE_LOCK}" != "1" ]]; then
  echo "DECODE_MAMBA_SKIP_DECODE_LOCK must be 0 or 1; got ${DECODE_MAMBA_SKIP_DECODE_LOCK}" >&2
  exit 2
fi

declare -a MODEL_OVERRIDE_ARGS=()
if [[ -n "${JSON_MODEL_OVERRIDE_ARGS}" ]]; then
  MODEL_OVERRIDE_ARGS=(--json-model-override-args "${JSON_MODEL_OVERRIDE_ARGS}")
fi

declare -a SERVING_PARSER_ARGS=()
if [[ ! "${TOOL_STRICT_LEVEL}" =~ ^[012]$ ]]; then
  echo "TOOL_STRICT_LEVEL must be 0 (off), 1 (function), or 2 (parameter); got ${TOOL_STRICT_LEVEL}" >&2
  exit 2
fi
export SGLANG_TOOL_STRICT_LEVEL="${TOOL_STRICT_LEVEL}"
if [[ ! "${ENABLE_REQUEST_LOGGING}" =~ ^[01]$ ]]; then
  echo "ENABLE_REQUEST_LOGGING must be 0 or 1; got ${ENABLE_REQUEST_LOGGING}" >&2
  exit 2
fi
if [[ ! "${REQUEST_LOG_LEVEL}" =~ ^[0-3]$ ]]; then
  echo "REQUEST_LOG_LEVEL must be 0, 1, 2, or 3; got ${REQUEST_LOG_LEVEL}" >&2
  exit 2
fi
if [[ -n "${REASONING_PARSER}" ]]; then
  SERVING_PARSER_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi
if [[ -n "${TOOL_CALL_PARSER}" ]]; then
  SERVING_PARSER_ARGS+=(--tool-call-parser "${TOOL_CALL_PARSER}")
fi
if [[ "${ENABLE_STRICT_THINKING}" == "1" ]]; then
  SERVING_PARSER_ARGS+=(--enable-strict-thinking)
elif [[ "${ENABLE_STRICT_THINKING}" != "0" ]]; then
  echo "ENABLE_STRICT_THINKING must be 0 or 1; got ${ENABLE_STRICT_THINKING}" >&2
  exit 2
fi

declare -a REQUEST_LOG_ARGS=()
if [[ "${ENABLE_REQUEST_LOGGING}" == "1" ]]; then
  REQUEST_LOG_ARGS=(--log-requests --log-requests-level "${REQUEST_LOG_LEVEL}")
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
if [[ -n "${PREFILL_MAMBA_SSM_DTYPE}" ]]; then
  PREFILL_MAMBA_ARGS+=(--mamba-ssm-dtype "${PREFILL_MAMBA_SSM_DTYPE}")
fi
declare -a DECODE_MAMBA_ARGS=(--mamba-full-memory-ratio "${DECODE_MAMBA_FULL_MEMORY_RATIO}")
if [[ -n "${DECODE_MAX_MAMBA_CACHE_SIZE}" ]]; then
  DECODE_MAMBA_ARGS+=(--max-mamba-cache-size "${DECODE_MAX_MAMBA_CACHE_SIZE}")
fi
if [[ -n "${DECODE_MAMBA_SSM_DTYPE}" ]]; then
  DECODE_MAMBA_ARGS+=(--mamba-ssm-dtype "${DECODE_MAMBA_SSM_DTYPE}")
fi

declare -a PREFILL_HICACHE_ARGS=()
if [[ "${ENABLE_PREFILL_HICACHE}" == "1" ]]; then
  PREFILL_HICACHE_ARGS=(
    --enable-hierarchical-cache
    --hicache-size "${PREFILL_HICACHE_SIZE_GB}"
    --hicache-write-policy "${PREFILL_HICACHE_WRITE_POLICY}"
    --hicache-io-backend "${PREFILL_HICACHE_IO_BACKEND}"
    --hicache-mem-layout "${PREFILL_HICACHE_MEM_LAYOUT}"
  )
fi

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------

readonly PREFILL_CHUNKED_PREFILL_SIZE=$((PREFILL_CHUNK_TOKENS * PREFILL_DP_SIZE))
readonly DECODE_CHUNKED_PREFILL_SIZE=$((65536 * DECODE_DP_SIZE))
readonly PREFILL_MORI_MAX_DISPATCH_TOKENS="${PREFILL_MORI_MAX_DISPATCH_TOKENS:-${PREFILL_CHUNKED_PREFILL_SIZE}}"

declare -a PREFILL_REQUEST_CHUNK_ARGS=()
if [[ -n "${PREFILL_REQUEST_CHUNK_TOKENS}" ]]; then
  if [[ ! "${PREFILL_REQUEST_CHUNK_TOKENS}" =~ ^[0-9]+$ ]] || \
      ((PREFILL_REQUEST_CHUNK_TOKENS < 1 || PREFILL_REQUEST_CHUNK_TOKENS > PREFILL_CHUNK_TOKENS)); then
    echo "PREFILL_REQUEST_CHUNK_TOKENS must be a positive integer no larger than PREFILL_CHUNK_TOKENS=${PREFILL_CHUNK_TOKENS}; got ${PREFILL_REQUEST_CHUNK_TOKENS}" >&2
    exit 2
  fi
  PREFILL_REQUEST_CHUNK_ARGS=(
    --chunked-prefill-request-quantum "${PREFILL_REQUEST_CHUNK_TOKENS}"
  )
  if [[ -n "${PREFILL_REQUEST_CHUNK_QUEUE_THRESHOLD}" ]]; then
    if [[ ! "${PREFILL_REQUEST_CHUNK_QUEUE_THRESHOLD}" =~ ^[0-9]+$ ]] || \
        ((PREFILL_REQUEST_CHUNK_QUEUE_THRESHOLD < 1)); then
      echo "PREFILL_REQUEST_CHUNK_QUEUE_THRESHOLD must be a positive integer; got ${PREFILL_REQUEST_CHUNK_QUEUE_THRESHOLD}" >&2
      exit 2
    fi
    PREFILL_REQUEST_CHUNK_ARGS+=(
      --chunked-prefill-request-quantum-queue-threshold "${PREFILL_REQUEST_CHUNK_QUEUE_THRESHOLD}"
    )
  fi
elif [[ -n "${PREFILL_REQUEST_CHUNK_QUEUE_THRESHOLD}" ]]; then
  echo "PREFILL_REQUEST_CHUNK_QUEUE_THRESHOLD requires PREFILL_REQUEST_CHUNK_TOKENS" >&2
  exit 2
fi

declare -a PREFILL_TRANSFER_ADMISSION_ARGS=()
if ((PREFILL_MAX_INFLIGHT_TRANSFERS > 0)); then
  PREFILL_TRANSFER_ADMISSION_ARGS=(
    --disaggregation-prefill-max-inflight-transfers "${PREFILL_MAX_INFLIGHT_TRANSFERS}"
  )
fi

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
declare -a PREFILL_ALLREDUCE_ARGS=()
if [[ "${PREFILL_CP_STRATEGY}" == "interleave" ]]; then
  PREFILL_CP_ARGS=(--enable-prefill-cp --cp-strategy interleave)
fi
if [[ "${PREFILL_AITER_ALLREDUCE_FUSION}" == "1" && "${PREFILL_CP_STRATEGY}" == "none" ]] && ((PREFILL_EP_SIZE == 1)); then
  PREFILL_ALLREDUCE_ARGS=(--enable-aiter-allreduce-fusion)
fi

declare -a DECODE_ALLREDUCE_ARGS=()
if [[ "${DECODE_AITER_ALLREDUCE_FUSION}" == "1" ]] && ((DECODE_EP_SIZE == 1)); then
  DECODE_ALLREDUCE_ARGS=(--enable-aiter-allreduce-fusion)
fi

declare -a PREFILL_AITER_CONFIG_ENV=()
if [[ -n "${PREFILL_AITER_CONFIG_FMOE}" ]]; then
  [[ -r "${PREFILL_AITER_CONFIG_FMOE}" ]] || {
    echo "PREFILL_AITER_CONFIG_FMOE is not readable: ${PREFILL_AITER_CONFIG_FMOE}" >&2
    exit 2
  }
  PREFILL_AITER_CONFIG_ENV=(AITER_CONFIG_FMOE="${PREFILL_AITER_CONFIG_FMOE}")
fi
declare -a DECODE_AITER_CONFIG_ENV=()
if [[ -n "${DECODE_AITER_CONFIG_FMOE}" ]]; then
  [[ -r "${DECODE_AITER_CONFIG_FMOE}" ]] || {
    echo "DECODE_AITER_CONFIG_FMOE is not readable: ${DECODE_AITER_CONFIG_FMOE}" >&2
    exit 2
  }
  DECODE_AITER_CONFIG_ENV=(AITER_CONFIG_FMOE="${DECODE_AITER_CONFIG_FMOE}")
fi

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
echo "[$(date -u +%FT%TZ)] Spec decode: algorithm=${SPECULATIVE_ALGORITHM} decode=${ENABLE_SPECULATIVE_DECODING} prefill_draft_state=${ENABLE_PREFILL_DRAFT_STATE} adaptive=${ENABLE_SPECULATIVE_ADAPTIVE} adaptive_config=${SPECULATIVE_ADAPTIVE_CONFIG:-default} eagle_steps=${SPECULATIVE_NUM_STEPS} eagle_draft=${SPECULATIVE_NUM_DRAFT_TOKENS} dflash_block=${SPECULATIVE_DFLASH_BLOCK_SIZE:-inferred}"
echo "[$(date -u +%FT%TZ)] Spec draft: model=${SPECULATIVE_DRAFT_MODEL_PATH:-in-checkpoint} attention=${SPECULATIVE_DRAFT_ATTENTION_BACKEND:-auto} window=${SPECULATIVE_DRAFT_WINDOW_SIZE:-full}"
echo "[$(date -u +%FT%TZ)] CUDA graph decode: ${DECODE_CUDA_GRAPH_BACKEND} max_bs=${DECODE_CUDA_GRAPH_MAX_BS}"
echo "[$(date -u +%FT%TZ)] Decode radix cache: ${ENABLE_DECODE_RADIX_CACHE}"
echo "[$(date -u +%FT%TZ)] Serving parsers: reasoning=${REASONING_PARSER:-disabled} tool_call=${TOOL_CALL_PARSER:-disabled} strict_level=${TOOL_STRICT_LEVEL} strict_thinking=${ENABLE_STRICT_THINKING} request_logging=${ENABLE_REQUEST_LOGGING} request_log_level=${REQUEST_LOG_LEVEL}"
echo "[$(date -u +%FT%TZ)] Model overrides: ${JSON_MODEL_OVERRIDE_ARGS:-none}"
echo "[$(date -u +%FT%TZ)] Mem fraction: prefill=${PREFILL_MEM_FRACTION_STATIC} decode=${DECODE_MEM_FRACTION_STATIC}"
echo "[$(date -u +%FT%TZ)] Running requests: prefill=${PREFILL_MAX_RUNNING_REQUESTS} decode=${DECODE_MAX_RUNNING_REQUESTS}"
echo "[$(date -u +%FT%TZ)] Prefill transfer admission slots: ${PREFILL_MAX_INFLIGHT_TRANSFERS} (0=disabled)"
echo "[$(date -u +%FT%TZ)] Radix eviction: prefill=${PREFILL_RADIX_EVICTION_POLICY} decode=${DECODE_RADIX_EVICTION_POLICY}"
echo "[$(date -u +%FT%TZ)] Prefill HiCache: enabled=${ENABLE_PREFILL_HICACHE} size_gb_per_rank=${PREFILL_HICACHE_SIZE_GB} write_policy=${PREFILL_HICACHE_WRITE_POLICY} io_backend=${PREFILL_HICACHE_IO_BACKEND} mem_layout=${PREFILL_HICACHE_MEM_LAYOUT}"
echo "[$(date -u +%FT%TZ)] Prefill request chunk quantum: ${PREFILL_REQUEST_CHUNK_TOKENS:-batch-limit}"
echo "[$(date -u +%FT%TZ)] Prefill request chunk queue threshold: ${PREFILL_REQUEST_CHUNK_QUEUE_THRESHOLD:-always}"
echo "[$(date -u +%FT%TZ)] KDA/Mamba: prefill_ratio=${PREFILL_MAMBA_FULL_MEMORY_RATIO} prefill_cap=${PREFILL_MAX_MAMBA_CACHE_SIZE:-auto} prefill_ssm_dtype=${PREFILL_MAMBA_SSM_DTYPE:-default} decode_ratio=${DECODE_MAMBA_FULL_MEMORY_RATIO} decode_cap=${DECODE_MAX_MAMBA_CACHE_SIZE:-auto} decode_ssm_dtype=${DECODE_MAMBA_SSM_DTYPE:-default} decode_skip_lock=${DECODE_MAMBA_SKIP_DECODE_LOCK}"
echo "[$(date -u +%FT%TZ)] QuickReduce: prefill=${PREFILL_QUICK_REDUCE_QUANTIZATION} decode=${DECODE_QUICK_REDUCE_QUANTIZATION}"
echo "[$(date -u +%FT%TZ)] AITER allreduce fusion: prefill=${PREFILL_AITER_ALLREDUCE_FUSION} decode=${DECODE_AITER_ALLREDUCE_FUSION}"
echo "[$(date -u +%FT%TZ)] AITER FMoE config: prefill=${PREFILL_AITER_CONFIG_FMOE:-default} decode=${DECODE_AITER_CONFIG_FMOE:-default}"
echo "[$(date -u +%FT%TZ)] Shared experts fusion: prefill=${PREFILL_SHARED_EXPERTS_FUSION} decode=${DECODE_SHARED_EXPERTS_FUSION}"
echo "[$(date -u +%FT%TZ)] Prefill ragged MQA logits: flydsl=${PREFILL_FLYDSL_FP8_MQA_LOGITS}"
echo "[$(date -u +%FT%TZ)] Prefill DSA K-pool allocator trim threshold: ${PREFILL_DSA_KPOOL_CACHE_TRIM_THRESHOLD}"
echo "[$(date -u +%FT%TZ)] Prefill reusable ragged logits workspace: ${PREFILL_DSA_REUSE_RAGGED_LOGITS}"
echo "[$(date -u +%FT%TZ)] Prefill segmented ragged logits: ${PREFILL_DSA_SEGMENT_RAGGED_LOGITS}"
echo "[$(date -u +%FT%TZ)] MoRI: qp_per_transfer=${MORI_QP_PER_TRANSFER} transfer_shards=${MORI_TRANSFER_SHARDS}"
echo "[$(date -u +%FT%TZ)] rocprof: prefill=${PREFILL_ROCPROF} decode=${DECODE_ROCPROF}"

# ---------------------------------------------------------------------------
# Prefill server
# ---------------------------------------------------------------------------

env -u MC_FORCE_TCP -u MOONCAKE_PROTOCOL -u SGLANG_PP_LAYER_PARTITION -u AITER_CONFIG_FMOE \
  "${PREFILL_AITER_CONFIG_ENV[@]}" \
  PYTHONPATH="${SRC}/python" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${PREFILL_LOGICAL_GPU_IDS}" \
  SGLANG_TORCH_PROFILER_DIR="${OUT}/profiles/prefill" \
  SGLANG_USE_AITER=1 \
  ROCM_QUICK_REDUCE_QUANTIZATION="${PREFILL_QUICK_REDUCE_QUANTIZATION}" \
  ROCR_VISIBLE_DEVICES="${PREFILL_GPU_IDS}" HIP_VISIBLE_DEVICES="${PREFILL_LOGICAL_GPU_IDS}" \
  SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 SGLANG_OPT_USE_FLYDSL_FP8_MQA_LOGITS="${PREFILL_FLYDSL_FP8_MQA_LOGITS}" \
  SGLANG_DSA_KPOOL_CACHE_TRIM_THRESHOLD="${PREFILL_DSA_KPOOL_CACHE_TRIM_THRESHOLD}" \
  SGLANG_DSA_REUSE_RAGGED_LOGITS="${PREFILL_DSA_REUSE_RAGGED_LOGITS}" \
  SGLANG_DSA_SEGMENT_RAGGED_LOGITS="${PREFILL_DSA_SEGMENT_RAGGED_LOGITS}" \
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
    "${SERVING_PARSER_ARGS[@]}" \
    --attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --linear-attn-backend triton --moe-runner-backend aiter \
    "${PREFILL_ALLREDUCE_ARGS[@]}" \
    --mem-fraction-static ${PREFILL_MEM_FRACTION_STATIC} --max-running-requests "${PREFILL_MAX_RUNNING_REQUESTS}" \
    "${PREFILL_MAMBA_ARGS[@]}" \
    "${PREFILL_HICACHE_ARGS[@]}" \
    --schedule-policy lpm --radix-eviction-policy "${PREFILL_RADIX_EVICTION_POLICY}" \
    --context-length "${CONTEXT_LENGTH}" --page-size 64 \
    --chunked-prefill-size ${PREFILL_CHUNKED_PREFILL_SIZE} "${PREFILL_REQUEST_CHUNK_ARGS[@]}" --max-prefill-tokens 16384 \
    --prefill-max-requests 16 "${PREFILL_OVERLAP_ARGS[@]}" \
    --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
    "${PREFILL_DRAFT_STATE_ARGS[@]}" \
    --enable-session-radix-cache \
    --disaggregation-mode prefill --disaggregation-transfer-backend ${DISAGGREGATION_TRANSFER_BACKEND} \
    --disaggregation-bootstrap-port ${BOOT} "${PREFILL_TRANSFER_ADMISSION_ARGS[@]}" --nccl-port ${P_NCCL} \
    --enable-metrics --enable-cache-report --enable-request-time-stats-logging \
    "${REQUEST_LOG_ARGS[@]}" \
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

env -u MC_FORCE_TCP -u MOONCAKE_PROTOCOL -u SGLANG_PP_LAYER_PARTITION -u AITER_CONFIG_FMOE \
  "${DECODE_AITER_CONFIG_ENV[@]}" \
  PYTHONPATH="${SRC}/python" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${DECODE_LOGICAL_GPU_IDS}" \
  SGLANG_TORCH_PROFILER_DIR="${OUT}/profiles/decode" \
  SGLANG_USE_AITER=1 \
  ROCM_QUICK_REDUCE_QUANTIZATION="${DECODE_QUICK_REDUCE_QUANTIZATION}" \
  ROCR_VISIBLE_DEVICES="${DECODE_GPU_IDS}" HIP_VISIBLE_DEVICES="${DECODE_LOGICAL_GPU_IDS}" \
  SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
  SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK="${DECODE_MAMBA_SKIP_DECODE_LOCK}" \
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
    "${SERVING_PARSER_ARGS[@]}" \
    --attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --linear-attn-backend triton --moe-runner-backend aiter \
    "${DECODE_ALLREDUCE_ARGS[@]}" \
    --mem-fraction-static ${DECODE_MEM_FRACTION_STATIC} --max-running-requests "${DECODE_MAX_RUNNING_REQUESTS}" \
    "${DECODE_MAMBA_ARGS[@]}" \
    --radix-eviction-policy "${DECODE_RADIX_EVICTION_POLICY}" \
    --context-length "${CONTEXT_LENGTH}" --page-size 64 \
    --chunked-prefill-size ${DECODE_CHUNKED_PREFILL_SIZE} --max-prefill-tokens 65536 \
    "${DECODE_OVERLAP_ARGS[@]}" \
    --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode "${DECODE_CUDA_GRAPH_BACKEND}" \
    --cuda-graph-max-bs-decode "${DECODE_CUDA_GRAPH_MAX_BS}" \
    "${DECODE_SPECULATIVE_ARGS[@]}" \
    --num-reserved-decode-tokens 1024 \
    --disaggregation-mode decode --disaggregation-transfer-backend ${DISAGGREGATION_TRANSFER_BACKEND} \
    "${DECODE_RADIX_ARGS[@]}" \
    --disaggregation-bootstrap-port ${BOOT} --nccl-port ${D_NCCL} \
    --enable-metrics --enable-cache-report --enable-request-time-stats-logging \
    "${REQUEST_LOG_ARGS[@]}" \
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
