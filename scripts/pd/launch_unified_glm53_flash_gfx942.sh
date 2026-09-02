#!/usr/bin/env bash
set -euo pipefail

# Matched non-PD control for the gfx942 GLM-5.3 Flash frontier study.
#
# This keeps the measured PD candidate's model, cache, prefill, speculative
# decoding, and kernel settings while replacing TP4 prefill + TP4 decode with
# one TP8 server.  Prefill CP remains role-specific inside SGLang; AITER's
# all-reduce fusion stays off when CP is enabled because it is a process-wide
# switch in a unified server.

: "${RUN_STAMP:?set RUN_STAMP}"
readonly S="${RUN_STAMP}"
readonly SRC="${SGLANG_SOURCE_ROOT:-/tmp/sglang-src}"
readonly OUT="/out/unified_glm53_gfx942_${S}"
readonly PORT="${PORT:-31000}"
readonly NCCL_PORT="${NCCL_PORT:-31300}"

readonly TP_SIZE="${TP_SIZE:-8}"
readonly GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
readonly CP_STRATEGY="${CP_STRATEGY:-interleave}"
readonly MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.83}"
readonly MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-120}"
readonly MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.9}"
readonly MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-}"
readonly CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
readonly CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
readonly MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
readonly PREFILL_MAX_REQUESTS="${PREFILL_MAX_REQUESTS:-16}"
readonly RADIX_EVICTION_POLICY="${RADIX_EVICTION_POLICY:-lru}"
readonly CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
readonly CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-64}"
readonly SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-5}"
readonly SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-6}"
readonly ENABLE_SPECULATIVE_ADAPTIVE="${ENABLE_SPECULATIVE_ADAPTIVE:-1}"
readonly AITER_ALLREDUCE_FUSION="${AITER_ALLREDUCE_FUSION:-0}"
readonly AITER_CONFIG_FMOE="${AITER_CONFIG_FMOE:-}"
readonly USE_FLYDSL_FP8_MQA_LOGITS="${USE_FLYDSL_FP8_MQA_LOGITS:-0}"
readonly DSA_KPOOL_CACHE_TRIM_THRESHOLD="${DSA_KPOOL_CACHE_TRIM_THRESHOLD:-0.85}"
readonly QUICK_REDUCE_QUANTIZATION="${QUICK_REDUCE_QUANTIZATION:-INT4}"

if [[ "${TP_SIZE}" != "8" ]]; then
  echo "This matched control requires TP_SIZE=8; got ${TP_SIZE}" >&2
  exit 2
fi
if [[ "${GPU_IDS}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "This matched control requires GPU_IDS=0,1,2,3,4,5,6,7; got ${GPU_IDS}" >&2
  exit 2
fi
case "${CP_STRATEGY}" in
  none|interleave) ;;
  *) echo "CP_STRATEGY must be none or interleave; got ${CP_STRATEGY}" >&2; exit 2 ;;
esac
case "${RADIX_EVICTION_POLICY}" in
  lru|lfu|slru|priority) ;;
  *) echo "Unsupported RADIX_EVICTION_POLICY=${RADIX_EVICTION_POLICY}" >&2; exit 2 ;;
esac
for setting in ENABLE_SPECULATIVE_ADAPTIVE AITER_ALLREDUCE_FUSION USE_FLYDSL_FP8_MQA_LOGITS; do
  if [[ "${!setting}" != "0" && "${!setting}" != "1" ]]; then
    echo "${setting} must be 0 or 1; got ${!setting}" >&2
    exit 2
  fi
done
if [[ ! "${SPECULATIVE_NUM_STEPS}" =~ ^[1-9][0-9]*$ ]] ||
    [[ ! "${SPECULATIVE_NUM_DRAFT_TOKENS}" =~ ^[1-9][0-9]*$ ]] ||
    ((SPECULATIVE_NUM_DRAFT_TOKENS != SPECULATIVE_NUM_STEPS + 1)); then
  echo "SPECULATIVE_NUM_DRAFT_TOKENS must equal SPECULATIVE_NUM_STEPS + 1" >&2
  exit 2
fi

declare -a CP_ARGS=()
if [[ "${CP_STRATEGY}" == "interleave" ]]; then
  CP_ARGS=(--enable-prefill-cp --cp-strategy interleave)
fi

declare -a ALLREDUCE_ARGS=()
if [[ "${AITER_ALLREDUCE_FUSION}" == "1" ]]; then
  if [[ "${CP_STRATEGY}" != "none" ]]; then
    echo "AITER_ALLREDUCE_FUSION=1 requires CP_STRATEGY=none in unified mode" >&2
    exit 2
  fi
  ALLREDUCE_ARGS=(--enable-aiter-allreduce-fusion)
fi

declare -a MAMBA_ARGS=(--mamba-full-memory-ratio "${MAMBA_FULL_MEMORY_RATIO}")
if [[ -n "${MAX_MAMBA_CACHE_SIZE}" ]]; then
  MAMBA_ARGS+=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
fi

declare -a SPECULATIVE_ARGS=(
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
  SPECULATIVE_ARGS+=(--speculative-adaptive)
fi

declare -a AITER_CONFIG_ENV=()
if [[ -n "${AITER_CONFIG_FMOE}" ]]; then
  [[ -r "${AITER_CONFIG_FMOE}" ]] || {
    echo "AITER_CONFIG_FMOE is not readable: ${AITER_CONFIG_FMOE}" >&2
    exit 2
  }
  AITER_CONFIG_ENV=(AITER_CONFIG_FMOE="${AITER_CONFIG_FMOE}")
fi

mkdir -p "${OUT}" "${OUT}/profiles"
echo "[$(date -u +%FT%TZ)] GLM-5.3 Flash FP8 unified TP8 -- stamp=${S}"
echo "[$(date -u +%FT%TZ)] GPUs=${GPU_IDS} CP=${CP_STRATEGY} mem=${MEM_FRACTION_STATIC} max_running=${MAX_RUNNING_REQUESTS}"
echo "[$(date -u +%FT%TZ)] chunk=${CHUNKED_PREFILL_SIZE} max_prefill=${MAX_PREFILL_TOKENS} prefill_max_requests=${PREFILL_MAX_REQUESTS} radix=${RADIX_EVICTION_POLICY}"
echo "[$(date -u +%FT%TZ)] EAGLE adaptive=${ENABLE_SPECULATIVE_ADAPTIVE} steps=${SPECULATIVE_NUM_STEPS} draft=${SPECULATIVE_NUM_DRAFT_TOKENS}"
echo "[$(date -u +%FT%TZ)] decode_graph=${CUDA_GRAPH_BACKEND_DECODE} max_bs=${CUDA_GRAPH_MAX_BS_DECODE} allreduce_fusion=${AITER_ALLREDUCE_FUSION}"
echo "[$(date -u +%FT%TZ)] AITER_FMoE=${AITER_CONFIG_FMOE:-default} FlyDSL_MQA=${USE_FLYDSL_FP8_MQA_LOGITS} trim=${DSA_KPOOL_CACHE_TRIM_THRESHOLD}"

env -u AITER_CONFIG_FMOE -u MC_FORCE_TCP -u MOONCAKE_PROTOCOL -u SGLANG_PP_LAYER_PARTITION \
  "${AITER_CONFIG_ENV[@]}" \
  PYTHONPATH="${SRC}/python" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ROCR_VISIBLE_DEVICES="${GPU_IDS}" HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  SGLANG_TORCH_PROFILER_DIR="${OUT}/profiles" \
  SGLANG_USE_AITER=1 SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
  SGLANG_OPT_USE_FLYDSL_FP8_MQA_LOGITS="${USE_FLYDSL_FP8_MQA_LOGITS}" \
  SGLANG_DSA_KPOOL_CACHE_TRIM_THRESHOLD="${DSA_KPOOL_CACHE_TRIM_THRESHOLD}" \
  SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0 \
  ROCM_QUICK_REDUCE_QUANTIZATION="${QUICK_REDUCE_QUANTIZATION}" \
  python3 -m sglang.launch_server \
    --model-path /model --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "${PORT}" --base-gpu-id 0 \
    --tp-size "${TP_SIZE}" --pp-size 1 \
    "${CP_ARGS[@]}" \
    --moe-a2a-backend none --deepep-dispatcher-output-dtype auto \
    --disable-shared-experts-fusion \
    --quantization fp8 --trust-remote-code --kv-cache-dtype fp8_e4m3 \
    --attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --linear-attn-backend triton --moe-runner-backend aiter \
    "${ALLREDUCE_ARGS[@]}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    "${MAMBA_ARGS[@]}" \
    --schedule-policy lpm --radix-eviction-policy "${RADIX_EVICTION_POLICY}" \
    --context-length "${CONTEXT_LENGTH}" --page-size 64 \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
    --prefill-max-requests "${PREFILL_MAX_REQUESTS}" \
    --disable-overlap-schedule \
    --cuda-graph-backend-prefill disabled \
    --cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}" \
    --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}" \
    "${SPECULATIVE_ARGS[@]}" \
    --num-reserved-decode-tokens 1024 \
    --enable-session-radix-cache \
    --nccl-port "${NCCL_PORT}" \
    --enable-metrics --enable-cache-report --enable-request-time-stats-logging \
    --random-seed 0 --watchdog-timeout 3600 \
    >"${OUT}/server.log" 2>&1 &

SERVER_PID=$!
echo "${SERVER_PID}" >"${OUT}/server.pid"
echo "[$(date -u +%FT%TZ)] Server PID=${SERVER_PID}"
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] Server healthy"
    echo SERVER_READY
    wait "${SERVER_PID}"
    exit $?
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo SERVER_DIED
    tail -80 "${OUT}/server.log"
    exit 1
  fi
  sleep 10
done

echo SERVER_NOT_HEALTHY
tail -80 "${OUT}/server.log"
exit 1
