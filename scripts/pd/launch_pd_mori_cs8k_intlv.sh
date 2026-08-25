#!/usr/bin/env bash
set -euo pipefail

# PD Prefill-Decode disaggregation launch script for GLM-5.2 MXFP4 on gfx942 (MI325X)
# Config: cs8k_intlv (chunked_prefill_size=8192, max_prefill_tokens=16384 on prefill)
# Transfer: MoRI XGMI
# Spec decoding: NEXTN (EAGLE, 6 draft tokens)
# Scheduler fixes: break->continue, chunk budget reservation, has_chunked_req guard
#
# Usage: docker exec -e RUN_STAMP=<tag> -e HOME=/tmp \
#   -e PYTHONPATH=/path/to/sglang/python \
#   container bash /path/to/launch_pd_mori_cs8k_intlv.sh

: "${RUN_STAMP:?set RUN_STAMP}"
readonly S=${RUN_STAMP}
readonly SRC=/tmp/sglang_nonpd_source
readonly OUT="/out/pd_tp4_nextn_p12k_${S}"
readonly P_PORT=31100 D_PORT=31200 P_NCCL=31300 D_NCCL=31400 BOOT=31500 RTR=31000

mkdir -p "${OUT}"

# Apply MoE kernel overlay if available
cp /tmp/aiter_mxfp4_triton_decode_tile_candidate.py \
  "${SRC}/python/sglang/srt/layers/moe/moe_runner/aiter_mxfp4_triton.py" 2>/dev/null || true
echo "[$(date -u +%FT%TZ)] Decode tile overlay applied (if available)"

# Prefill server: GPUs 0-3, TP4, mem-fraction 0.85
env -u CUDA_VISIBLE_DEVICES -u MC_FORCE_TCP -u MOONCAKE_PROTOCOL \
  PYTHONPATH="${SRC}/python" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  ROCR_VISIBLE_DEVICES=0,1,2,3 HIP_VISIBLE_DEVICES=0,1,2,3 \
  SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 MORI_DISABLE_AUTO_XGMI=0 MC_TE_METRIC=1 \
  python3 -m sglang.launch_server \
    --model-path /model --served-model-name glm52-pd-prefill \
    --host 0.0.0.0 --port ${P_PORT} --base-gpu-id 0 \
    --tp-size 4 --moe-a2a-backend none --quantization quark \
    --trust-remote-code --kv-cache-dtype fp8_e4m3 \
    --attention-backend dsa --dsa-prefill-backend tilelang \
    --dsa-decode-backend tilelang --dsa-topk-backend sgl-kernel \
    --moe-runner-backend aiter --enable-aiter-allreduce-fusion \
    --mem-fraction-static 0.85 --max-running-requests 16 \
    --schedule-policy lpm --context-length 262144 --page-size 64 \
    --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
    --prefill-max-requests 16 --disable-overlap-schedule \
    --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
    --speculative-algorithm NEXTN --speculative-num-steps 5 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
    --speculative-attention-mode prefill \
    --speculative-accept-threshold-single 1.0 --speculative-accept-threshold-acc 1.0 \
    --speculative-draft-model-quantization unquant \
    --enable-session-radix-cache \
    --disaggregation-mode prefill --disaggregation-transfer-backend mori \
    --disaggregation-bootstrap-port ${BOOT} --nccl-port ${P_NCCL} \
    --enable-metrics --enable-cache-report --enable-request-time-stats-logging \
    --random-seed 0 --watchdog-timeout 3600 \
    > "${OUT}/prefill.log" 2>&1 &
P_PID=$!
echo "${P_PID}" > "${OUT}/prefill.pid"
echo "[$(date -u +%FT%TZ)] Prefill PID=${P_PID}"
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${P_PORT}/health" >/dev/null 2>&1 && break
  kill -0 ${P_PID} 2>/dev/null || { echo PREFILL_DIED; tail -30 "${OUT}/prefill.log"; exit 1; }
  sleep 10
done
echo "[$(date -u +%FT%TZ)] Prefill healthy"

# Decode server: GPUs 4-7, TP4, mem-fraction 0.80
env -u CUDA_VISIBLE_DEVICES -u MC_FORCE_TCP -u MOONCAKE_PROTOCOL \
  PYTHONPATH="${SRC}/python" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  ROCR_VISIBLE_DEVICES=4,5,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 \
  SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 MORI_DISABLE_AUTO_XGMI=0 MC_TE_METRIC=1 \
  python3 -m sglang.launch_server \
    --model-path /model --served-model-name glm52-pd-decode \
    --host 0.0.0.0 --port ${D_PORT} --base-gpu-id 0 \
    --tp-size 4 --moe-a2a-backend none --quantization quark \
    --trust-remote-code --kv-cache-dtype fp8_e4m3 \
    --attention-backend dsa --dsa-prefill-backend tilelang \
    --dsa-decode-backend tilelang --dsa-topk-backend sgl-kernel \
    --moe-runner-backend aiter --enable-aiter-allreduce-fusion \
    --mem-fraction-static 0.80 --max-running-requests 120 \
    --context-length 262144 --page-size 64 \
    --chunked-prefill-size 65536 --max-prefill-tokens 65536 \
    --disable-overlap-schedule \
    --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
    --cuda-graph-max-bs-decode 64 \
    --speculative-algorithm NEXTN --speculative-num-steps 5 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
    --speculative-attention-mode prefill \
    --speculative-accept-threshold-single 1.0 --speculative-accept-threshold-acc 1.0 \
    --speculative-draft-model-quantization unquant \
    --disaggregation-mode decode --disaggregation-transfer-backend mori \
    --disaggregation-decode-enable-radix-cache \
    --num-reserved-decode-tokens 1024 \
    --disaggregation-bootstrap-port ${BOOT} --nccl-port ${D_NCCL} \
    --enable-metrics --enable-cache-report --enable-request-time-stats-logging \
    --random-seed 0 --watchdog-timeout 3600 \
    > "${OUT}/decode.log" 2>&1 &
D_PID=$!
echo "${D_PID}" > "${OUT}/decode.pid"
echo "[$(date -u +%FT%TZ)] Decode PID=${D_PID}"
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${D_PORT}/health" >/dev/null 2>&1 && break
  kill -0 ${D_PID} 2>/dev/null || { echo DECODE_DIED; tail -30 "${OUT}/decode.log"; exit 1; }
  sleep 10
done
echo "[$(date -u +%FT%TZ)] Decode healthy"

# Router
PYTHONPATH="${SRC}/python" python3 -m sglang_router.launch_router \
  --host 0.0.0.0 --port ${RTR} \
  --pd-disaggregation --mini-lb \
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
