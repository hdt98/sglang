#!/usr/bin/env bash
# GLM-5.3-MXFP4 baseline on 4x AMD Instinct MI355X (gfx950), TP4/EP4.
#
# Recipe source: the "Stock InferenceX/SGLang recipe" on
# https://huggingface.co/OneNexus/GLM-5.3-MXFP4, with two deliberate changes for a
# shared endpoint (see README.md): batch sizing raised from the card's c2 latency
# cell, and --api-key added.
#
# Weights:  hf download OneNexus/GLM-5.3-MXFP4 --revision 104690ed94d48341ec9de43b1bc12d30f7eaa86e
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the GLM-5.3-MXFP4 checkout}"
API_KEY="${SGLANG_API_KEY:?set SGLANG_API_KEY}"
IMAGE="${IMAGE:-lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260902}"
PORT="${PORT:-30000}"
GPUS="${GPUS:-0,1,2,3}"

docker run --rm --name glm53-mxfp4 \
  --device /dev/kfd --device /dev/dri \
  --security-opt label=disable --ipc host --shm-size 32g \
  -p "0.0.0.0:${PORT}:30000" \
  -v "${MODEL_DIR}:/model:ro" \
  -e ROCR_VISIBLE_DEVICES="${GPUS}" \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_OPT_USE_TOPK_V2=false \
  -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
  -e SGLANG_TIMEOUT_KEEP_ALIVE=900 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path /model --served-model-name OneNexus/GLM-5.3-MXFP4 \
  --host 0.0.0.0 --port 30000 --api-key "${API_KEY}" \
  --tp-size 4 --ep-size 4 --quantization quark --trust-remote-code \
  --context-length 1048576 --page-size 64 --mem-fraction-static 0.849 \
  --max-running-requests 64 --cuda-graph-max-bs 64 --min-free-slots-delay 1 \
  --chunked-prefill-size 32768 --max-prefill-tokens 16384 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --moe-runner-backend aiter \
  --speculative-algorithm EAGLE --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --speculative-attention-mode prefill \
  --speculative-accept-threshold-single 1.0 --speculative-accept-threshold-acc 1.0 \
  --speculative-draft-model-quantization unquant \
  --enable-hierarchical-cache --hicache-ratio 0.50 \
  --hicache-write-policy write_through --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --mm-feature-transport cpu \
  --log-requests --log-requests-level 3 \
  --watchdog-timeout 1800 --dist-timeout 600 \
  --enable-metrics --enable-cache-report --enable-request-time-stats-logging \
  --random-seed 91292229
