#!/usr/bin/env bash
# GLM-5.3-Flash-MXFP4 baseline on 4x AMD Instinct MI355X (gfx950), TP4/EP4.
#
# NOTE: this model REQUIRES the patched candidate-v30 SGLang overlay. glm5_next
# (Glm5NextForConditionalGeneration) is not in upstream SGLang or in any
# lmsysorg/sglang-rocm image -- the overlay is the implementation, not a tuning
# patch. Build the runtime tree with prepare_overlay.sh; see README.md.
#
# Weights: hf download OneNexus/GLM-5.3-Flash-MXFP4 \
#            --revision 854c8481e0c1f4cf95d16b9cd57c59c9e9ac01e1
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the GLM-5.3-Flash-MXFP4 checkout}"
OVERLAY_DIR="${OVERLAY_DIR:?set OVERLAY_DIR to the candidate-v30 python/sglang tree}"
AITER_JIT_DIR="${AITER_JIT_DIR:?set AITER_JIT_DIR to the pinned aiter jit cache}"
TRANSFORMERS_DIR="${TRANSFORMERS_DIR:?set TRANSFORMERS_DIR to the pinned Transformers checkout}"
TRANSFORMERS_RUNTIME_DIR="${TRANSFORMERS_RUNTIME_DIR:?set TRANSFORMERS_RUNTIME_DIR to the pinned processor dependencies}"
API_KEY="${SGLANG_API_KEY:?set SGLANG_API_KEY}"
IMAGE="${IMAGE:-lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260903}"
PORT="${PORT:-30037}"
GPUS="${GPUS:-4,5,6,7}"
CPUSET_CPUS="${CPUSET_CPUS:-96-191}"
CPUSET_MEMS="${CPUSET_MEMS:-1}"

if ! grep -Fq 'batch.mamba_track_indices[freed_rows] = -1' \
  "${OVERLAY_DIR}/srt/managers/schedule_batch.py" || \
  ! grep -Fq 'def _select_finished_checkpoint(' \
  "${OVERLAY_DIR}/srt/mem_cache/unified_cache/components/mamba_component.py" || \
  ! grep -Fq 'cache_controller.load_fence_stream' \
  "${OVERLAY_DIR}/srt/managers/scheduler.py" || \
  ! grep -Fq 'self.downsample = Conv2dLayer(' \
  "${OVERLAY_DIR}/srt/models/glm5_next.py"; then
  echo "OVERLAY_DIR is unpatched; run prepare_overlay.sh first: ${OVERLAY_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TRANSFORMERS_DIR}/src/transformers/models/glm5_next/processing_glm5_next.py" ]] || \
  [[ ! -d "${TRANSFORMERS_RUNTIME_DIR}/tokenizers" ]] || \
  [[ ! -f "${TRANSFORMERS_RUNTIME_DIR}/glm53_flash_chat_template.jinja" ]]; then
  echo "GLM-5.3 processor stack is incomplete; run prepare_transformers.sh first" >&2
  exit 1
fi

docker run --rm --name glm53-flash-mxfp4 \
  --device /dev/kfd --device /dev/dri \
  --security-opt label=disable --ipc host --shm-size 32g \
  --cpuset-cpus "${CPUSET_CPUS}" --cpuset-mems "${CPUSET_MEMS}" \
  -p "0.0.0.0:${PORT}:30000" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${OVERLAY_DIR}:/sgl-workspace/sglang/python/sglang:ro" \
  -v "${AITER_JIT_DIR}:/sgl-workspace/aiter/aiter/jit" \
  -v "${TRANSFORMERS_DIR}:/transformers:ro" \
  -v "${TRANSFORMERS_RUNTIME_DIR}:/transformers-runtime:ro" \
  -e ROCR_VISIBLE_DEVICES="${GPUS}" \
  -e PYTHONPATH=/transformers-runtime:/transformers/src:/sgl-workspace/sglang/python \
  -e SGLANG_SET_CPU_AFFINITY=0 \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_OPT_USE_TOPK_V2=false \
  -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
  -e SGLANG_TIMEOUT_KEEP_ALIVE=900 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path /model --served-model-name OneNexus/GLM-5.3-Flash-MXFP4 \
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
  --speculative-draft-model-quantization quark \
  --enable-hierarchical-cache --hicache-ratio 0.50 \
  --hicache-write-policy write_through --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --enable-strict-thinking \
  --mm-feature-transport cpu \
  --chat-template /transformers-runtime/glm53_flash_chat_template.jinja \
  --log-requests --log-requests-level 3 \
  --watchdog-timeout 1800 --dist-timeout 600 \
  --enable-metrics --enable-cache-report --enable-request-time-stats-logging \
  --random-seed 91292229
