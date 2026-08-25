#!/usr/bin/env bash
set -euo pipefail

# GSP (Generated Shared Prefix) benchmark script for PD disaggregation
# Usage: bash bench_gsp.sh <concurrency> <output_dir> [profile]
# profile: normal_primary (default), normal_stress, degraded_primary, degraded_stress

C=${1:?usage: bench_gsp.sh <concurrency> <output_dir> [profile]}
OUTDIR=${2:-/out/gsp_chunkfix}
PROFILE=${3:-normal_primary}
PORT=${PORT:-31000}

mkdir -p "${OUTDIR}"

PYTHONPATH=/tmp/sglang_nonpd_source/python python3 -m sglang.bench_serving \
  --backend sglang-oai-chat --host 127.0.0.1 --port ${PORT} \
  --model glm-5.2 --tokenizer /model \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 32 --gsp-prompts-per-group 1 --gsp-num-turns 10 \
  --gsp-system-prompt-len 223834 --gsp-question-len 581 --gsp-output-len 602 \
  --gsp-range-ratio 0.16 --gsp-group-distribution uniform --gsp-ordered \
  --max-concurrency ${C} --request-rate inf --warmup-requests 0 \
  --cache-report --output-details \
  --output-file "${OUTDIR}/gsp_${PROFILE}_C${C}.jsonl"
