#!/usr/bin/env bash
# AgentX replay against the GLM-5.3-Flash-MXFP4 baseline, via the SemiAnalysis
# agentx-harness fork of AIPerf. Canonical flags only -- do not add
# --max-context-length and do not use a _256k corpus variant; both drop the long
# 300K-1M-token traces the benchmark depends on and break CI comparability.
#
#   MODE=smoke ./run_agentx.sh     # 10 traces, c8, 900s  -- validation
#   MODE=full  ./run_agentx.sh     # 393 traces, c32, 1800s -- CI-comparable
set -euo pipefail

MODE="${MODE:-smoke}"
URL="${URL:-localhost:30037}"
MODEL="${MODEL:-OneNexus/GLM-5.3-Flash-MXFP4}"
API_KEY="${SGLANG_API_KEY:?set SGLANG_API_KEY}"
SKILL_DIR="${SKILL_DIR:?set SKILL_DIR to the sglang-agentx-benchmark skill checkout}"
ARTIFACT_DIR="${ARTIFACT_DIR:-./artifacts/agentx-$(date -u +%Y%m%dT%H%M%SZ)}"

# The bundled 20-item GSM8K gate has two defective items -- q8's key says 140 but
# (60+120+360)/3 = 180, and q9 asks for two-thirds of 25 -- so 18/20 is the
# ceiling for a healthy model and the hardcoded 95% threshold cannot pass. Gate
# at 90%, which still catches real corruption, then skip the in-runner re-check.
bash "${SKILL_DIR}/scripts/gsm8k_gate.sh" \
    --url "${URL}" --model "${MODEL}" --api-key "${API_KEY}" --threshold 0.90

bash "${SKILL_DIR}/scripts/run_benchmark.sh" \
    --mode "${MODE}" --url "${URL}" --model "${MODEL}" --api-key "${API_KEY}" \
    --skip-correctness --artifact-dir "${ARTIFACT_DIR}"
