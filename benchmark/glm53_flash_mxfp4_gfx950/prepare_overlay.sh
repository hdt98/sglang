#!/usr/bin/env bash
# Copy the candidate-v30 GLM-5.3 overlay and apply this branch's cache-integrity
# fixes without modifying the source overlay in place.
set -euo pipefail

BASE_OVERLAY_DIR="${BASE_OVERLAY_DIR:?set BASE_OVERLAY_DIR to the unmodified candidate-v30 python/sglang tree}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to a new patched python/sglang tree}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/patches/0002-mamba-radix-finished-state-integrity.patch"

if [[ ! -f "${BASE_OVERLAY_DIR}/srt/managers/schedule_batch.py" ]]; then
  echo "BASE_OVERLAY_DIR is not a python/sglang tree: ${BASE_OVERLAY_DIR}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "OUTPUT_DIR already exists; refusing to overwrite it: ${OUTPUT_DIR}" >&2
  exit 1
fi

# Check compatibility before copying so a mismatched overlay leaves no partial
# output directory behind.
patch --batch --forward --dry-run -p1 -d "${BASE_OVERLAY_DIR}" < "${PATCH_FILE}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"
cp -a "${BASE_OVERLAY_DIR}" "${OUTPUT_DIR}"
patch --batch --forward -p1 -d "${OUTPUT_DIR}" < "${PATCH_FILE}"
python3 -m compileall -q "${OUTPUT_DIR}/srt"

grep -Fq 'batch.mamba_track_indices[freed_rows] = -1' \
  "${OUTPUT_DIR}/srt/managers/schedule_batch.py"
grep -Fq 'def _select_finished_checkpoint(' \
  "${OUTPUT_DIR}/srt/mem_cache/unified_cache/components/mamba_component.py"

echo "Prepared patched overlay: ${OUTPUT_DIR}"
