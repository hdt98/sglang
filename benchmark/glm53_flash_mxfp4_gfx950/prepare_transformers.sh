#!/usr/bin/env bash
# Prepare the pinned Hugging Face processor stack required for GLM-5.3 vision.
set -euo pipefail

TRANSFORMERS_DIR="${TRANSFORMERS_DIR:?set TRANSFORMERS_DIR to a new source directory}"
TRANSFORMERS_RUNTIME_DIR="${TRANSFORMERS_RUNTIME_DIR:?set TRANSFORMERS_RUNTIME_DIR to a new dependency directory}"
TRANSFORMERS_COMMIT="${TRANSFORMERS_COMMIT:-e4052f55b26e1e29ef7bd54b28600787bcc62ef8}"
TOKENIZERS_VERSION="${TOKENIZERS_VERSION:-0.23.2}"
CHAT_TEMPLATE_REVISION="${CHAT_TEMPLATE_REVISION:-690b705278a3a58e538fcb37c2ca8b5f9511213c}"
CHAT_TEMPLATE_SHA256="${CHAT_TEMPLATE_SHA256:-0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5}"
CHAT_TEMPLATE_FILE="${TRANSFORMERS_RUNTIME_DIR}/glm53_flash_chat_template.jinja"

if [[ -e "${TRANSFORMERS_DIR}" ]]; then
  echo "TRANSFORMERS_DIR already exists; refusing to overwrite it: ${TRANSFORMERS_DIR}" >&2
  exit 1
fi
if [[ -e "${TRANSFORMERS_RUNTIME_DIR}" ]]; then
  echo "TRANSFORMERS_RUNTIME_DIR already exists; refusing to overwrite it: ${TRANSFORMERS_RUNTIME_DIR}" >&2
  exit 1
fi

mkdir -p "${TRANSFORMERS_DIR}" "${TRANSFORMERS_RUNTIME_DIR}"
git -C "${TRANSFORMERS_DIR}" init -q
git -C "${TRANSFORMERS_DIR}" remote add origin https://github.com/huggingface/transformers.git
git -C "${TRANSFORMERS_DIR}" fetch -q --depth 1 origin "${TRANSFORMERS_COMMIT}"
git -C "${TRANSFORMERS_DIR}" checkout -q --detach FETCH_HEAD

actual_commit="$(git -C "${TRANSFORMERS_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${TRANSFORMERS_COMMIT}" ]]; then
  echo "unexpected Transformers commit: ${actual_commit}" >&2
  exit 1
fi

python3 -m pip install --disable-pip-version-check --no-cache-dir --no-deps \
  --target "${TRANSFORMERS_RUNTIME_DIR}" "tokenizers==${TOKENIZERS_VERSION}"

curl --fail --location --silent --show-error \
  "https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/${CHAT_TEMPLATE_REVISION}/chat_template.jinja" \
  --output "${CHAT_TEMPLATE_FILE}.tmp"
printf '%s  %s\n' "${CHAT_TEMPLATE_SHA256}" "${CHAT_TEMPLATE_FILE}.tmp" | \
  sha256sum -c - >/dev/null
mv "${CHAT_TEMPLATE_FILE}.tmp" "${CHAT_TEMPLATE_FILE}"

test -f "${TRANSFORMERS_DIR}/src/transformers/models/glm5_next/processing_glm5_next.py"
test -d "${TRANSFORMERS_RUNTIME_DIR}/tokenizers"
test -f "${CHAT_TEMPLATE_FILE}"
echo "Prepared Transformers ${actual_commit}, tokenizers ${TOKENIZERS_VERSION}, and GLM-5.3-Flash template ${CHAT_TEMPLATE_REVISION}"
