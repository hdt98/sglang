"""ROCm backend selection for ragged DSA MQA logits."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx942_supported


def resolve_hip_ragged_mqa_logits() -> Callable:
    """Resolve the AITER ragged MQA-logits implementation at worker startup."""
    if not envs.SGLANG_OPT_USE_FLYDSL_FP8_MQA_LOGITS.get():
        module = importlib.import_module("aiter.ops.triton.attention.fp8_mqa_logits")
        return module.fp8_mqa_logits

    if not is_gfx942_supported():
        raise ValueError("SGLANG_OPT_USE_FLYDSL_FP8_MQA_LOGITS requires gfx942.")

    try:
        module = importlib.import_module("aiter.ops.flydsl")
    except ImportError as exc:
        raise RuntimeError(
            "SGLANG_OPT_USE_FLYDSL_FP8_MQA_LOGITS requires an AITER build "
            "with AITER FlyDSL support."
        ) from exc
    return module.flydsl_fp8_mqa_logits
