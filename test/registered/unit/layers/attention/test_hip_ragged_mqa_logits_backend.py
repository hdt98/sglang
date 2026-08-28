"""Unit tests for the ROCm ragged MQA-logits backend selector."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch, sentinel

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import ragged_mqa_logits_backend as backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHipRaggedMQALogitsBackend(unittest.TestCase):
    def test_default_uses_aiter_triton(self):
        with patch.object(backend.importlib, "import_module") as import_module:
            import_module.return_value = SimpleNamespace(
                fp8_mqa_logits=sentinel.triton_logits
            )
            with envs.SGLANG_OPT_USE_FLYDSL_FP8_MQA_LOGITS.override(False):
                fn = backend.resolve_hip_ragged_mqa_logits()

        self.assertIs(fn, sentinel.triton_logits)
        import_module.assert_called_once_with(
            "aiter.ops.triton.attention.fp8_mqa_logits"
        )

    def test_opt_in_uses_flydsl_on_gfx942(self):
        with patch.object(
            backend, "is_gfx942_supported", return_value=True
        ), patch.object(backend.importlib, "import_module") as import_module:
            import_module.return_value = SimpleNamespace(
                flydsl_fp8_mqa_logits=sentinel.flydsl_logits
            )
            with envs.SGLANG_OPT_USE_FLYDSL_FP8_MQA_LOGITS.override(True):
                fn = backend.resolve_hip_ragged_mqa_logits()

        self.assertIs(fn, sentinel.flydsl_logits)
        import_module.assert_called_once_with("aiter.ops.flydsl")

    def test_opt_in_rejects_other_architectures(self):
        with patch.object(backend, "is_gfx942_supported", return_value=False):
            with envs.SGLANG_OPT_USE_FLYDSL_FP8_MQA_LOGITS.override(True):
                with self.assertRaisesRegex(ValueError, "requires gfx942"):
                    backend.resolve_hip_ragged_mqa_logits()

    def test_opt_in_reports_missing_flydsl(self):
        with patch.object(
            backend, "is_gfx942_supported", return_value=True
        ), patch.object(
            backend.importlib,
            "import_module",
            side_effect=ImportError("flydsl unavailable"),
        ):
            with envs.SGLANG_OPT_USE_FLYDSL_FP8_MQA_LOGITS.override(True):
                with self.assertRaisesRegex(RuntimeError, "AITER FlyDSL"):
                    backend.resolve_hip_ragged_mqa_logits()


if __name__ == "__main__":
    unittest.main()
