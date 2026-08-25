"""Unit tests for Mori's RDMA-to-XGMI fallback setup."""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci
from sglang.test.test_utils import CustomTestCase

try:
    from mori.io import BackendType

    from sglang.srt.disaggregation.mori.conn import (
        _ensure_xgmi_fallback_kernels,
    )
except ModuleNotFoundError as exc:
    if exc.name is None or exc.name.split(".", 1)[0] != "mori":
        raise
    _ensure_xgmi_fallback_kernels = None
    BackendType = None


register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_amd_ci(est_time=2, suite="stage-a-test-1-gpu-small-amd")


@unittest.skipIf(_ensure_xgmi_fallback_kernels is None, "Mori is not installed")
class TestMoriXgmiFallback(CustomTestCase):
    def test_loads_xgmi_kernels_for_fallback_sentinel(self):
        engine = MagicMock()

        loaded = _ensure_xgmi_fallback_kernels(engine, actual_port=1)

        self.assertTrue(loaded)
        engine.create_backend.assert_called_once_with(BackendType.XGMI)

    def test_does_not_touch_active_rdma_backend(self):
        engine = MagicMock()

        loaded = _ensure_xgmi_fallback_kernels(engine, actual_port=23456)

        self.assertFalse(loaded)
        engine.create_backend.assert_not_called()

    def test_preserves_fallback_when_xgmi_kernel_setup_fails(self):
        engine = MagicMock()
        engine.create_backend.side_effect = RuntimeError("kernel setup failed")

        with self.assertLogs("sglang.srt.disaggregation.mori.conn", level="WARNING"):
            loaded = _ensure_xgmi_fallback_kernels(engine, actual_port=1)

        self.assertFalse(loaded)


if __name__ == "__main__":
    unittest.main()
