"""The DCP verify path's two fp32 scalars are allocated once, not per call.

``_forward_verify_asm_cprr`` hands the asm kernel a kv_scale and a q_scale as
0-dim fp32 device tensors. Rebuilt per call they are two kernel launches per
MLA layer per forward pass, which a B1 decode census measured as 48 launches
and 0.1831 ms of device time per step on an 8-rank TP8 run -- entirely launch
overhead, since the scalar values do not change between calls.

The values come from ``_resolve_fp8_kv_scale_float``, which already caches the
float on the layer, so only the device tensor was being rebuilt. These tests
pin that the buffers are reused, that a layer whose scale differs gets its own
value rather than a stale one, and that nothing is copied from host memory --
``torch.as_tensor`` would be a host copy, which a cuda graph cannot record,
and that is why the original code built the tensor on device each time.
"""

import unittest

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


class _Backend:
    """Just enough of the backend to exercise the scalar cache."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self._cprr_scalars = {}

    _cprr_scalar = AiterAttnBackend._cprr_scalar


class TestCprrScalarBuffers(CustomTestCase):
    def test_the_same_value_returns_the_same_tensor(self):
        """A value already built is handed back, not rebuilt."""
        backend = _Backend()

        first = backend._cprr_scalar(1.0)
        second = backend._cprr_scalar(1.0)

        self.assertIs(first, second)

    def test_the_tensor_is_a_zero_dim_fp32_device_tensor(self):
        """The asm kernel's contract: a 0-dim fp32 tensor, not a Python float."""
        backend = _Backend()

        scalar = backend._cprr_scalar(0.5)

        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.dtype, torch.float32)
        self.assertEqual(scalar.device, backend.device)
        self.assertEqual(scalar.item(), 0.5)

    def test_a_different_value_gets_its_own_tensor(self):
        """Two layers with different kv scales must not share one buffer.

        A single reused buffer overwritten in place would hand the second
        layer the first layer's scale, which is a wrong number rather than a
        slower one.
        """
        backend = _Backend()

        one = backend._cprr_scalar(1.0)
        other = backend._cprr_scalar(0.25)

        self.assertIsNot(one, other)
        self.assertEqual(one.item(), 1.0)
        self.assertEqual(other.item(), 0.25)

    def test_a_repeated_pass_allocates_nothing_further(self):
        """The point of the cache: a steady state that stops allocating.

        Twelve passes over three distinct scales build three tensors, which
        is what removes the per-pass launches the census measured.
        """
        backend = _Backend()

        for _ in range(12):
            for value in (1.0, 0.25, 448.0):
                backend._cprr_scalar(value)

        self.assertEqual(len(backend._cprr_scalars), 3)

    def test_the_cache_key_separates_values_that_compare_equal_loosely(self):
        """1 and 1.0 are the same scale and may share a tensor."""
        backend = _Backend()

        self.assertIs(backend._cprr_scalar(1), backend._cprr_scalar(1.0))


if __name__ == "__main__":
    unittest.main()
