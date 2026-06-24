import unittest

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    materialize_bpreshuffle_fp8_scale,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBpreshuffleScaleMaterialization(CustomTestCase):
    def _dequant_with_logical_scale(self, q_input, scale):
        m, n = q_input.shape
        ng = scale.shape[1]
        group = n // ng
        return (
            q_input.to(torch.float32)
            .view(m, ng, group)
            .mul(scale.to(torch.float32).unsqueeze(-1))
            .view(m, n)
        )

    def test_materializes_transposed_physical_storage(self):
        scale = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        materialized = materialize_bpreshuffle_fp8_scale(scale)

        self.assertTrue(torch.equal(materialized, scale))
        self.assertEqual(materialized.shape, scale.shape)
        self.assertEqual(materialized.stride(), (1, scale.shape[0]))
        self.assertTrue(materialized.t().is_contiguous())

    def test_materialization_is_idempotent_for_bpreshuffle_layout(self):
        scale = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        materialized = materialize_bpreshuffle_fp8_scale(scale)

        rematerialized = materialize_bpreshuffle_fp8_scale(materialized)

        self.assertTrue(torch.equal(rematerialized, scale))
        self.assertEqual(rematerialized.stride(), materialized.stride())

    def test_materialization_preserves_logical_scale_dequantization(self):
        q_input = torch.ones((3, 8), dtype=torch.float32)
        scale = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        materialized = materialize_bpreshuffle_fp8_scale(scale)

        # Simulates a row-major tensor whose underlying values were written in
        # bpreshuffle/transposed physical order. This is the layout that is unsafe
        # for generic consumers that index scale[m, k_block] logically.
        shuffled_storage_row_major = scale.t().contiguous().view_as(scale)

        expected = self._dequant_with_logical_scale(q_input, scale)

        self.assertTrue(
            torch.equal(
                self._dequant_with_logical_scale(q_input, materialized), expected
            )
        )
        self.assertFalse(
            torch.equal(
                self._dequant_with_logical_scale(
                    q_input, shuffled_storage_row_major
                ),
                expected,
            )
        )


if __name__ == "__main__":
    unittest.main()
