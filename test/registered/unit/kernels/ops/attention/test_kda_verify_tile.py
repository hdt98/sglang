import unittest

import torch

from sglang.kernels.ops.attention.linear.fused_kda_conv_recurrent_verify import (
    KDA_VERIFY_BLOCK_V_HIP_T4,
    _kda_verify_t4_hip_tile_applies,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKDAVerifyT4Tile(unittest.TestCase):
    def test_tile_is_selected_for_the_measured_t4_range(self):
        shape = dict(
            B=5,
            T=4,
            H=12,
            HV=12,
            K=128,
            V=128,
            conv_weight_dtype=torch.float32,
            num_warps=1,
            is_hip=True,
        )
        self.assertTrue(_kda_verify_t4_hip_tile_applies(**shape))
        self.assertEqual(KDA_VERIFY_BLOCK_V_HIP_T4, 16)

        for name, value in (
            ("B", 4),
            ("B", 17),
            ("T", 3),
            ("H", 11),
            ("HV", 13),
            ("K", 64),
            ("V", 64),
            ("conv_weight_dtype", torch.bfloat16),
            ("num_warps", 2),
            ("is_hip", False),
        ):
            with self.subTest(field=name, value=value):
                changed = shape | {name: value}
                self.assertFalse(_kda_verify_t4_hip_tile_applies(**changed))


if __name__ == "__main__":
    unittest.main()
