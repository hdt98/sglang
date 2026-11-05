import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kda_backend import (
    _kda_fused_verify_t4_batch_enabled,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKDAFusedVerifyT4Gate(unittest.TestCase):
    def setUp(self):
        envs.SGLANG_KDA_FUSED_VERIFY_T4_MAX_BS.clear()
        self.addCleanup(envs.SGLANG_KDA_FUSED_VERIFY_T4_MAX_BS.clear)

    def test_default_limit_covers_only_the_measured_t4_batches(self):
        self.assertFalse(_kda_fused_verify_t4_batch_enabled(0))
        self.assertTrue(_kda_fused_verify_t4_batch_enabled(1))
        self.assertTrue(_kda_fused_verify_t4_batch_enabled(4))
        self.assertFalse(_kda_fused_verify_t4_batch_enabled(5))

    def test_limit_can_be_extended_without_changing_the_lower_bound(self):
        with envs.SGLANG_KDA_FUSED_VERIFY_T4_MAX_BS.override(8):
            self.assertFalse(_kda_fused_verify_t4_batch_enabled(0))
            self.assertTrue(_kda_fused_verify_t4_batch_enabled(1))
            self.assertTrue(_kda_fused_verify_t4_batch_enabled(8))
            self.assertFalse(_kda_fused_verify_t4_batch_enabled(9))


if __name__ == "__main__":
    unittest.main()
