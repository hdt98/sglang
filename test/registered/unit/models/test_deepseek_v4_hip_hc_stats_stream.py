import unittest
from types import SimpleNamespace
from unittest import mock

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ForwardMode:
    def __init__(self, *, decode=False, target_verify=False):
        self._decode = decode
        self._target_verify = target_verify

    def is_decode(self):
        return self._decode

    def is_target_verify(self):
        return self._target_verify


class TestHipHcStatsStreamSelection(unittest.TestCase):
    def setUp(self):
        self.layer = SimpleNamespace(
            hc_stats_stream="cuda-stream",
            hc_stats_stream_hip="hip-stream",
        )

    def _select(self, *, rows=1, decode=False, target_verify=False):
        hidden_states = SimpleNamespace(shape=(rows, 4096))
        forward_batch = SimpleNamespace(
            forward_mode=_ForwardMode(
                decode=decode,
                target_verify=target_verify,
            )
        )
        return deepseek_v4.DeepseekV4DecoderLayer._get_hc_stats_stream(
            self.layer,
            hidden_states,
            forward_batch,
        )

    def test_hip_stream_is_limited_to_decode_and_nonempty_verify(self):
        with mock.patch.object(deepseek_v4, "_is_hip", True):
            self.assertEqual(self._select(decode=True), "hip-stream")
            self.assertEqual(
                self._select(rows=48, target_verify=True),
                "hip-stream",
            )
            self.assertIsNone(self._select(rows=0, target_verify=True))
            self.assertIsNone(self._select())

    def test_cuda_path_keeps_existing_sm90_shape_gate(self):
        with (
            mock.patch.object(deepseek_v4, "_is_hip", False),
            mock.patch.object(
                deepseek_v4,
                "get_platform",
                return_value=SimpleNamespace(is_sm90=True),
            ),
        ):
            self.assertEqual(self._select(rows=1, decode=True), "cuda-stream")
            self.assertIsNone(self._select(rows=2, decode=True))


if __name__ == "__main__":
    unittest.main()
