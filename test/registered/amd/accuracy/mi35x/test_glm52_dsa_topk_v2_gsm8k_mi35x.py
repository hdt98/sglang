"""MI350X GLM-5.2-FP8 DSA fused top-k v2 accuracy gate (GPUs 0-3, TP=4).

Nightly AMD A/B regression test for the DeepSeek-V4 fused top-k v2 JIT
kernel on ROCm: launches the GLM-5.2 DSA recipe once with
SGLANG_OPT_USE_TOPK_V2=0 (legacy v1) and once with =1 (fused v2), runs the
full GSM8K 8-shot split against both, and requires the v2 score to meet
the absolute threshold and to stay within parity tolerance of v1.

Registry: nightly-amd-8-gpu-mi35x-glm52-dsa-topk-v2 suite
"""

import os
import resource
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=7200,
    suite="nightly-amd-8-gpu-mi35x-glm52-dsa-topk-v2",
    nightly=True,
)

GLM52_FP8_MODEL_ID = "zai-org/GLM-5.2-FP8"
SERVER_LAUNCH_TIMEOUT = 5400

GSM8K_ACCURACY_THRESHOLD = 0.90
GSM8K_PARITY_TOLERANCE = 0.02
GSM8K_NUM_EXAMPLES = None
GSM8K_NUM_SHOTS = 8
GSM8K_NUM_THREADS = 512


def _raise_nofile_limit() -> None:
    """GSM8K with high parallelism can exceed the default soft nofile=1024."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65535 if hard == resource.RLIM_INFINITY else min(hard, 65535)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


class TestGLM52DSATopKV2GSM8KMI35x(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _raise_nofile_limit()
        cls.model = GLM52_FP8_MODEL_ID
        cls.base_url = DEFAULT_URL_FOR_TEST

    def _launch_and_score(self, use_topk_v2: str) -> float:
        env = dict(os.environ, SGLANG_OPT_USE_TOPK_V2=use_topk_v2)
        process = popen_launch_server(
            model=self.model,
            base_url=self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            env=env,
            other_args=[
                "--tp",
                "4",
                "--trust-remote-code",
                "--attention-backend",
                "dsa",
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--dsa-topk-backend",
                "sgl-kernel",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--context-length",
                "65536",
                "--mem-fraction-static",
                "0.85",
                "--watchdog-timeout",
                "1200",
            ],
        )
        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                api="completion",
                num_examples=GSM8K_NUM_EXAMPLES,
                num_shots=GSM8K_NUM_SHOTS,
                num_threads=GSM8K_NUM_THREADS,
                max_tokens=512,
                temperature=0.0,
            )
            metrics = run_eval(args)
            print(f"topk_v2={use_topk_v2} {metrics=}", flush=True)
            return metrics["score"]
        finally:
            kill_process_tree(process.pid)

    def test_topk_v1_vs_v2_accuracy(self):
        score_v1 = self._launch_and_score("0")
        score_v2 = self._launch_and_score("1")

        if is_in_ci():
            write_github_step_summary(
                "### GLM-5.2-FP8 DSA top-k v2 GSM8K 8-shot (MI350X, TP=4)\n\n"
                "| Variant | Examples | Shots | Score |\n"
                "| ------- | -------- | ----- | ----- |\n"
                f"| topk v1 (baseline) | full split | {GSM8K_NUM_SHOTS} | {score_v1:.3f} |\n"
                f"| topk v2 (fused) | full split | {GSM8K_NUM_SHOTS} | {score_v2:.3f} |\n"
            )

        self.assertGreaterEqual(score_v2, GSM8K_ACCURACY_THRESHOLD)
        self.assertGreaterEqual(
            score_v2,
            score_v1 - GSM8K_PARITY_TOLERANCE,
            f"topk v2 regressed vs v1: {score_v1:.3f} -> {score_v2:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
