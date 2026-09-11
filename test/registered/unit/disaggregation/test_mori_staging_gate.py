import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMoriStagingGate(unittest.TestCase):
    @patch(
        "sglang.srt.disaggregation.prefill.get_schedule",
        return_value=SimpleNamespace(chunked_prefill_size=8192),
    )
    @patch(
        "sglang.srt.disaggregation.prefill.get_parallel",
        return_value=SimpleNamespace(enable_prefill_context_parallel=False),
    )
    @patch.object(PrefillBootstrapQueue, "_init_kv_manager")
    def test_mori_staging_enables_without_generic_staging(
        self, _mock_init, _mock_parallel, _mock_schedule
    ):
        with patch.dict(
            os.environ,
            {"SGLANG_MORI_STAGING_BUFFER": "1"},
        ):
            queue = PrefillBootstrapQueue(
                token_to_kv_pool=SimpleNamespace(),
                draft_token_to_kv_pool=None,
                req_to_metadata_buffer_idx_allocator=SimpleNamespace(),
                metadata_buffers=SimpleNamespace(),
                tp_rank=0,
                tp_size=1,
                gpu_id=0,
                bootstrap_port=0,
                gloo_group=SimpleNamespace(),
                max_total_num_tokens=1,
                scheduler=SimpleNamespace(
                    token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
                    tp_worker=SimpleNamespace(
                        model_runner=SimpleNamespace(effective_max_total_num_tokens=1)
                    ),
                ),
                scheduler_stage_metrics=SimpleNamespace(),
                pp_rank=0,
                pp_size=1,
                transfer_backend=TransferBackend.MORI,
            )

        self.assertTrue(queue.enable_staging)


if __name__ == "__main__":
    unittest.main()
