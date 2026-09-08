"""CPU regression tests for the DSA kpool write plan and paged-MQA selector.

Two HIP-path invariants must hold without CUDA:

1. update_kpool_write_plan executes on non-CUDA devices: the layout gate no
   longer short-circuits on is_cuda, so the write-plan kernel launch is
   reached with layout-enabled inputs.
2. The paged-MQA backend selector reads the head count from axis 1 of the 3D
   q tensor ([num_q, num_heads, head_dim]), not axis 2.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.layers.attention.dsa import kpool_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeWritePlan:
    """Stands in for the real write-plan dataclass.

    The kernel launch is mocked, so every output buffer the plan is asked for
    can be an empty tensor. Only the two optional post-launch fields must be
    None so the trailing effective-n and schedule updates are skipped.
    """

    def __init__(self):
        self.effective_n_per_batch = None
        self.pool_schedule_metadata = None

    def __getattr__(self, name):
        return torch.empty(0)


def _decode_forward_mode():
    class _FakeForwardMode:
        def is_target_verify(self):
            return False

        def is_decode_or_idle(self):
            return True

        def is_draft_extend_v2(self):
            return False

    return _FakeForwardMode()


class TestKpoolWritePlanNonCuda(CustomTestCase):
    def test_update_kpool_write_plan_executes_without_cuda(self):
        # Layout-enabled inputs: the only former non-CUDA blocker was the
        # removed "or not is_cuda()" clause, so the kernel must be reached.
        self.assertTrue(kpool_plan._is_kpool_layout_enabled(8, 64))
        with mock.patch.object(
            kpool_plan, "update_kpool_write_plan_cuda_graph"
        ) as kernel:
            kpool_plan.update_kpool_write_plan(
                mock.Mock(kpool_write_plan=_FakeWritePlan()),
                write_start=torch.zeros(2, dtype=torch.int32),
                req_pool_indices=torch.zeros(2, dtype=torch.int64),
                real_page_table=torch.zeros(2, 4, dtype=torch.int32),
                pool_size=8,
                real_page_size=64,
                num_draft_tokens=1,
                forward_mode=_decode_forward_mode(),
                slots_per_page=1,
            )
        self.assertEqual(kernel.call_count, 1)
        kwargs = kernel.call_args.kwargs
        self.assertEqual(kwargs["pool_size"], 8)
        self.assertEqual(kwargs["num_draft_tokens"], 1)
        self.assertEqual(kwargs["slots_per_page"], 1)

    def test_update_kpool_write_plan_skips_disabled_layout(self):
        with mock.patch.object(
            kpool_plan, "update_kpool_write_plan_cuda_graph"
        ) as kernel:
            kpool_plan.update_kpool_write_plan(
                mock.Mock(kpool_write_plan=_FakeWritePlan()),
                write_start=torch.zeros(2, dtype=torch.int32),
                req_pool_indices=torch.zeros(2, dtype=torch.int64),
                real_page_table=torch.zeros(2, 4, dtype=torch.int32),
                pool_size=1,
                real_page_size=64,
                num_draft_tokens=1,
                forward_mode=_decode_forward_mode(),
                slots_per_page=1,
            )
        self.assertEqual(kernel.call_count, 0)


class TestPagedMqaSelectorHeadAxis(CustomTestCase):
    def test_selector_reads_heads_from_axis_1_of_3d_q(self):
        selector = (
            dsa_indexer_kpool.IndexerKPool._should_use_tilelang_paged_mqa_logits
        )
        # 3D q: [num_q, num_heads, head_dim]. heads=32 at axis 1 is excluded,
        # while head_dim=128 at axis 2 is not - a selector reading axis 2
        # would wrongly enable the tilelang path here.
        q_excluded_heads = torch.zeros(4, 32, 128, dtype=torch.float16)
        q_allowed_heads = torch.zeros(4, 7, 128, dtype=torch.float16)

        with mock.patch.object(
            dsa_indexer_kpool, "is_cuda", return_value=True
        ), mock.patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ):
            self.assertFalse(selector(q_excluded_heads))
            self.assertTrue(selector(q_allowed_heads))

        # Arch gate: a non-Hopper major never selects the tilelang path.
        with mock.patch.object(
            dsa_indexer_kpool, "is_cuda", return_value=True
        ), mock.patch(
            "torch.cuda.get_device_capability", return_value=(10, 0)
        ):
            self.assertFalse(selector(q_allowed_heads))

        # Without CUDA the selector short-circuits regardless of shape.
        self.assertFalse(selector(q_allowed_heads))


if __name__ == "__main__":
    unittest.main()
