import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.attention.aiter_backend import (
    AiterAttnBackend,
    _parse_cprr_verify_split_caps,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAiterCPRRVerifySplitCaps(unittest.TestCase):
    def setUp(self):
        envs.SGLANG_AITER_MLA_CPRR_VERIFY_SPLIT_CAPS.clear()
        self.addCleanup(envs.SGLANG_AITER_MLA_CPRR_VERIFY_SPLIT_CAPS.clear)

    def test_unset_table_keeps_the_planned_split_count(self):
        backend = AiterAttnBackend.__new__(AiterAttnBackend)
        backend.cprr_verify_split_caps = _parse_cprr_verify_split_caps(None, 4)
        backend.max_split_per_batch = 32

        self.assertEqual(backend.cprr_verify_split_caps, {})
        self.assertEqual(backend._cprr_verify_split_cap(16), 32)

        num_kv_splits = backend._cprr_verify_num_kv_splits(
            16, use_asm_cprr_verify=False
        )
        self.assertEqual(num_kv_splits, 32)

    def test_table_selects_caps_by_verify_tokens_and_batch(self):
        with envs.SGLANG_AITER_MLA_CPRR_VERIFY_SPLIT_CAPS.override(
            '{"4":{"8":24,"16":12},"6":{"16":6}}'
        ):
            caps = _parse_cprr_verify_split_caps(
                envs.SGLANG_AITER_MLA_CPRR_VERIFY_SPLIT_CAPS.get(),
                num_draft_tokens=4,
            )

        backend = AiterAttnBackend.__new__(AiterAttnBackend)
        backend.cprr_verify_split_caps = caps
        backend.max_split_per_batch = 32

        self.assertEqual(backend._cprr_verify_split_cap(8), 24)
        self.assertEqual(backend._cprr_verify_split_cap(16), 12)
        self.assertEqual(backend._cprr_verify_split_cap(32), 32)
        self.assertEqual(
            backend._cprr_verify_num_kv_splits(
                16, use_asm_cprr_verify=True
            ),
            12,
        )
        self.assertEqual(
            backend._cprr_verify_num_kv_splits(
                16, use_asm_cprr_verify=False
            ),
            32,
        )

    def test_missing_verify_token_shape_keeps_the_default(self):
        caps = _parse_cprr_verify_split_caps(
            {"6": {"16": 6}},
            num_draft_tokens=4,
        )
        backend = AiterAttnBackend.__new__(AiterAttnBackend)
        backend.cprr_verify_split_caps = caps
        backend.max_split_per_batch = 32

        self.assertEqual(caps, {})
        self.assertEqual(backend._cprr_verify_split_cap(16), 32)


if __name__ == "__main__":
    unittest.main()
