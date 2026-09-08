import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHybridMLARankMapping(unittest.TestCase):
    def test_tp_mismatch_uses_one_real_prefill_writer(self):
        manager = object.__new__(CommonKVManager)
        manager.attn_tp_size = 2
        manager.kv_args = SimpleNamespace(engine_rank=0)
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = True
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.enable_all_cp_ranks_for_transfer = False
        manager.pp_size = 1
        manager.pp_rank = 0
        info = SimpleNamespace(
            attn_tp_size=4,
            attn_cp_size=1,
            pp_size=1,
            enable_dsa_cache_layer_split=False,
        )

        manager._resolve_rank_mapping(info)

        self.assertEqual(info.target_tp_rank, 0)
        self.assertEqual(info.target_tp_ranks, [0, 1])
        self.assertEqual(info.required_dst_info_num, 1)
        self.assertEqual(info.required_prefill_response_num, 1)


if __name__ == "__main__":
    unittest.main()
