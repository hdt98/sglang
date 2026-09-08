"""CPU-only tests for UnifiedRadixCache request lock lifecycle."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import IncLockRefResult, InsertResult
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUnifiedRadixLockRefScenarios(unittest.TestCase):
    def test_pd_checkpoint_behind_locked_prefix_does_not_rewind_ownership(self):
        protected_len, input_len = 233280, 237163
        for checkpoint_len in (None, 231808, 233280, 237120):
            with self.subTest(checkpoint_len=checkpoint_len):
                cache = object.__new__(UnifiedRadixCache)
                cache.session = MagicMock()
                cache.session.try_cache_unfinished_req.return_value = False
                cache.disable = False
                cache.enable_mamba_extra_buffer = True
                cache.tree_core = SimpleNamespace(is_eagle=False, page_size=64)
                pool = MagicMock(mamba_ckpt_pool=None)
                pool.req_to_token = torch.arange(input_len).reshape(1, -1)
                pool.mamba_allocator.alloc.return_value = torch.tensor([99])
                cache.req_to_token_pool = pool
                mamba = object.__new__(MambaComponent)
                mamba.cache = cache
                cache._components_tuple = (mamba,)
                cache._dec_req_lock = MagicMock()
                cache.inc_lock_ref = MagicMock(return_value=IncLockRefResult())
                matched_len = checkpoint_len or 0
                cache.insert = MagicMock(
                    return_value=InsertResult(prefix_len=min(protected_len, matched_len))
                )
                cache.match_prefix = MagicMock(
                    return_value=SimpleNamespace(
                        device_indices=torch.arange(matched_len), last_device_node=8
                    )
                )
                token_ids = array("q", range(input_len))
                req = SimpleNamespace(
                    get_fill_ids=lambda: token_ids,
                    extra_key=None,
                    cache_salt=None,
                    last_node=7,
                    kv=SimpleNamespace(
                        req_pool_idx=0,
                        cache_protected_len=protected_len,
                        mamba_last_track_seqlen=checkpoint_len,
                        pd_prompt_checkpoint_seqlen=checkpoint_len,
                        mamba_ping_pong_track_buffer=torch.tensor([42]),
                    ),
                )

                def donate(req, new_slot):
                    old_slot = req.kv.mamba_ping_pong_track_buffer.clone()
                    req.kv.mamba_ping_pong_track_buffer[:] = new_slot
                    return old_slot

                pool.donate_mamba_ping_pong_slot.side_effect = donate
                cache.cache_unfinished_req(req)

                self.assertTrue(torch.equal(req.prefix_indices, pool.req_to_token[0]))
                self.assertIsNone(req.kv.mamba_last_track_seqlen)
                self.assertEqual(req.kv.pd_prompt_checkpoint_seqlen, checkpoint_len)
                if checkpoint_len is None or checkpoint_len < protected_len:
                    self.assertEqual(req.kv.cache_protected_len, protected_len)
                    self.assertEqual(req.last_node, 7)
                    cache.insert.assert_not_called()
                    cache._dec_req_lock.assert_not_called()
                    cache.inc_lock_ref.assert_not_called()
                    pool.write.assert_not_called()
                    pool.mamba_allocator.alloc.assert_not_called()
                    pool.donate_mamba_ping_pong_slot.assert_not_called()
                    pool.mamba_allocator.free.assert_not_called()
                    self.assertEqual(req.kv.mamba_ping_pong_track_buffer.tolist(), [42])
                    if checkpoint_len is not None:
                        cache.session.try_cache_finished_req.return_value = False
                        cache.enable_session_radix_cache = False
                        cache.free_kv_row = MagicMock()
                        pool.get_mamba_ping_pong_keep_idx.return_value = 0
                        req.origin_input_ids, req.output_ids = token_ids, array("q")
                        req.swa_prefix_lock_released = False
                        cache.cache_finished_req(req, kv_len_to_handle=input_len)
                        params = cache.insert.call_args.args[0]
                        self.assertEqual(params.mamba_value.tolist(), [42])
                        self.assertEqual(len(params.key), checkpoint_len)
                        cache.free_kv_row.assert_called_once_with(
                            req.kv,
                            [(checkpoint_len, checkpoint_len), (protected_len, input_len)],
                        )
                        pool.free_mamba_cache.assert_called_once_with(
                            req, mamba_ping_pong_track_buffer_to_keep=0
                        )
                else:
                    self.assertEqual(req.kv.cache_protected_len, checkpoint_len)
                    self.assertEqual(req.last_node, 8)
                    cache.insert.assert_called_once()
                    cache._dec_req_lock.assert_called_once_with(req)
                    cache.inc_lock_ref.assert_called_once()
                    pool.donate_mamba_ping_pong_slot.assert_called_once()

    def test_no_insert_without_last_node_skips_lock_release(self):
        cache = object.__new__(UnifiedRadixCache)
        cache.session = MagicMock()
        cache.session.try_cache_finished_req.return_value = False
        cache.disable = False
        cache.req_to_token_pool = MagicMock()
        cache.req_to_token_pool.req_to_token = torch.arange(8).reshape(1, 8)
        cache.free_kv_row = MagicMock()
        cache._dec_req_lock = MagicMock()
        cache._components_tuple = ()
        cache.enable_session_radix_cache = False

        kv = SimpleNamespace(req_pool_idx=0, cache_protected_len=0)
        req = SimpleNamespace(
            origin_input_ids=array("q", [1, 2, 3]),
            output_ids=array("q"),
            kv=kv,
            last_node=None,
            swa_prefix_lock_released=False,
        )

        cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=3)

        cache.free_kv_row.assert_called_once_with(kv, [(0, 3)])
        cache._dec_req_lock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
