from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "hybrid,tbo", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("dcp_size", [1, 2])
def test_rocm_fp8_prefix_reads_full_attention_page_table(hybrid, tbo, dcp_size):
    slot_indices = torch.tensor([4, 8, 12])
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.forward_metadata = SimpleNamespace(page_table_1_flattened=slot_indices)
    if hybrid:
        wrapper = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        wrapper.full_attn_backend = backend
        backend = wrapper
    if tbo:
        wrapper = TboAttnBackend.__new__(TboAttnBackend)
        wrapper.primary = backend
        backend = wrapper

    kv_a = torch.ones(3, 1, 4, dtype=torch.bfloat16)
    k_pe = torch.ones(3, 1, 2, dtype=torch.bfloat16)
    pool = Mock()
    pool.get_mla_kv_buffer.return_value = kv_a, k_pe
    layer = SimpleNamespace(attn_mha=object())
    with (
        patch.object(forward_mha, "get_attn_backend", return_value=backend),
        patch.object(forward_mha, "get_token_to_kv_pool", return_value=pool),
        patch.object(forward_mha, "_use_aiter_gfx95", True),
        patch.object(
            forward_mha,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=dcp_size > 1, dcp_size=dcp_size),
        ),
        patch.object(
            forward_mha, "filter_dcp_local_kv_indices", return_value=slot_indices
        ) as filter_indices,
    ):
        actual_kv_a, actual_k_pe = (
            forward_mha.DeepseekMHAForwardMixin._get_mla_kv_buffer_from_fp8_for_dsa(
                layer, SimpleNamespace()
            )
        )

    assert filter_indices.call_args.kwargs["kv_indices"] is slot_indices
    assert pool.get_mla_kv_buffer.call_args.args[0] is layer.attn_mha
    torch.testing.assert_close(
        pool.get_mla_kv_buffer.call_args.args[1], slot_indices // dcp_size
    )
    assert pool.get_mla_kv_buffer.call_args.args[2] == torch.bfloat16
    torch.testing.assert_close(actual_kv_a, kv_a.squeeze(1))
    assert actual_k_pe is k_pe
