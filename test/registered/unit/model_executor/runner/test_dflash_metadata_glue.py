from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
    bind_metadata_glue_static_inputs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hip_backend(
    *,
    is_dspark: bool = True,
    is_draft_worker: bool = False,
    row_limit: int | None = 4096,
    draft_tokens: int = 6,
):
    backend = DeepseekV4HipRadixBackend.__new__(DeepseekV4HipRadixBackend)
    backend.is_dspark = is_dspark
    backend.is_draft_worker = is_draft_worker
    backend._fp4_graph_row_limit = row_limit
    backend.target_verify_num_draft_tokens = draft_tokens
    return backend


def test_dflash_backend_requires_explicit_metadata_glue_opt_in():
    assert not AttentionBackend().supports_dflash_metadata_glue_graph(
        ForwardMode.TARGET_VERIFY, 8
    )


def test_hip_dspark_metadata_glue_eligibility_is_per_tier():
    backend = _hip_backend(row_limit=48, draft_tokens=6)

    assert backend.supports_dflash_metadata_glue_graph(
        ForwardMode.TARGET_VERIFY, 8
    )
    assert not backend.supports_dflash_metadata_glue_graph(
        ForwardMode.TARGET_VERIFY, 9
    )
    assert not backend.supports_dflash_metadata_glue_graph(ForwardMode.DECODE, 8)

    backend.is_draft_worker = True
    assert not backend.supports_dflash_metadata_glue_graph(
        ForwardMode.TARGET_VERIFY, 8
    )


def test_raw_verify_deferral_rejects_ragged_and_oversized_batches():
    backend = _hip_backend(row_limit=48, draft_tokens=6)
    out_cache_loc = torch.zeros(48, dtype=torch.int64)

    assert backend._can_defer_target_verify_metadata(
        batch_size=8,
        out_cache_loc=out_cache_loc,
        ragged_layout=None,
        use_prefill_cuda_graph=True,
    )
    assert not backend._can_defer_target_verify_metadata(
        batch_size=9,
        out_cache_loc=out_cache_loc,
        ragged_layout=None,
        use_prefill_cuda_graph=True,
    )
    assert not backend._can_defer_target_verify_metadata(
        batch_size=8,
        out_cache_loc=out_cache_loc,
        ragged_layout=object(),
        use_prefill_cuda_graph=True,
    )
    assert not backend._can_defer_target_verify_metadata(
        batch_size=8,
        out_cache_loc=None,
        ragged_layout=None,
        use_prefill_cuda_graph=True,
    )


def test_metadata_glue_binds_pointer_stable_cache_location_slot():
    live = torch.tensor([7, 8, 9], dtype=torch.int64)
    static = torch.tensor([17, 18, 19, 20], dtype=torch.int64)
    fb_view = SimpleNamespace(out_cache_loc=live)
    buffers = SimpleNamespace(out_cache_loc=static)

    result = bind_metadata_glue_static_inputs(fb_view, buffers, num_tokens=3)

    assert result is fb_view
    assert result.out_cache_loc.data_ptr() == static.data_ptr()
    assert result.out_cache_loc.tolist() == [17, 18, 19]


def test_runner_delegates_dflash_eligibility_to_attention_backend():
    backend = _hip_backend(row_limit=48, draft_tokens=6)
    runner = SimpleNamespace(
        model_runner=SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True)
        ),
        capture_forward_mode=ForwardMode.TARGET_VERIFY,
    )

    eligible = DecodeCudaGraphRunner._metadata_glue_backend_eligible
    assert eligible(runner, backend, 8)
    assert not eligible(runner, backend, 9)


def test_runner_rejects_layout_attached_replay_even_when_ragged_mode_is_off():
    backend = _hip_backend(row_limit=48, draft_tokens=6)
    runner = SimpleNamespace(
        _metadata_glue=SimpleNamespace(disabled=False),
        enable_two_batch_overlap=False,
        enable_pdmux=False,
        model_runner=SimpleNamespace(
            lora_manager=None,
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True),
        ),
        capture_forward_mode=ForwardMode.TARGET_VERIFY,
    )
    runner._metadata_glue_backend_eligible = (
        DecodeCudaGraphRunner._metadata_glue_backend_eligible.__get__(runner)
    )

    eligible = DecodeCudaGraphRunner._metadata_glue_replay_eligible
    assert eligible(
        runner,
        backend,
        batch_size=8,
        raw_batch_size=8,
        attached_ragged_layout=None,
    )
    assert not eligible(
        runner,
        backend,
        batch_size=8,
        raw_batch_size=8,
        attached_ragged_layout=object(),
    )
