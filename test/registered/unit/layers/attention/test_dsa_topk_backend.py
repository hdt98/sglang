import sys

import pytest
import torch

import sglang.srt.layers.attention.dsa.dsa_topk_backend as dsa_topk_backend
from sglang.srt.layers.attention.dsa.dsa_topk_backend import _topk_unfused
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _reference_topk(score, lengths, topk, row_starts):
    result = torch.full((score.shape[0], topk), -1, dtype=torch.int32)
    for row in range(score.shape[0]):
        start = int(row_starts[row])
        end = start + int(lengths[row])
        valid = score[row, start:end]
        count = min(topk, valid.numel())
        if count:
            indices = torch.topk(valid, count).indices.to(torch.int32)
            result[row, :count] = indices
    return result


@pytest.mark.parametrize(
    "batch_size,topk,hip,expected_calls",
    [
        (4096, 16, True, 1),
        (4097, 8, True, 2),
        (4097, 8, False, 1),
    ],
    ids=["hip-chunk-boundary", "hip-first-chunked-row", "cuda-unchanged"],
)
def test_topk_unfused_chunks_wide_hip_batches_only(
    monkeypatch, batch_size, topk, hip, expected_calls
):
    monkeypatch.setattr(dsa_topk_backend, "_is_hip", hip)
    generator = torch.Generator().manual_seed(0)
    score = torch.randn(batch_size, 16, generator=generator)
    lengths = torch.randint(
        0, 17, (batch_size,), generator=generator, dtype=torch.int32
    )
    row_starts = torch.randint(
        0, 5, (batch_size,), generator=generator, dtype=torch.int32
    )
    calls = []

    def counted_topk(*args, **kwargs):
        calls.append(1)
        return torch.topk(*args, **kwargs)

    actual = _topk_unfused(
        score, lengths, topk, row_starts=row_starts, topk_op=counted_topk
    )
    expected = _reference_topk(score, lengths, topk, row_starts)

    assert len(calls) == expected_calls
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
