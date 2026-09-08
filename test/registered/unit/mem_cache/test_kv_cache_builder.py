from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.kv_cache_builder import (
    _validate_decode_radix_tree_support,
)


@pytest.mark.parametrize(
    ("enabled", "mode", "is_hybrid_ssm"),
    [
        (False, "decode", True),
        (True, "prefill", True),
        (True, "decode", False),
    ],
)
def test_decode_radix_mamba_validation_is_scoped_to_hybrid_decode(
    enabled, mode, is_hybrid_ssm
):
    tree_cache = SimpleNamespace(supports_mamba=lambda: False)

    _validate_decode_radix_tree_support(
        enabled=enabled,
        mode=mode,
        tree_cache=tree_cache,
        is_hybrid_ssm=is_hybrid_ssm,
    )


def test_decode_radix_accepts_mamba_capable_tree():
    tree_cache = SimpleNamespace(supports_mamba=lambda: True)

    _validate_decode_radix_tree_support(
        enabled=True,
        mode="decode",
        tree_cache=tree_cache,
        is_hybrid_ssm=True,
    )


def test_decode_radix_rejects_tree_without_mamba_support():
    tree_cache = SimpleNamespace(supports_mamba=lambda: False)

    with pytest.raises(ValueError, match="requires a radix-cache backend"):
        _validate_decode_radix_tree_support(
            enabled=True,
            mode="decode",
            tree_cache=tree_cache,
            is_hybrid_ssm=True,
        )
