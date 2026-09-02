from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.models.glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_glm5_next_dflash_contracts_mhc_hidden_state():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = True

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    residual = torch.full_like(hidden_states, 2)

    actual = model._prepare_aux_hidden_state(hidden_states, residual)
    expected = (hidden_states + residual).unflatten(-1, (4, -1)).mean(dim=-2)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3)


def test_glm5_next_eagle_capture_keeps_mhc_hidden_state():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = False

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    residual = torch.full_like(hidden_states, 2)

    actual = model._prepare_aux_hidden_state(hidden_states, residual)

    torch.testing.assert_close(actual, hidden_states + residual)


def test_glm5_next_dflash_contracts_combined_mhc_layer_output():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = True

    combined_hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    actual = model._prepare_aux_hidden_state(combined_hidden_states, residual=None)
    expected = combined_hidden_states.unflatten(-1, (4, -1)).mean(dim=-2)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3)


def test_glm5_next_dflash_maps_target_layers_to_capture_points():
    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.model = SimpleNamespace(dflash_capture=False, layers_to_capture=[])
    model.capture_aux_hidden_states = False

    model.set_dflash_layers_to_capture([5, 14, 24, 33, 42])

    assert model.capture_aux_hidden_states
    assert model.model.dflash_capture
    assert model.model.layers_to_capture == [6, 15, 25, 34, 43]


def test_glm5_next_gathers_dflash_aux_states_after_legacy_prefill_cp():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.enable_a2a_moe = False
    model.cp_size = 4
    aux_hidden_states = [
        torch.full((2, 3), float(layer_id)) for layer_id in range(5)
    ]
    forward_batch = SimpleNamespace(extend_seq_lens_cpu=[3, 2])

    with patch(
        "sglang.srt.models.glm5_next.cp_plain_all_gather",
        side_effect=lambda hidden, cp_size: hidden.repeat(cp_size, 1),
    ) as gather:
        actual = model._gather_aux_hidden_states_after_prefill_cp(
            aux_hidden_states, forward_batch
        )

    assert gather.call_count == len(aux_hidden_states)
    assert all(hidden.shape == (5, 3) for hidden in actual)
    for layer_id, hidden in enumerate(actual):
        torch.testing.assert_close(
            hidden, torch.full((5, 3), float(layer_id))
        )


def test_glm5_next_does_not_regather_a2a_aux_states_after_prefill_cp():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.enable_a2a_moe = True
    model.cp_size = 4
    aux_hidden_states = [torch.ones((8, 3))]
    forward_batch = SimpleNamespace(extend_seq_lens_cpu=[3, 2])

    with patch("sglang.srt.models.glm5_next.cp_plain_all_gather") as gather:
        actual = model._gather_aux_hidden_states_after_prefill_cp(
            aux_hidden_states, forward_batch
        )

    gather.assert_not_called()
    assert actual[0].shape == (5, 3)
    torch.testing.assert_close(actual[0], torch.ones((5, 3)))


def test_glm5_next_rejects_short_prefill_cp_aux_state():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.enable_a2a_moe = True
    model.cp_size = 4
    forward_batch = SimpleNamespace(extend_seq_lens_cpu=[3, 2])

    with pytest.raises(RuntimeError, match="gathered=4, logical=5"):
        model._gather_aux_hidden_states_after_prefill_cp(
            [torch.ones((4, 3))], forward_batch
        )
