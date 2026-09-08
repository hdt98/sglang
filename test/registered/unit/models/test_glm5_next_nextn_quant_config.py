"""Unit tests for GLM-5 NextN draft quantization exclusion."""

import unittest
from types import SimpleNamespace

from sglang.srt.models.glm5_next_nextn import (
    Glm5NextForConditionalGenerationNextN,
)


class TestGlm5NextNextnQuantConfig(unittest.TestCase):
    def test_concrete_excludes_are_mapped_to_runtime_draft_names(self):
        config = SimpleNamespace(
            num_hidden_layers=45,
            quantization_config={
                "exclude": [
                    "model.layers.45.eh_proj",
                    "model.layers.45.mlp.experts.0.gate_proj",
                ],
                "ignore": [],
            },
        )
        quant_config = SimpleNamespace(
            get_name=lambda: "quark",
            quant_config={
                "layer_quant_config": {
                    "model.layers.45.eh_proj": {"weight": "bf16"},
                    "model.layers.45.mlp.experts.*": {"weight": "bf16"},
                }
            },
            exclude_layers=config.quantization_config["exclude"].copy(),
        )
        model = object.__new__(Glm5NextForConditionalGenerationNextN)

        resolved = model._resolve_nextn_quant_config(config, quant_config)

        self.assertIsNot(resolved, quant_config)
        self.assertIn("model.eh_proj", resolved.exclude_layers)
        self.assertIn("model.decoder.mlp.experts", resolved.exclude_layers)
        self.assertIn(
            "model.eh_proj", resolved.quant_config["layer_quant_config"]
        )
        self.assertNotIn("model.eh_proj", quant_config.exclude_layers)
        self.assertNotIn(
            "model.eh_proj", quant_config.quant_config["layer_quant_config"]
        )

    def test_wildcard_ignore_entry_disables_draft_quantization(self):
        config = SimpleNamespace(
            num_hidden_layers=45,
            quantization_config={
                "exclude": [],
                "ignore": ["model.layers.45.*"],
            },
        )
        quant_config = SimpleNamespace(
            get_name=lambda: "quark",
            quant_config={},
            exclude_layers=[],
        )
        model = object.__new__(Glm5NextForConditionalGenerationNextN)

        self.assertIsNone(
            model._resolve_nextn_quant_config(config, quant_config)
        )


if __name__ == "__main__":
    unittest.main()
