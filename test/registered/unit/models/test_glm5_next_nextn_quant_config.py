"""Unit tests for GLM-5 NextN draft quantization exclusion."""

import unittest
from types import SimpleNamespace

from sglang.srt.models.glm5_next_nextn import (
    Glm5NextForConditionalGenerationNextN,
)


class TestGlm5NextNextnQuantConfig(unittest.TestCase):
    def test_concrete_exclude_entry_disables_draft_quantization(self):
        config = SimpleNamespace(
            num_hidden_layers=45,
            quantization_config={
                "exclude": ["model.layers.45.eh_proj"],
                "ignore": [],
            },
        )
        quant_config = SimpleNamespace(get_name=lambda: "quark")
        model = object.__new__(Glm5NextForConditionalGenerationNextN)

        self.assertIsNone(
            model._resolve_nextn_quant_config(config, quant_config)
        )

    def test_wildcard_ignore_entry_disables_draft_quantization(self):
        config = SimpleNamespace(
            num_hidden_layers=45,
            quantization_config={
                "exclude": [],
                "ignore": ["model.layers.45.*"],
            },
        )
        quant_config = SimpleNamespace(get_name=lambda: "quark")
        model = object.__new__(Glm5NextForConditionalGenerationNextN)

        self.assertIsNone(
            model._resolve_nextn_quant_config(config, quant_config)
        )


if __name__ == "__main__":
    unittest.main()
