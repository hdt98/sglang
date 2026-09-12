import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestQuarkShouldIgnoreLayer(CustomTestCase):
    def test_exact_fused_exclusion_is_not_expanded_away(self):
        from sglang.srt.layers.quantization.quark.utils import should_ignore_layer

        self.assertTrue(
            should_ignore_layer(
                "visual.blocks.0.attn.qkv_proj",
                ignore=["visual.blocks.0.attn.qkv_proj"],
                fused_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
            )
        )

    def test_fused_shards_still_control_scheme(self):
        from sglang.srt.layers.quantization.quark.utils import should_ignore_layer

        with self.assertRaises(ValueError):
            should_ignore_layer(
                "model.layers.0.self_attn.qkv_proj",
                ignore=[
                    "model.layers.0.self_attn.k_proj",
                    "model.layers.0.self_attn.v_proj",
                ],
                fused_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
            )


class TestGlm5NextNextNQuantConfig(CustomTestCase):
    def test_quark_exclude_is_unquantized_nextn(self):
        from types import SimpleNamespace

        from sglang.srt.models.glm5_next_nextn import (
            Glm5NextForConditionalGenerationNextN,
        )

        config = SimpleNamespace(
            num_hidden_layers=45,
            quantization_config={
                "quant_method": "quark",
                "exclude": ["model.layers.45.*"],
            },
        )

        self.assertIsNone(
            Glm5NextForConditionalGenerationNextN._resolve_nextn_quant_config(
                None, config, object()
            )
        )

    def test_expanded_quark_exclude_is_unquantized_nextn(self):
        from types import SimpleNamespace

        from sglang.srt.models.glm5_next_nextn import (
            Glm5NextForConditionalGenerationNextN,
        )

        config = SimpleNamespace(
            num_hidden_layers=45,
            quantization_config={
                "quant_method": "quark",
                "exclude": [
                    "model.layers.45.self_attn.q_a_proj",
                    "model.layers.45.self_attn.kv_a_proj_with_mqa",
                ],
            },
        )

        self.assertIsNone(
            Glm5NextForConditionalGenerationNextN._resolve_nextn_quant_config(
                None, config, object()
            )
        )


if __name__ == "__main__":
    unittest.main()
