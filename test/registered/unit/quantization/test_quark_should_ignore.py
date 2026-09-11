from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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
