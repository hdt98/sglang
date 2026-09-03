# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only GLM5-Next Speculative Decoding."""

import logging

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper

logger = logging.getLogger(__name__)


class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    _NEXTN_SPEC_WEIGHT_NAMES = ("shared_head.norm", "eh_proj", "enorm", "hnorm")

    @classmethod
    def get_hf_to_sglang_mapper(cls, config) -> WeightsMapper:
        text_config = getattr(config, "text_config", config)
        return WeightsMapper(
            orig_to_new_substr={
                f"model.layers.{text_config.num_hidden_layers}": "model.decoder",
            },
        )

    @classmethod
    def _map_mtp_ckpt_name(cls, name: str, layer_prefix: str) -> str:
        if any(part in name for part in cls._NEXTN_SPEC_WEIGHT_NAMES):
            return name.replace(layer_prefix, "model", 1)
        return name.replace(layer_prefix, "model.decoder", 1)

    def _resolve_nextn_quant_config(self, config, quant_config):
        """Remap checkpoint-declared NextN exclusions to runtime module names.

        NextN weights are stored under ``model.layers.<num_hidden_layers>`` but
        instantiated under ``model`` and ``model.decoder``. Quark exclusions
        use checkpoint names, so leaving them unmapped allocates quantized
        parameters for BF16 V29 tensors and makes weight loading fail.
        """
        raw_quant_config = getattr(config, "quantization_config", None) or {}
        if hasattr(raw_quant_config, "to_dict"):
            raw_quant_config = raw_quant_config.to_dict()
        ignored = (
            raw_quant_config.get("ignore", [])
            if isinstance(raw_quant_config, dict)
            else []
        )
        nextn_layer_pattern = f"model.layers.{config.num_hidden_layers}.*"
        if nextn_layer_pattern in ignored:
            logger.warning(
                "GLM5 NextN layer %s is checkpoint-declared unquantized; "
                "using BF16 draft modules",
                nextn_layer_pattern,
            )
            return None

        if quant_config is None or quant_config.get_name() != "quark":
            return quant_config

        layer_prefix = f"model.layers.{config.num_hidden_layers}"
        mapped_layer_prefix = self.get_hf_to_sglang_mapper(config)._map_name(
            layer_prefix
        )

        layer_quant_config = quant_config.quant_config.get("layer_quant_config")
        if layer_quant_config:
            quant_config.quant_config["layer_quant_config"] = {
                (
                    self._map_mtp_ckpt_name(pattern, layer_prefix)
                    if pattern.startswith(layer_prefix + ".")
                    else self._map_mtp_ckpt_name(pattern, mapped_layer_prefix)
                    if pattern.startswith(mapped_layer_prefix + ".")
                    and any(
                        part in pattern for part in self._NEXTN_SPEC_WEIGHT_NAMES
                    )
                    else pattern
                ): pattern_config
                for pattern, pattern_config in layer_quant_config.items()
            }

        raw_mtp_excluded = [
            name
            for name in quant_config.exclude_layers
            if name.startswith(layer_prefix + ".")
        ]
        mapped_mtp_excluded = [
            name
            for name in quant_config.exclude_layers
            if name.startswith(mapped_layer_prefix + ".")
        ]
        if not raw_mtp_excluded and not mapped_mtp_excluded:
            return quant_config

        names = set(quant_config.exclude_layers)
        for name in raw_mtp_excluded:
            names.add(self._map_mtp_ckpt_name(name, layer_prefix))
        for name in mapped_mtp_excluded:
            if any(part in name for part in self._NEXTN_SPEC_WEIGHT_NAMES):
                names.add(self._map_mtp_ckpt_name(name, mapped_layer_prefix))

        if any(
            ".mlp.experts." in name
            for name in raw_mtp_excluded + mapped_mtp_excluded
        ):
            names.add("model.decoder.mlp.experts")

        import copy

        quant_config = copy.copy(quant_config)
        quant_config.exclude_layers = list(names)
        return quant_config

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(
            getattr(config, "text_config", config),
            quant_config=quant_config,
            prefix=prefix,
        )

    def load_weights(self, weights):
        if not hasattr(self, "fuse_qkv_a_proj"):
            self.fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None
        layer_id = self.config.num_hidden_layers
        layer_prefixes = (
            f"model.layers.{layer_id}.",
            f"model.language_model.layers.{layer_id}.",
        )
        nextn_weights = (
            (name, weight)
            for name, weight in weights
            if name.startswith(layer_prefixes)
        )
        return Glm5NextForConditionalGeneration.load_weights(
            self, nextn_weights, is_nextn=True
        )


EntryClass = [Glm5NextForConditionalGenerationNextN]
