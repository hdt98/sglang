"""Regression tests for avoiding a second GLM text-only prompt tokenization."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGlm53TextTokenIds(unittest.TestCase):
    def setUp(self):
        self.chat = OpenAIServingChat.__new__(OpenAIServingChat)
        self.chat.is_gpt_oss = False
        self.chat.chat_encoding_spec = None
        self.chat._tokenizer_auto_adds_specials = False
        self.chat.default_sampling_params = {}
        self.chat.template_manager = SimpleNamespace(chat_template_name=None)
        self.chat.tokenizer_manager = SimpleNamespace(
            model_config=SimpleNamespace(
                is_multimodal=True,
                hf_config=SimpleNamespace(model_type="glm5_next"),
            )
        )
        self.chat.extract_custom_labels = Mock(return_value=None)
        self.chat.extract_routed_dp_rank_from_header = Mock(return_value=None)
        self.chat.extract_routing_key = Mock(return_value=None)
        self.processed = MessageProcessingResult(
            prompt="rendered prompt",
            prompt_ids=[11, 22, 33],
            stop=["<stop>"],
            image_data=None,
            video_data=None,
            audio_data=None,
            modalities=[],
        )
        self.chat._process_messages = Mock(return_value=self.processed)
        self.request = ChatCompletionRequest(
            model="OneNexus/GLM-5.3-Flash-MXFP4",
            messages=[{"role": "user", "content": "Explain the code."}],
            temperature=0.6,
            max_tokens=128,
            rid="text-id-probe",
            session_id="session-1",
            cache_salt="salt-1",
            bootstrap_host="127.0.0.1",
            bootstrap_port=32500,
            bootstrap_room=44331,
            disagg_prefill_dp_rank=0,
        )

    def convert(self):
        adapted, request = self.chat._convert_to_internal_request(self.request)
        self.assertIs(request, self.request)
        return adapted

    def assert_text_path(self):
        adapted = self.convert()
        self.assertEqual(adapted.text, self.processed.prompt)
        self.assertIsNone(adapted.input_ids)
        return adapted

    def test_glm_text_reuses_ids_and_preserves_pd_request_controls(self):
        adapted = self.convert()
        self.assertEqual(adapted.input_ids, self.processed.prompt_ids)
        self.assertIsNone(adapted.text)
        for field in (
            "rid", "session_id", "cache_salt", "bootstrap_host",
            "bootstrap_port", "bootstrap_room", "disagg_prefill_dp_rank",
        ):
            self.assertEqual(getattr(adapted, field), getattr(self.request, field))
        self.assertEqual(adapted.sampling_params["temperature"], 0.6)
        self.assertEqual(adapted.sampling_params["max_new_tokens"], 128)
        self.assertEqual(adapted.sampling_params["stop"], ["<stop>"])

    def test_streaming_uses_the_same_token_ids(self):
        self.request.stream = True
        adapted = self.convert()
        self.assertEqual(adapted.input_ids, self.processed.prompt_ids)
        self.assertTrue(adapted.stream)

    def test_media_inputs_keep_text_and_payload(self):
        for field in ("image_data", "video_data", "audio_data", "modalities"):
            with self.subTest(field=field):
                setattr(self.processed, field, ["media-placeholder"])
                adapted = self.assert_text_path()
                self.assertEqual(getattr(adapted, field), ["media-placeholder"])
                setattr(self.processed, field, [] if field == "modalities" else None)

    def test_other_multimodal_models_keep_existing_path(self):
        self.chat.tokenizer_manager.model_config.hf_config.model_type = "qwen3_vl"
        self.assert_text_path()

    def test_custom_conversation_template_keeps_existing_path(self):
        self.chat.template_manager.chat_template_name = "custom-template"
        self.assert_text_path()

    def test_special_token_inserting_tokenizer_keeps_existing_path(self):
        self.chat._tokenizer_auto_adds_specials = True
        self.assert_text_path()

    def test_empty_or_string_prompt_ids_keep_text(self):
        for value in ([], "unencoded prompt"):
            with self.subTest(prompt_ids=value):
                self.processed.prompt_ids = value
                self.assert_text_path()

    def test_custom_encoder_keeps_existing_path(self):
        self.chat.chat_encoding_spec = "custom-encoder"
        self.assert_text_path()

    def test_existing_inkling_and_kimi_id_routes_are_unchanged(self):
        self.processed.image_data = ["image-placeholder"]
        for spec in ("inkling", "kimi_k3"):
            with self.subTest(spec=spec):
                self.chat.chat_encoding_spec = spec
                self.assertEqual(self.convert().input_ids, self.processed.prompt_ids)

    def test_explicit_request_ids_still_take_precedence(self):
        self.request.input_ids = [11, 22, 33]
        self.chat.tokenizer_manager.model_config.hf_config.model_type = "other"
        self.assertEqual(self.convert().input_ids, self.request.input_ids)


if __name__ == "__main__":
    unittest.main()
