"""Device selection tests for embedding backend."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sanskrit_pipeline.embeddings import DEFAULT_MODEL_ID, DEFAULT_QUERY_INSTRUCTION, TextEmbedder, _resolve_torch_device


class EmbeddingDeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.backends.mps.is_available", return_value=True):
                self.assertEqual(_resolve_torch_device("auto"), "cuda")

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                self.assertEqual(_resolve_torch_device("auto"), "mps")

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                self.assertEqual(_resolve_torch_device("auto"), "cpu")

    def test_explicit_cpu_always_valid(self) -> None:
        self.assertEqual(_resolve_torch_device("cpu"), "cpu")

    def test_explicit_unavailable_device_raises(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                _resolve_torch_device("cuda")

        with patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                _resolve_torch_device("mps")

    def test_sentence_transformer_receives_selected_device(self) -> None:
        with patch("sanskrit_pipeline.embeddings.SentenceTransformer") as mock_st:
            embedder = TextEmbedder(model_id="fake/model", device="cpu")
            embedder._ensure_backend()
            mock_st.assert_called_once_with("fake/model", device="cpu")

    def test_query_format_uses_model_card_template(self) -> None:
        embedder = TextEmbedder(model_id="fake/model", device="cpu")
        formatted = embedder._format_query("धर्मो रक्षति")
        self.assertEqual(
            formatted,
            f"<instruct>{DEFAULT_QUERY_INSTRUCTION}\n<query>धर्मो रक्षति",
        )

    def test_query_instruction_references_sanskrit(self) -> None:
        self.assertIn("Sanskrit", DEFAULT_QUERY_INSTRUCTION)

    def test_default_model_receives_torch_dtype_and_device_map(self) -> None:
        with patch("sanskrit_pipeline.embeddings.AutoTokenizer") as mock_tokenizer_cls:
            with patch("sanskrit_pipeline.embeddings.AutoModelForCausalLM") as mock_model_cls:
                mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value
                mock_tokenizer.pad_token = "<pad>"
                mock_tokenizer.eos_token = "</s>"
                embedder = TextEmbedder(
                    model_id=DEFAULT_MODEL_ID,
                    device="cpu",
                    torch_dtype="float16",
                    device_map="auto",
                )
                embedder._ensure_backend()

        mock_model_cls.from_pretrained.assert_called_once_with(
            DEFAULT_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        mock_model_cls.from_pretrained.return_value.to.assert_not_called()

    def test_cuda_dtype_loads_directly_to_cuda_with_low_cpu_mem_usage(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            with patch("sanskrit_pipeline.embeddings.AutoTokenizer") as mock_tokenizer_cls:
                with patch("sanskrit_pipeline.embeddings.AutoModelForCausalLM") as mock_model_cls:
                    mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value
                    mock_tokenizer.pad_token = "<pad>"
                    mock_tokenizer.eos_token = "</s>"
                    embedder = TextEmbedder(
                        model_id=DEFAULT_MODEL_ID,
                        device="cuda",
                        torch_dtype="bfloat16",
                    )
                    embedder._ensure_backend()

        mock_model_cls.from_pretrained.assert_called_once_with(
            DEFAULT_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
            low_cpu_mem_usage=True,
        )
        mock_model_cls.from_pretrained.return_value.to.assert_not_called()

    def test_low_cpu_mem_usage_can_be_disabled_explicitly(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            embedder = TextEmbedder(
                model_id=DEFAULT_MODEL_ID,
                device="cuda",
                torch_dtype="bfloat16",
                low_cpu_mem_usage=False,
            )

        self.assertEqual(
            embedder._model_load_kwargs(trust_remote_code=True),
            {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "device_map": {"": "cuda"},
            },
        )

    def test_8bit_loading_uses_quantization_config_without_to_device(self) -> None:
        with patch("sanskrit_pipeline.embeddings.AutoTokenizer") as mock_tokenizer_cls:
            with patch("sanskrit_pipeline.embeddings.AutoModelForCausalLM") as mock_model_cls:
                with patch("sanskrit_pipeline.embeddings.BitsAndBytesConfig") as mock_bnb_cls:
                    mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value
                    mock_tokenizer.pad_token = "<pad>"
                    mock_tokenizer.eos_token = "</s>"
                    embedder = TextEmbedder(
                        model_id=DEFAULT_MODEL_ID,
                        device="cpu",
                        load_in_8bit=True,
                    )
                    embedder._ensure_backend()

        mock_bnb_cls.assert_called_once_with(load_in_8bit=True)
        mock_model_cls.from_pretrained.assert_called_once_with(
            DEFAULT_MODEL_ID,
            trust_remote_code=True,
            quantization_config=mock_bnb_cls.return_value,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        mock_model_cls.from_pretrained.return_value.to.assert_not_called()

    def test_base_model_last_hidden_state_skips_lm_head(self) -> None:
        sentinel = object()

        class FakeBase:
            def __init__(self) -> None:
                self.called_with = None

            def __call__(self, **kwargs):
                self.called_with = kwargs
                return type("Out", (), {"last_hidden_state": sentinel})()

        class FakeCausalLM:
            def __init__(self) -> None:
                self.model = FakeBase()

            def __call__(self, **kwargs):  # full LM path must NOT be used
                raise AssertionError("full causal LM forward should be skipped")

        embedder = TextEmbedder(model_id=DEFAULT_MODEL_ID, device="cpu")
        embedder._model = FakeCausalLM()
        encoded = {"input_ids": "x", "attention_mask": "y"}

        result = embedder._base_model_last_hidden_state(encoded)

        self.assertIs(result, sentinel)
        self.assertEqual(embedder._model.model.called_with, encoded)

    def test_base_model_last_hidden_state_falls_back_to_decoder(self) -> None:
        sentinel = object()

        class FakeDecoder:
            def __call__(self, **kwargs):
                return type("Out", (), {"last_hidden_state": sentinel})()

        class FakeCausalLM:
            model = None

            def get_decoder(self):
                return FakeDecoder()

        embedder = TextEmbedder(model_id=DEFAULT_MODEL_ID, device="cpu")
        embedder._model = FakeCausalLM()

        result = embedder._base_model_last_hidden_state({"input_ids": "x"})
        self.assertIs(result, sentinel)

    def test_input_device_uses_first_non_cpu_device_from_model_map(self) -> None:
        embedder = TextEmbedder(model_id=DEFAULT_MODEL_ID, device="cpu")
        model = type("FakeModel", (), {})()
        model.hf_device_map = {"layer0": "cpu", "layer1": "cuda:0"}
        embedder._model = model

        self.assertEqual(embedder._input_device(), "cuda:0")


if __name__ == "__main__":
    unittest.main()
