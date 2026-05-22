"""Embedding backends for Sanskrit sentence lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_ID = "buddhist-nlp/gemma-2-mitra-e"
DEFAULT_QUERY_INSTRUCTION = "Please find the semantically most similar text in Sanskrit."
TorchDTypeName = Literal["auto", "float16", "bfloat16", "float32"]


@dataclass(slots=True)
class EmbeddingResult:
    """Embedding output with model metadata."""

    model_id: str
    embeddings: np.ndarray


class TextEmbedder:
    """A flexible text embedder supporting sentence-transformers and fallback pooling."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        normalize_embeddings: bool = True,
        batch_size: int = 8,
        device: Literal["auto", "cpu", "mps", "cuda"] = "auto",
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        max_length: int = 512,
        embedding_progress: Literal["off", "batch", "sentence"] = "off",
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool = False,
        low_cpu_mem_usage: bool | None = None,
    ) -> None:
        self.model_id = model_id
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.device = device
        self.query_instruction = query_instruction
        self.max_length = max_length
        self.embedding_progress = embedding_progress
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self._backend = None
        self._tokenizer = None
        self._model = None
        self._device = _resolve_torch_device(device)

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """Encode a list of texts into dense vectors."""
        if not texts:
            return EmbeddingResult(self.model_id, np.empty((0, 0), dtype=np.float32))

        self._ensure_backend()
        if self._backend == "sentence-transformers":
            embeddings = self._sentence_transformer.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )
            return EmbeddingResult(self.model_id, embeddings)

        embeddings = self._transformers_encode(texts)
        return EmbeddingResult(self.model_id, embeddings)

    def encode_queries(self, texts: list[str]) -> EmbeddingResult:
        """Encode retrieval queries (asymmetric when supported by the model)."""
        if not texts:
            return EmbeddingResult(self.model_id, np.empty((0, 0), dtype=np.float32))
        self._ensure_backend()
        if self._backend == "gemma-last-token":
            processed = [self._format_query(text) for text in texts]
            return EmbeddingResult(self.model_id, self._transformers_encode(processed))
        return self.encode(texts)

    def encode_corpus(self, texts: list[str]) -> EmbeddingResult:
        """Encode corpus passages (raw text for asymmetric retrieval models)."""
        return self.encode(texts)

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return

        if self.model_id == DEFAULT_MODEL_ID:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    **self._model_load_kwargs(trust_remote_code=True),
                )
                self._move_model_to_device()
                self._model.eval()
                if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                self._backend = "gemma-last-token"
                return
            except Exception as exc:
                if _is_mps_oom(exc):
                    raise RuntimeError(
                        "MPS out of memory while loading embedding model. "
                        "Rerun with device='cpu' (CLI: --device cpu)."
                    ) from exc
                raise RuntimeError(
                    f"Failed to load {self.model_id} with transformers backend required by the model card."
                ) from exc

        try:
            self._sentence_transformer = SentenceTransformer(self.model_id, device=self._device)
            self._backend = "sentence-transformers"
        except Exception as exc:
            if _is_mps_oom(exc):
                raise RuntimeError(
                    "MPS out of memory while loading embedding model. "
                    "Rerun with device='cpu' (CLI: --device cpu)."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id, **self._model_load_kwargs())
            self._move_model_to_device()
            self._model.eval()
            self._backend = "transformers"

    def _transformers_encode(self, texts: list[str]) -> np.ndarray:
        output_batches: list[np.ndarray] = []
        total = len(texts)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        self._log(
            "batch",
            f"[embed] backend={self._backend} device={self._device} total_sentences={total} "
            f"batch_size={self.batch_size} batches={num_batches}",
        )

        for batch_idx, batch_start in enumerate(range(0, total, self.batch_size), start=1):
            batch = texts[batch_start : batch_start + self.batch_size]
            batch_end = min(batch_start + len(batch), total)
            self._log(
                "batch",
                f"[embed] batch {batch_idx}/{num_batches} sentences {batch_start + 1}-{batch_end}/{total}",
            )
            if self.embedding_progress == "sentence":
                for sentence_idx in range(batch_start, batch_end):
                    self._log("sentence", f"[embed] sentence {sentence_idx + 1}/{total}")
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_device = self._input_device()
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            with torch.no_grad():
                if self._backend == "gemma-last-token":
                    outputs = self._model(**encoded, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    last_token_idx = encoded["attention_mask"].sum(dim=1) - 1
                    pooled = hidden[
                        torch.arange(hidden.size(0), device=hidden.device),
                        last_token_idx,
                    ]
                else:
                    outputs = self._model(**encoded)
                    hidden = outputs.last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            if self.normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            output_batches.append(pooled.float().cpu().numpy().astype(np.float32))

        return np.vstack(output_batches)

    def _format_query(self, text: str) -> str:
        return f"<instruct>{self.query_instruction}\n<query>{text}"

    def _model_load_kwargs(self, *, trust_remote_code: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = _resolve_torch_dtype(self.torch_dtype)
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        elif self._should_direct_load_on_cuda():
            kwargs["device_map"] = {"": self._device}
        if self.load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs.setdefault("device_map", "auto")
        if self._should_use_low_cpu_mem_usage(kwargs):
            kwargs["low_cpu_mem_usage"] = True
        return kwargs

    def _move_model_to_device(self) -> None:
        if self._uses_loader_device_placement():
            return
        self._model.to(self._device)

    def _uses_loader_device_placement(self) -> bool:
        return self.device_map is not None or self.load_in_8bit or self._should_direct_load_on_cuda()

    def _should_direct_load_on_cuda(self) -> bool:
        return self._device == "cuda" and self.torch_dtype is not None and not self.load_in_8bit

    def _should_use_low_cpu_mem_usage(self, load_kwargs: dict[str, Any]) -> bool:
        if self.low_cpu_mem_usage is not None:
            return self.low_cpu_mem_usage
        return "device_map" in load_kwargs or self.load_in_8bit

    def _input_device(self) -> str:
        if self._model is None:
            return self._device
        device_map = getattr(self._model, "hf_device_map", None)
        if not device_map:
            return self._device
        for device in device_map.values():
            if device not in {"cpu", "disk"}:
                return str(device)
        return self._device

    def _log(self, level: Literal["batch", "sentence"], message: str) -> None:
        if self.embedding_progress == "off":
            return
        if self.embedding_progress == "batch" and level == "sentence":
            return
        print(message, flush=True)


def _resolve_torch_device(preferred: Literal["auto", "cpu", "mps", "cuda"] = "auto") -> str:
    """Pick the best available PyTorch device for inference."""
    if preferred not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported device: {preferred}")

    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    if preferred == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return "mps"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_torch_dtype(dtype: TorchDTypeName) -> str | torch.dtype:
    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {dtype}")


def _is_mps_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "mps backend out of memory" in message or ("mps" in message and "out of memory" in message)
