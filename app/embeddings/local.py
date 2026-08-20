"""
локальная модель через sentence-transformers + опционально onnx
если модель не скачана / нет зависимостей - тихо фолбэчимся на Dummy
"""
import asyncio
import logging
import numpy as np

from .base import EmbeddingProvider
from .dummy import DummyEmbedding

logger = logging.getLogger(__name__)


class LocalEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu", use_onnx: bool = False):
        self.model_name = model_name
        self.device = device
        self.use_onnx = use_onnx
        self.dim = 384  # для minilm
        self._model = None
        self._fallback = DummyEmbedding(dim=self.dim)
        self._load_attempted = False
        self._is_fallback = False

    def _load_model(self):
        if self._load_attempted:
            return
        self._load_attempted = True

        # пробуем onnx
        if self.use_onnx:
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer

                logger.info(f"loading ONNX model {self.model_name} ...")
                self._model = ORTModelForFeatureExtraction.from_pretrained(self.model_name)
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._is_onnx = True
                logger.info("ONNX model loaded")
                return
            except Exception as e:
                logger.warning(f"ONNX load failed ({e}), trying sentence-transformers")

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"loading sentence-transformers {self.model_name} on {self.device} ...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self._model.get_sentence_embedding_dimension()
            self._fallback.dim = self.dim
            self._is_onnx = False
            logger.info(f"model loaded, dim={self.dim}")
        except Exception as e:
            logger.warning(f"failed to load ST model: {e} -> using DummyEmbedding")
            self._model = None
            self._is_fallback = True

    async def embed(self, text: str) -> np.ndarray:
        # грузим лениво и в треде чтобы не блочить event loop
        if not self._load_attempted:
            await asyncio.to_thread(self._load_model)

        if self._model is None or self._is_fallback:
            return await self._fallback.embed(text)

        # инференс тоже в треде - модель синхронная
        def _encode():
            if getattr(self, "_is_onnx", False):
                import torch
                import numpy as np

                # onnx путь - mean pooling
                encoded = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                outputs = self._model(**encoded)
                # last_hidden_state -> mean pooling
                attention_mask = encoded["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_emb = (token_embeddings * input_mask_expanded).sum(dim=1)
                sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
                emb = (sum_emb / sum_mask).detach().numpy()[0]
                # normalize
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                return emb.astype(np.float32)
            else:
                vec = self._model.encode(text, normalize_embeddings=True)
                return vec.astype(np.float32)

        try:
            return await asyncio.to_thread(_encode)
        except Exception as e:
            logger.warning(f"encode failed {e}, fallback to dummy")
            return await self._fallback.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        # пока просто по одному, можно оптимизировать батчем
        results = []
        for t in texts:
            results.append(await self.embed(t))
        return results


# синглтон провайдер
_provider: LocalEmbedding | None = None


def get_embedding_provider(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    use_onnx: bool = False,
    force_dummy: bool = False,
) -> EmbeddingProvider:
    global _provider
    if force_dummy:
        return DummyEmbedding()
    if _provider is None:
        _provider = LocalEmbedding(model_name, device, use_onnx)
    return _provider
