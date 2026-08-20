import numpy as np
from .base import EmbeddingProvider
from app.utils.hashing import deterministic_embedding


class DummyEmbedding(EmbeddingProvider):
    """детерминированный хеш-эмбеддинг, без зависимостей"""

    def __init__(self, dim: int = 384):
        self.dim = dim

    async def embed(self, text: str) -> np.ndarray:
        return deterministic_embedding(text, self.dim)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [deterministic_embedding(t, self.dim) for t in texts]
