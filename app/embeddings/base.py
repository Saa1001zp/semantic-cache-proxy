from abc import ABC, abstractmethod
import numpy as np


class EmbeddingProvider(ABC):
    dim: int = 384

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """возвращает нормированный вектор"""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        ...

    def get_dim(self) -> int:
        return self.dim
