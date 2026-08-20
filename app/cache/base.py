from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import time
import uuid


@dataclass
class CacheEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = ""
    # embedding храним как numpy, сериализуем при надобности
    embedding: np.ndarray | None = None
    response: dict = field(default_factory=dict)
    model: str = ""
    created_at: float = field(default_factory=lambda: time.time())
    hits: int = 0
    # для дашборда - сколько токенов сэкономили
    input_tokens: int = 0
    output_tokens: int = 0

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.created_at) > ttl_seconds

    def to_dict(self, include_embedding: bool = False) -> dict:
        d = {
            "id": self.id,
            "query_text": self.query_text,
            "response": self.response,
            "model": self.model,
            "created_at": self.created_at,
            "hits": self.hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
        if include_embedding and self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d


@dataclass
class SearchResult:
    entry: CacheEntry
    similarity: float


class CacheStore(ABC):
    @abstractmethod
    async def search(self, query_embedding: np.ndarray, threshold: float = 0.93) -> SearchResult | None:
        ...

    @abstractmethod
    async def add(self, entry: CacheEntry) -> None:
        ...

    @abstractmethod
    async def clear(self) -> None:
        ...

    @abstractmethod
    async def size(self) -> int:
        ...

    @abstractmethod
    async def all_entries(self) -> list[CacheEntry]:
        ...
