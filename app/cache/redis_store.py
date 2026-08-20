"""
заглушка для redis + redisvl
в проде: redisvl + RediSearch с VECTOR полем и FT.SEARCH
"""
import logging
import numpy as np
from .base import CacheStore, CacheEntry, SearchResult
from .memory import MemoryStore

logger = logging.getLogger(__name__)


class RedisStore(CacheStore):
    """
    TODO:
      from redisvl.index import SearchIndex
      schema = {
        "fields": [
          {"name": "query_text", "type": "text"},
          {"name": "embedding", "type": "vector", "attrs": {"dims": 384, "distance_metric": "cosine", "algorithm": "hnsw"}},
          {"name": "response", "type": "tag"}
        ]
      }
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 604800, max_size: int = 10000):
        logger.warning("RedisStore not fully implemented, falling back to MemoryStore")
        self._fallback = MemoryStore(ttl_seconds=ttl_seconds, max_size=max_size)
        self.redis_url = redis_url

    async def search(self, query_embedding: np.ndarray, threshold: float = 0.93) -> SearchResult | None:
        return await self._fallback.search(query_embedding, threshold)

    async def add(self, entry: CacheEntry) -> None:
        await self._fallback.add(entry)

    async def clear(self) -> None:
        await self._fallback.clear()

    async def size(self) -> int:
        return await self._fallback.size()

    async def all_entries(self) -> list[CacheEntry]:
        return await self._fallback.all_entries()
