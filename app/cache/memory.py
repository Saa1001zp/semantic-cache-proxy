"""
in-memory векторный кэш с косинусным поиском через numpy
для продакшена можно заменить на pgvector/redis без смены интерфейса
"""
import asyncio
import time
import logging
import numpy as np
from .base import CacheStore, CacheEntry, SearchResult
from app.utils.cosine import cosine_similarity_batch

logger = logging.getLogger(__name__)


class MemoryStore(CacheStore):
    def __init__(self, ttl_seconds: int = 604800, max_size: int = 10000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._entries: list[CacheEntry] = []
        self._lock = asyncio.Lock()

    async def search(self, query_embedding: np.ndarray, threshold: float = 0.93) -> SearchResult | None:
        async with self._lock:
            # чистим протухшие на лету (лениво)
            now = time.time()
            alive = [e for e in self._entries if (now - e.created_at) <= self.ttl_seconds]
            if len(alive) != len(self._entries):
                self._entries = alive

            if not self._entries:
                return None

            # собираем матрицу эмбеддингов
            # фильтруем без эмбеддинга (не должно быть, но на всякий)
            valid = [e for e in self._entries if e.embedding is not None]
            if not valid:
                return None

            matrix = np.stack([e.embedding for e in valid])
            scores = cosine_similarity_batch(query_embedding, matrix)

            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score >= threshold:
                entry = valid[best_idx]
                # хит - инкрементим счетчик
                entry.hits += 1
                logger.debug(f"cache HIT sim={best_score:.4f} q='{entry.query_text[:60]}'")
                return SearchResult(entry=entry, similarity=best_score)

            logger.debug(f"cache MISS best={best_score:.4f} threshold={threshold}")
            return None

    async def add(self, entry: CacheEntry) -> None:
        async with self._lock:
            # eviction если переполнен - выкидываем самый старый (FIFO + LRU гибрид)
            # сортируем по created_at, удаляем 10% самых старых
            if len(self._entries) >= self.max_size:
                self._entries.sort(key=lambda e: e.created_at)
                drop = max(1, self.max_size // 10)
                self._entries = self._entries[drop:]
                logger.info(f"cache eviction: dropped {drop} oldest entries")

            self._entries.append(entry)
            logger.debug(f"cache ADD id={entry.id} size={len(self._entries)}")

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    async def size(self) -> int:
        async with self._lock:
            return len(self._entries)

    async def all_entries(self) -> list[CacheEntry]:
        async with self._lock:
            return list(self._entries)

    # для тестов - синхронный хелпер
    def sync_size(self) -> int:
        return len(self._entries)
