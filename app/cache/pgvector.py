"""
pgvector реализация - честный векторный поиск через postgres

схема:
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE TABLE cache_entries (
    id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    embedding vector(384),
    response JSONB NOT NULL,
    model TEXT,
    created_at DOUBLE PRECISION,
    hits INT DEFAULT 0,
    input_tokens INT DEFAULT 0,
    output_tokens INT DEFAULT 0
  );
  CREATE INDEX ON cache_entries USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

поиск:
  SELECT ..., 1 - (embedding <=> $1) AS similarity
  FROM cache_entries
  WHERE 1 - (embedding <=> $1) > $2
    AND created_at > $3
  ORDER BY embedding <=> $1
  LIMIT 1;

если postgres недоступен - тихо фолбэчимся на MemoryStore (чтобы тесты и локалка без докера не падали)
"""
import json
import time
import logging
import numpy as np

from .base import CacheStore, CacheEntry, SearchResult
from .memory import MemoryStore

logger = logging.getLogger(__name__)


def _to_vector_literal(vec: np.ndarray) -> str:
    # pgvector ожидает формат '[0.1,0.2,0.3]'
    # округляем до 6 знаков чтобы не раздувать запрос
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


class PgVectorStore(CacheStore):
    def __init__(self, dsn: str, ttl_seconds: int = 604800, max_size: int = 10000):
        # dsn приходит как postgresql+asyncpg://... - asyncpg хочет без +asyncpg
        self.raw_dsn = dsn
        self.dsn = dsn.replace("postgresql+asyncpg://", "postgresql://").replace("postgresql+psycopg://", "postgresql://")
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

        self._fallback = MemoryStore(ttl_seconds=ttl_seconds, max_size=max_size)
        self._pool = None
        self._init_failed = False
        self._table_ready = False
        # lazy import чтобы не падать если asyncpg не установлен
        self._asyncpg = None

    async def _ensure_pool(self):
        if self._pool is not None or self._init_failed:
            return self._pool is not None

        try:
            import asyncpg  # type: ignore

            self._asyncpg = asyncpg
            # asyncpg не любит +asyncpg в dsn, уже почистили
            self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5, timeout=5)
            logger.info(f"pgvector pool created: {self.dsn.split('@')[-1]}")
            return True
        except Exception as e:
            logger.warning(f"pgvector pool failed ({e}) - fallback to MemoryStore (set VECTOR_BACKEND=memory to silence)")
            self._init_failed = True
            return False

    async def _ensure_table(self):
        if self._table_ready or self._init_failed:
            return self._table_ready
        if not await self._ensure_pool():
            return False

        try:
            async with self._pool.acquire() as conn:
                # расширение
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                # таблица - dim берем из конфига, но пока 384 (minilm), pgvector требует точный размер
                # делаем 384, если модель другая - миграция руками
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        id TEXT PRIMARY KEY,
                        query_text TEXT NOT NULL,
                        embedding vector(384),
                        response JSONB NOT NULL,
                        model TEXT,
                        created_at DOUBLE PRECISION,
                        hits INT DEFAULT 0,
                        input_tokens INT DEFAULT 0,
                        output_tokens INT DEFAULT 0
                    );
                """)
                # индекс - создаем если нет (ivfflat требует хотя бы немного данных, поэтому IF NOT EXISTS + проверка)
                # для пустой таблицы индекс может не создаться - ок, создастся позже
                try:
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_cache_embedding
                        ON cache_entries USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                    """)
                except Exception as e:
                    # ivfflat не любит пустые таблицы - не критично, будет seq scan пока мало данных
                    logger.debug(f"ivfflat index not created yet (ok for empty table): {e}")

                self._table_ready = True
                logger.info("pgvector table ready")
                return True
        except Exception as e:
            logger.warning(f"pgvector table init failed ({e}) - fallback to MemoryStore")
            self._init_failed = True
            return False

    async def search(self, query_embedding: np.ndarray, threshold: float = 0.93) -> SearchResult | None:
        # если не смогли подключиться - фолбэк
        if not await self._ensure_table():
            return await self._fallback.search(query_embedding, threshold)

        try:
            vec_literal = _to_vector_literal(query_embedding)
            cutoff = time.time() - self.ttl_seconds

            async with self._pool.acquire() as conn:
                # чистим протухшие лениво (раз в N запросов можно, но пока каждый раз - ок для джун-проекта)
                await conn.execute("DELETE FROM cache_entries WHERE created_at < $1", cutoff)

                row = await conn.fetchrow(
                    """
                    SELECT id, query_text, embedding, response, model, created_at, hits,
                           input_tokens, output_tokens,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM cache_entries
                    WHERE 1 - (embedding <=> $1::vector) > $2
                      AND created_at > $3
                    ORDER BY embedding <=> $1::vector
                    LIMIT 1;
                    """,
                    vec_literal,
                    threshold,
                    cutoff,
                )

                if row is None:
                    logger.debug(f"pgvector MISS (threshold={threshold})")
                    return None

                sim = float(row["similarity"])
                # инкрементим hits
                await conn.execute("UPDATE cache_entries SET hits = hits + 1 WHERE id = $1", row["id"])

                # собираем CacheEntry - embedding из строки не нужен для выдачи, берем из запроса
                entry = CacheEntry(
                    id=row["id"],
                    query_text=row["query_text"],
                    embedding=query_embedding,  # не тянем вектор обратно ради экономии
                    response=json.loads(row["response"]) if isinstance(row["response"], str) else row["response"],
                    model=row["model"] or "",
                    created_at=float(row["created_at"]),
                    hits=int(row["hits"]) + 1,
                    input_tokens=int(row["input_tokens"] or 0),
                    output_tokens=int(row["output_tokens"] or 0),
                )
                logger.debug(f"pgvector HIT sim={sim:.4f} id={entry.id}")
                return SearchResult(entry=entry, similarity=sim)

        except Exception as e:
            logger.warning(f"pgvector search failed ({e}) - fallback to memory for this request")
            # пробуем найти в фолбэке (там может быть что-то)
            fallback_res = await self._fallback.search(query_embedding, threshold)
            if fallback_res:
                return fallback_res
            # если и там пусто - считаем MISS, но не падаем
            return None

    async def add(self, entry: CacheEntry) -> None:
        # всегда пишем и в fallback (чтобы при падении pg не терять)
        await self._fallback.add(entry)

        if not await self._ensure_table():
            return

        try:
            # eviction - если переполнено, удаляем самые старые 10%
            async with self._pool.acquire() as conn:
                cnt = await conn.fetchval("SELECT COUNT(*) FROM cache_entries;")
                if cnt is not None and cnt >= self.max_size:
                    drop = max(1, self.max_size // 10)
                    await conn.execute(
                        """
                        DELETE FROM cache_entries
                        WHERE id IN (
                            SELECT id FROM cache_entries ORDER BY created_at ASC LIMIT $1
                        );
                        """,
                        drop,
                    )
                    logger.info(f"pgvector eviction: dropped {drop} oldest")

                vec_literal = _to_vector_literal(entry.embedding) if entry.embedding is not None else None
                if vec_literal is None:
                    logger.warning("pgvector add: entry without embedding, skip")
                    return

                # response как jsonb
                resp_json = json.dumps(entry.response, ensure_ascii=False)

                await conn.execute(
                    """
                    INSERT INTO cache_entries (id, query_text, embedding, response, model, created_at, hits, input_tokens, output_tokens)
                    VALUES ($1, $2, $3::vector, $4::jsonb, $5, $6, $7, $8, $9)
                    ON CONFLICT (id) DO UPDATE SET
                        hits = EXCLUDED.hits,
                        response = EXCLUDED.response;
                    """,
                    entry.id,
                    entry.query_text,
                    vec_literal,
                    resp_json,
                    entry.model,
                    float(entry.created_at),
                    int(entry.hits),
                    int(entry.input_tokens),
                    int(entry.output_tokens),
                )
                logger.debug(f"pgvector ADD id={entry.id}")

        except Exception as e:
            logger.warning(f"pgvector add failed ({e}) - kept in memory fallback")

    async def clear(self) -> None:
        await self._fallback.clear()
        if not await self._ensure_pool():
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("DELETE FROM cache_entries;")
                logger.info("pgvector cleared")
        except Exception as e:
            logger.warning(f"pgvector clear failed: {e}")

    async def size(self) -> int:
        if not await self._ensure_table():
            return await self._fallback.size()
        try:
            async with self._pool.acquire() as conn:
                cnt = await conn.fetchval("SELECT COUNT(*) FROM cache_entries;")
                return int(cnt or 0)
        except Exception:
            return await self._fallback.size()

    async def all_entries(self) -> list[CacheEntry]:
        # для дашборда - тянем из pg, но лимитируем
        if not await self._ensure_table():
            return await self._fallback.all_entries()
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id, query_text, response, model, created_at, hits, input_tokens, output_tokens FROM cache_entries ORDER BY created_at DESC LIMIT 100;"
                )
                out = []
                for r in rows:
                    out.append(
                        CacheEntry(
                            id=r["id"],
                            query_text=r["query_text"],
                            embedding=None,
                            response=json.loads(r["response"]) if isinstance(r["response"], str) else r["response"],
                            model=r["model"] or "",
                            created_at=float(r["created_at"]),
                            hits=int(r["hits"]),
                            input_tokens=int(r["input_tokens"] or 0),
                            output_tokens=int(r["output_tokens"] or 0),
                        )
                    )
                return out
        except Exception:
            return await self._fallback.all_entries()

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None
