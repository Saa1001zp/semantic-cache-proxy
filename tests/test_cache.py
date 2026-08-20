import pytest
import numpy as np
from app.cache.memory import MemoryStore
from app.cache.base import CacheEntry
from app.utils.hashing import deterministic_embedding
from app.utils.cosine import cosine_similarity


@pytest.mark.asyncio
async def test_memory_store_hit():
    store = MemoryStore(ttl_seconds=3600, max_size=100)
    text = "привет как дела"
    emb = deterministic_embedding(text)
    entry = CacheEntry(query_text=text, embedding=emb, response={"answer": "норм"})
    await store.add(entry)

    # точно такой же запрос - должен быть хит
    q_emb = deterministic_embedding("привет как дела")
    res = await store.search(q_emb, threshold=0.93)
    assert res is not None
    assert res.similarity > 0.99
    assert res.entry.query_text == text


@pytest.mark.asyncio
async def test_memory_store_miss_different_text():
    store = MemoryStore(ttl_seconds=3600, max_size=100)
    emb = deterministic_embedding("привет как дела")
    await store.add(CacheEntry(query_text="привет как дела", embedding=emb, response={"ok": 1}))

    # совсем другой текст - мисс
    q_emb = deterministic_embedding("рецепт борща со свеклой")
    res = await store.search(q_emb, threshold=0.93)
    assert res is None


@pytest.mark.asyncio
async def test_ttl_expired():
    store = MemoryStore(ttl_seconds=0, max_size=100)  # сразу протухает
    emb = deterministic_embedding("hello")
    await store.add(CacheEntry(query_text="hello", embedding=emb, response={"x": 1}))
    # подождать чуть
    import asyncio

    await asyncio.sleep(0.05)
    res = await store.search(emb, threshold=0.5)
    assert res is None  # протух


@pytest.mark.asyncio
async def test_eviction():
    store = MemoryStore(ttl_seconds=3600, max_size=5)
    for i in range(10):
        emb = deterministic_embedding(f"text {i}")
        await store.add(CacheEntry(query_text=f"text {i}", embedding=emb, response={"i": i}))
    size = await store.size()
    assert size <= 5


def test_cosine_basic():
    a = np.array([1, 0, 0], dtype=float)
    b = np.array([1, 0, 0], dtype=float)
    assert cosine_similarity(a, b) == pytest.approx(1.0)

    c = np.array([0, 1, 0], dtype=float)
    assert cosine_similarity(a, c) == pytest.approx(0.0, abs=1e-6)

    d = np.array([-1, 0, 0], dtype=float)
    assert cosine_similarity(a, d) == pytest.approx(-1.0)
