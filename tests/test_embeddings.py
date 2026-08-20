import pytest
import numpy as np
from app.embeddings.dummy import DummyEmbedding


@pytest.mark.asyncio
async def test_dummy_deterministic():
    emb = DummyEmbedding(dim=64)
    v1 = await emb.embed("hello world")
    v2 = await emb.embed("hello world")
    assert np.allclose(v1, v2), "одинаковый текст должен давать одинаковый вектор"


@pytest.mark.asyncio
async def test_dummy_normalized():
    emb = DummyEmbedding(dim=64)
    v = await emb.embed("test")
    norm = np.linalg.norm(v)
    assert norm == pytest.approx(1.0, abs=1e-5)


@pytest.mark.asyncio
async def test_dummy_similar_texts_closer():
    emb = DummyEmbedding(dim=128)
    a = await emb.embed("привет как дела")
    b = await emb.embed("привет как дела у тебя")
    c = await emb.embed("рецепт борща")

    from app.utils.cosine import cosine_similarity

    sim_ab = cosine_similarity(a, b)
    sim_ac = cosine_similarity(a, c)
    # похожие тексты должны быть ближе чем рандомные
    assert sim_ab > sim_ac


@pytest.mark.asyncio
async def test_dummy_batch():
    emb = DummyEmbedding()
    vecs = await emb.embed_batch(["hello", "world", "test"])
    assert len(vecs) == 3
    assert all(v.shape[0] == 384 for v in vecs)
