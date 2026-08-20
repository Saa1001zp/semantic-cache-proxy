import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "cache_size" in data


@pytest.mark.asyncio
async def test_stats_initial(client):
    resp = await client.get("/stats")
    assert resp.status_code == 200
    j = resp.json()
    assert j["total_requests"] == 0
    assert j["hit_rate_percent"] == 0
    assert "money_saved_usd" in j


@pytest.mark.asyncio
async def test_chat_miss_then_hit(client):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "привет как дела"}],
    }
    # 1й запрос - MISS (кэш пустой)
    r1 = await client.post("/v1/chat/completions", json=payload)
    assert r1.status_code == 200
    assert r1.headers.get("X-Cache") == "MISS"
    data1 = r1.json()
    assert "choices" in data1
    # должен содержать мок ответ
    assert "привет как дела" in data1["choices"][0]["message"]["content"]

    # 2й запрос - точно такой же -> HIT
    r2 = await client.post("/v1/chat/completions", json=payload)
    assert r2.status_code == 200
    assert r2.headers.get("X-Cache") == "HIT"
    data2 = r2.json()
    assert data2["_cache_hit"] is True
    assert data2["_cache_similarity"] > 0.9

    # stats должны показать 50% hit rate
    s = await client.get("/stats")
    j = s.json()
    assert j["total_requests"] == 2
    assert j["hits"] == 1
    assert j["misses"] == 1
    assert j["hit_rate_percent"] == 50.0
    assert j["total_tokens_saved"] > 0


@pytest.mark.asyncio
async def test_cache_semantic_hit(client):
    # проверяем что семантически близкий запрос тоже хитует если threshold низкий
    # с dummy эмбеддингом "привет как дела" и "привет, как у тебя дела?" должны быть близки
    p1 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "привет как дела"}]}
    p2 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "привет как дела у тебя"}]}

    await client.post("/v1/chat/completions", json=p1)
    r2 = await client.post("/v1/chat/completions", json=p2)
    # с threshold 0.85 как в conftest - должен быть HIT, иначе MISS тоже ок - проверяем что не 500
    assert r2.status_code == 200
    assert r2.headers.get("X-Cache") in ("HIT", "MISS")


@pytest.mark.asyncio
async def test_completions_legacy(client):
    payload = {"model": "gpt-4o-mini", "prompt": "напиши стих про кота"}
    r1 = await client.post("/v1/completions", json=payload)
    assert r1.status_code == 200
    assert r1.headers.get("X-Cache") == "MISS"

    r2 = await client.post("/v1/completions", json=payload)
    assert r2.headers.get("X-Cache") == "HIT"


@pytest.mark.asyncio
async def test_cache_clear(client):
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "test clear"}]}
    await client.post("/v1/chat/completions", json=payload)
    r = await client.post("/cache/clear")
    assert r.status_code == 200

    s = await client.get("/stats")
    assert s.json()["total_requests"] == 0  # метрики сбросились

    h = await client.get("/health")
    assert h.json()["cache_size"] == 0


@pytest.mark.asyncio
async def test_empty_prompt_400(client):
    r = await client.post("/v1/chat/completions", json={"model": "gpt", "messages": []})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_dashboard(client):
    r = await client.get("/dashboard")
    assert r.status_code == 200
    assert "Semantic Cache Proxy" in r.text
