"""
semantic cache proxy - главный файл
легкий асинхронный прокси между клиентом и любой llm
"""
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.metrics import metrics
from app.cache.memory import MemoryStore
from app.cache.pgvector import PgVectorStore
from app.cache.redis_store import RedisStore
from app.cache.base import CacheEntry
from app.proxy.upstream import UpstreamClient
from app.embeddings.local import get_embedding_provider
from app.embeddings.dummy import DummyEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic-cache")

# глобальные синглтоны - инициализируются в lifespan
cache_store = None
upstream = None
embedder = None


def _make_cache_store():
    backend = settings.vector_backend.lower()
    if backend == "pgvector":
        return PgVectorStore(
            dsn=settings.database_url,
            ttl_seconds=settings.cache_ttl_seconds,
            max_size=settings.max_cache_size,
        )
    elif backend == "redis":
        return RedisStore(
            redis_url=settings.redis_url,
            ttl_seconds=settings.cache_ttl_seconds,
            max_size=settings.max_cache_size,
        )
    else:
        return MemoryStore(ttl_seconds=settings.cache_ttl_seconds, max_size=settings.max_cache_size)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache_store, upstream, embedder
    logger.info("🚀 starting semantic cache proxy ...")
    logger.info(f"  threshold={settings.similarity_threshold} ttl={settings.cache_ttl_seconds}s backend={settings.vector_backend}")
    logger.info(f"  embedding_model={settings.embedding_model} onnx={settings.use_onnx}")

    cache_store = _make_cache_store()
    # для pgvector - прогреваем таблицу заранее, чтобы первый запрос не ждал DDL
    if settings.vector_backend.lower() == "pgvector":
        try:
            await cache_store._ensure_table()  # type: ignore
        except Exception as e:
            logger.warning(f"pgvector warmup failed: {e}")

    upstream = UpstreamClient(
        api_url=settings.upstream_api_url,
        api_key=settings.upstream_api_key,
        model=settings.upstream_model,
    )

    # embedder - если нет sentence-transformers или модель не качается, уйдем в dummy
    # для тестов можно форсить dummy через env FORCE_DUMMY_EMBEDDING=1
    import os

    force_dummy = os.getenv("FORCE_DUMMY_EMBEDDING") == "1"
    if force_dummy:
        logger.info("using DummyEmbedding (FORCE_DUMMY_EMBEDDING=1)")
        embedder = DummyEmbedding()
    else:
        embedder = get_embedding_provider(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            use_onnx=settings.use_onnx,
        )

    logger.info("✅ ready on port 8000")
    yield

    logger.info("shutting down ...")
    if upstream:
        await upstream.close()
    # pgvector pool тоже закрыть если есть
    if cache_store and hasattr(cache_store, "close"):
        try:
            await cache_store.close()  # type: ignore
        except Exception:
            pass


app = FastAPI(
    title="Semantic Cache Proxy",
    description="Семантический кэширующий прокси для LLM API - экономит токены и режет latency с 1-2с до ~15мс",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- helpers ----------


def _extract_query_text(body: dict) -> str:
    """
    вытаскиваем текст запроса для эмбеддинга
    для chat: берем последнее user сообщение (самое важное)
    для completions: поле prompt
    """
    # chat completions
    messages = body.get("messages")
    if messages:
        # ищем последний user
        for m in reversed(messages):
            if m.get("role") == "user" and m.get("content"):
                return m["content"].strip()
        # fallback: склеить все
        return " ".join([m.get("content", "") for m in messages]).strip()

    # legacy completions
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        return " ".join(prompt).strip()
    return str(prompt).strip()


def _estimate_tokens(text: str) -> int:
    # грубо: 1 токен ~ 4 символа или 0.75 слова
    return max(1, int(len(text.split()) * 1.3))


# ---------- routes ----------


@app.get("/health")
async def health():
    size = await cache_store.size() if cache_store else 0
    return {
        "status": "ok",
        "version": "0.1.0",
        "cache_backend": settings.vector_backend,
        "cache_size": size,
        "threshold": settings.similarity_threshold,
        "model": settings.embedding_model,
    }


@app.get("/stats")
async def stats():
    snap = metrics.snapshot()
    size = await cache_store.size() if cache_store else 0
    snap["cache_size"] = size
    snap["threshold"] = settings.similarity_threshold
    snap["ttl_seconds"] = settings.cache_ttl_seconds
    return snap


@app.post("/cache/clear")
async def clear_cache():
    await cache_store.clear()
    metrics.reset()
    return {"status": "cleared"}


@app.get("/cache/entries")
async def list_entries(limit: int = 20):
    entries = await cache_store.all_entries()
    # последние
    entries = sorted(entries, key=lambda e: e.created_at, reverse=True)[:limit]
    return {"count": len(entries), "entries": [e.to_dict() for e in entries]}


# главный прокси эндпоинт - openai совместимый
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, response: Response):
    start = time.time()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")

    query_text = _extract_query_text(body)
    if not query_text:
        raise HTTPException(status_code=400, detail="empty prompt / messages")

    # streaming пока без кэша - просто прокидываем
    if body.get("stream") is True:
        logger.info(f"stream request bypass cache: '{query_text[:60]}'")
        result = await upstream.chat_completion(body)
        # не кэшируем стримы (сложно), но считаем как miss
        metrics.record_miss(latency_ms=(time.time() - start) * 1000)
        return JSONResponse(content=result, headers={"X-Cache": "BYPASS", "X-Cache-Reason": "stream"})

    # 1. эмбеддинг запроса
    try:
        query_emb = await embedder.embed(query_text)
    except Exception as e:
        logger.error(f"embedding failed: {e}")
        # если эмбеддинг упал - идем мимо кэша
        result = await upstream.chat_completion(body)
        metrics.record_miss(latency_ms=(time.time() - start) * 1000)
        return JSONResponse(content=result, headers={"X-Cache": "BYPASS", "X-Cache-Reason": "embedding_error"})

    # 2. поиск в кэше
    search_result = await cache_store.search(query_emb, threshold=settings.similarity_threshold)
    latency_ms = (time.time() - start) * 1000

    if search_result is not None:
        # HIT - отдаем закэшированный ответ за ~10-20мс
        cached = search_result.entry.response
        # обновляем id/created чтобы выглядело как свежий ответ, но помечаем что из кэша
        # копируем чтобы не мутировать оригинал
        import copy

        out = copy.deepcopy(cached)
        # добавляем служебные поля (клиенту удобно дебажить)
        out["_cache_hit"] = True
        out["_cache_similarity"] = round(search_result.similarity, 4)
        out["_cache_query"] = search_result.entry.query_text[:120]

        # usage считаем для дашборда
        usage = cached.get("usage", {})
        inp = usage.get("prompt_tokens", _estimate_tokens(query_text))
        outp = usage.get("completion_tokens", _estimate_tokens(str(cached.get("choices", [{}])[0].get("message", {}).get("content", ""))))

        # пишем метрики - hit это ~15ms вместо 1200ms
        metrics.record_hit(latency_ms=latency_ms, input_tokens=inp, output_tokens=outp)

        logger.info(f"✅ HIT sim={search_result.similarity:.3f} latency={latency_ms:.1f}ms q='{query_text[:50]}'")
        return JSONResponse(
            content=out,
            headers={
                "X-Cache": "HIT",
                "X-Cache-Similarity": f"{search_result.similarity:.4f}",
                "X-Cache-Latency-Ms": f"{latency_ms:.2f}",
            },
        )

    # 3. MISS - идем в upstream
    upstream_start = time.time()
    try:
        result = await upstream.chat_completion(body)
    except Exception as e:
        logger.error(f"upstream error: {e}")
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")

    total_latency = (time.time() - start) * 1000
    upstream_latency = (time.time() - upstream_start) * 1000

    # 4. сохраняем в кэш для будущих хитов
    try:
        usage = result.get("usage", {})
        entry = CacheEntry(
            query_text=query_text,
            embedding=query_emb,
            response=result,
            model=body.get("model", settings.upstream_model),
            input_tokens=usage.get("prompt_tokens", _estimate_tokens(query_text)),
            output_tokens=usage.get("completion_tokens", 0),
        )
        await cache_store.add(entry)
    except Exception as e:
        logger.warning(f"cache add failed: {e}")

    metrics.record_miss(latency_ms=total_latency)

    logger.info(f"⬇️ MISS latency={total_latency:.1f}ms (upstream {upstream_latency:.1f}ms) cached for next time")

    return JSONResponse(
        content=result,
        headers={
            "X-Cache": "MISS",
            "X-Cache-Latency-Ms": f"{total_latency:.2f}",
            "X-Upstream-Latency-Ms": f"{upstream_latency:.2f}",
        },
    )


@app.post("/v1/completions")
async def completions(request: Request, response: Response):
    # legacy эндпоинт - переиспользуем ту же логику, просто формат чуть другой
    start = time.time()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    # приводим к chat формату для кэша
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = " ".join(prompt)
    # эмулируем messages
    fake_body = {"model": body.get("model", settings.upstream_model), "messages": [{"role": "user", "content": str(prompt)}], **body}

    query_text = _extract_query_text(fake_body)
    if not query_text:
        raise HTTPException(status_code=400, detail="empty prompt")

    if body.get("stream"):
        result = await upstream.chat_completion(body)
        metrics.record_miss(latency_ms=(time.time() - start) * 1000)
        return JSONResponse(content=result, headers={"X-Cache": "BYPASS"})

    query_emb = await embedder.embed(query_text)
    search_result = await cache_store.search(query_emb, threshold=settings.similarity_threshold)
    latency_ms = (time.time() - start) * 1000

    if search_result is not None:
        import copy

        out = copy.deepcopy(search_result.entry.response)
        out["_cache_hit"] = True
        out["_cache_similarity"] = round(search_result.similarity, 4)
        usage = out.get("usage", {})
        metrics.record_hit(latency_ms=latency_ms, input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0))
        return JSONResponse(
            content=out,
            headers={"X-Cache": "HIT", "X-Cache-Similarity": f"{search_result.similarity:.4f}"},
        )

    result = await upstream.chat_completion(body)
    try:
        await cache_store.add(
            CacheEntry(
                query_text=query_text,
                embedding=query_emb,
                response=result,
                model=body.get("model", settings.upstream_model),
            )
        )
    except Exception as e:
        logger.warning(f"cache add failed {e}")

    metrics.record_miss(latency_ms=(time.time() - start) * 1000)
    return JSONResponse(content=result, headers={"X-Cache": "MISS"})


# ---------- dashboard ----------


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    # инлайним html чтобы не возиться со статикой
    snap = metrics.snapshot()
    size = await cache_store.size() if cache_store else 0
    return f"""
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic Cache Proxy - Dashboard</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background:#0a0a0b; color:#e8e8e8; padding:32px}}
  h1{{font-size:28px; font-weight:700; letter-spacing:-0.02em}}
  .sub{{color:#9a9a9e; margin-top:6px; font-size:14px}}
  .grid{{display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:16px; margin-top:28px}}
  .card{{background:#141416; border:1px solid #232326; border-radius:16px; padding:20px}}
  .card h3{{font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#9a9a9e}}
  .card .val{{font-size:28px; font-weight:700; margin-top:8px}}
  .card .hint{{font-size:12px; color:#6b6b6f; margin-top:4px}}
  .bar{{height:8px; background:#232326; border-radius:999px; overflow:hidden; margin-top:12px}}
  .bar span{{display:block; height:100%; background: linear-gradient(90deg,#7c3aed,#06b6d4); border-radius:999px}}
  .row{{display:flex; gap:16px; margin-top:16px; flex-wrap:wrap}}
  .wide{{flex:1; min-width:320px}}
  code{{background:#1f1f22; padding:2px 6px; border-radius:6px; font-size:12px}}
  a{{color:#8b9bff}}
  table{{width:100%; border-collapse:collapse; margin-top:12px; font-size:13px}}
  th{{text-align:left; color:#9a9a9e; font-weight:500; padding:8px 0; border-bottom:1px solid #232326}}
  td{{padding:10px 0; border-bottom:1px solid #141416; color:#d6d6d8}}
  .pill{{display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600}}
  .hit{{background:#0f2a1a; color:#34d399; border:1px solid #14532d}}
  .miss{{background:#2a1a1a; color:#f87171; border:1px solid #7f1d1d}}
  footer{{margin-top:32px; color:#6b6b6f; font-size:12px}}
</style>
</head>
<body>
  <h1>⚡ Semantic Cache Proxy</h1>
  <p class="sub">семантический кэш для LLM - режет latency с 1.2с до ~15мс и экономит токены. Обновляется каждую секунду.</p>

  <div class="grid">
    <div class="card">
      <h3>Hit rate</h3>
      <div class="val">{snap['hit_rate_percent']}%</div>
      <div class="hint">{snap['hits']} хитов из {snap['total_requests']} запросов</div>
      <div class="bar"><span style="width:{snap['hit_rate_percent']}%"></span></div>
    </div>
    <div class="card">
      <h3>Latency cached</h3>
      <div class="val">{snap['avg_latency_cached_ms']} ms</div>
      <div class="hint">vs {snap['avg_latency_miss_ms']} ms без кэша - ускорение ×{snap['cache_speedup'] or '-'}</div>
    </div>
    <div class="card">
      <h3>Сэкономлено токенов</h3>
      <div class="val">{snap['total_tokens_saved']:,}</div>
      <div class="hint">{snap['total_input_tokens_saved']:,} in + {snap['total_output_tokens_saved']:,} out</div>
    </div>
    <div class="card">
      <h3>Сэкономлено $</h3>
      <div class="val">${snap['money_saved_usd']}</div>
      <div class="hint">по тарифу ${settings.price_input_per_1k}/1k in, ${settings.price_output_per_1k}/1k out</div>
    </div>
  </div>

  <div class="row">
    <div class="card wide">
      <h3>Кэш</h3>
      <div class="val" style="font-size:20px">{size} записей</div>
      <div class="hint">threshold={settings.similarity_threshold} · ttl={settings.cache_ttl_seconds//3600}ч · backend={settings.vector_backend} · model={settings.embedding_model}</div>
      <table>
        <tr><th>Метрика</th><th>Значение</th></tr>
        <tr><td>Total requests</td><td>{snap['total_requests']}</td></tr>
        <tr><td>Hits / Misses</td><td><span class="pill hit">{snap['hits']} HIT</span> &nbsp; <span class="pill miss">{snap['misses']} MISS</span></td></tr>
        <tr><td>Avg cached latency</td><td>{snap['avg_latency_cached_ms']} ms</td></tr>
        <tr><td>Avg miss latency</td><td>{snap['avg_latency_miss_ms']} ms</td></tr>
      </table>
    </div>
    <div class="card wide">
      <h3>Как это работает</h3>
      <p style="margin-top:10px; font-size:13px; line-height:1.6; color:#b8b8bb">
        1. Клиент шлет <code>POST /v1/chat/completions</code> как в обычный OpenAI API<br>
        2. Прокси превращает текст в эмбеддинг (<code>all-MiniLM-L6-v2</code> ~15мс на CPU)<br>
        3. Ищет ближайший вектор в кэше по косинусу<br>
        4. Если <code>similarity &gt; {settings.similarity_threshold}</code> - отдает кэш за ~15мс<br>
        5. Иначе - прокидывает в upstream, сохраняет ответ и эмбеддинг
      </p>
      <p style="margin-top:12px; font-size:13px">
        <a href="/docs">→ Swagger /docs</a> &nbsp; <a href="/stats">→ JSON /stats</a> &nbsp; <a href="/health">→ /health</a>
      </p>
      <p style="margin-top:12px; font-size:12px; color:#6b6b6f">
        Попробуй: <code>curl -X POST localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":"привет как дела"}}]}}' -i</code>
      </p>
    </div>
  </div>

  <footer>
    semantic-cache-proxy v0.1.0 · сделай <code>docker compose up --build</code> и все заведется · проверь заголовок <code>X-Cache: HIT/MISS</code>
  </footer>

<script>
// автообновление каждую секунду
setTimeout(()=> location.reload(), 2000)
</script>
</body>
</html>
    """


# для прямого запуска python -m app.main
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.app_port, reload=True)
