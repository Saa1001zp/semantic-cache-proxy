# semantic cache proxy

легкий прокси между твоим кодом и любой LLM. каждый раз платить за api и ждать 1-2 сек бесит, поэтому сделал штуку которая превращает запрос в эмбеддинг, ищет похожий в базе по косинусу и отдает кэш за 15мс.

делал для портфолио, вечерами после работы. если есть идеи как улучшить - кидай pr, буду рад.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Tests](https://img.shields.io/badge/tests-17%20passed-brightgreen)
![Docker](https://img.shields.io/badge/docker-ready-blue)

---

## зачем вообще

две боли с LLM в проде:

1. бабки - каждый запрос это токены. "привет как дела" и "привет, как у тебя дела?" стоят одинаково, хотя ответ один и тот же
2. latency - 1-2 сек на каждый чих, даже если такой вопрос уже задавали 100 раз

прокси решает обе штуки: если `similarity > 0.93` - отдаю кэш за миллисекунды и считаю сколько баксов сэкономил.

кому зайдет:
- хочешь показать что умеешь не просто дергать openai, а делать инфру
- нужен красивый пункт в резюме с графиками
- хочешь сэкономить на токенах в своем проекте

---

## как работает

```
Клиент -> POST /v1/chat/completions -> [ Proxy ]
                                            |
                                 1. текст -> эмбеддинг      all-MiniLM-L6-v2, где-то 15ms на cpu
                                 2. поиск по косинусу       pgvector / memory
                                 3. similarity > 0.93?  - да -> отдать кэш (15ms) + X-Cache: HIT
                                                        - нет -> сходить в LLM, сохранить, отдать
```

- api совместим с openai - просто меняешь base_url на http://localhost:8000/v1
- ищет не по точному совпадению, а по смыслу. "что такое fastapi" и "расскажи про fastapi фреймворк" попадут в один кэш
- протухшие записи сами улетают по TTL, если кэш переполнен - чистит старые

---

## стек

- бэк: FastAPI, async
- вектора: postgres + pgvector (можно и без него, тогда memory на numpy)
- эмбеддинги: sentence-transformers all-MiniLM-L6-v2, опционально onnx, если модели нет - фолбэк на хеш
- прокси: httpx
- остальное: docker, pytest, locust

ничего сверхестественного, просто склеил то что работает.

---

## быстрый старт

```bash
cp .env.example .env
# впиши UPSTREAM_API_KEY если хочешь реальные запросы, без него будет mock

docker compose up --build
# или make up
```

проверяем:

```bash
curl http://localhost:8000/health
# дашборд
open http://localhost:8000/dashboard
# сваггер
open http://localhost:8000/docs

# тест прокси
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "привет как дела"}]}' -i
# первый раз X-Cache: MISS, второй раз HIT
```

без докера:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## конфиг

все через .env:

```ini
UPSTREAM_API_URL=https://api.openai.com/v1/chat/completions
UPSTREAM_API_KEY=sk-xxxx
SIMILARITY_THRESHOLD=0.93   # 0.90 больше хитов, 0.95 строже
CACHE_TTL_SECONDS=604800    # 7 дней
MAX_CACHE_SIZE=10000
VECTOR_BACKEND=memory       # memory, pgvector, redis
EMBEDDING_MODEL=all-MiniLM-L6-v2
USE_ONNX=false
```

хочешь pgvector - просто `docker compose up --build`, он уже настроен. хочешь без постгреса - `docker compose -f docker-compose.memory.yml up --build`

---

## api

### POST /v1/chat/completions

обычный openai запрос, поддерживает stream (стримы пока мимо кэша).

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "что такое fastapi?"}]}'
```

в ответе смотри хедеры:
- X-Cache: HIT + X-Cache-Similarity: 0.97 - из кэша, 15ms
- X-Cache: MISS - сходили в upstream
- X-Cache: BYPASS - стрим

плюс в теле `_cache_hit` и `_cache_similarity` для дебага.

еще есть `POST /v1/completions`, `GET /stats`, `GET /health`, `POST /cache/clear`, `GET /dashboard`.

пример `/stats`:

```json
{
  "total_requests": 1250,
  "hits": 780,
  "hit_rate_percent": 62.4,
  "avg_latency_cached_ms": 14.2,
  "avg_latency_miss_ms": 1240.5,
  "money_saved_usd": 4.68
}
```

---

## дашборд

http://localhost:8000/dashboard - там счетчики хитов, latency, сколько токенов и баксов сэкономил. обновляется сам.

![Dashboard](docs/dashboard.svg)

скрин сгенерил скриптом, живой выглядит так же только с твоими цифрами. делал на коленке, не судите строго.

---

## бенчмарки

запусти `make up` и потом:

```bash
locust -f locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 --run-time 30s
# или проще
python scripts/benchmark.py --requests 200 --concurrency 10
```

у меня на ноуте с mock upstream вышло так:

```
avg          HIT 12.4ms   vs MISS 1120ms   x90 быстрее
p95          HIT 18ms     vs MISS 1350ms
```

```
HIT  : █ 12.4ms
MISS : ████████████████████████████████████████ 1120ms
```

на реальном openai разница еще больше, потому что сеть + генерация.

скрипт сохраняет benchmark_result.json, можешь свой график в ридми кинуть.

---

## тесты

```bash
pytest -v
# 17 тестов, без внешних зависимостей

make test
```

в ci используется `FORCE_DUMMY_EMBEDDING=1` чтобы не качать модель. если хочешь с реальной моделью: `FORCE_DUMMY_EMBEDDING=0 pytest -v`

---

## структура

```
app/main.py              - fastapi, эндпоинты
app/config.py            - настройки из .env
app/cache/memory.py      - кэш на numpy
app/cache/pgvector.py    - постгрес + pgvector, если постгреса нет - падает на memory
app/embeddings/local.py  - грузит minilm, если не получилось - хеш
app/proxy/upstream.py    - дергает llm, без ключа отдает mock
tests/                   - pytest
scripts/benchmark.py     - бенч
```

старался без овер инжиниринга, код простой.

---

## makefile

```
make up        - поднять все
make down      - остановить
make logs      - логи
make dev       - локально без докера
make test      - тесты
make benchmark - прогнать бенч
make locust    - локust ui
```

---

## что дальше

- [x] pgvector уже работает
- [ ] redis вариант если кому надо
- [ ] стрим кэш (сейчас стримы мимо)
- [ ] метрики для прометея
- [ ] auth

если хочешь помочь - кидай issue.

---

## лицензия

MIT, делай что хочешь.

если понравилось - поставь звезду, мне приятно :)

делал с кофе и злостью на лишние токены.
