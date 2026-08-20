"""
простой бенчмарк без locust - меряет latency кэша vs промаха
запуск: python scripts/benchmark.py --requests 100 --concurrency 10

рисует в консоль табличку и сохраняет json для ридми графика
"""
import argparse
import asyncio
import time
import json
import statistics
import httpx

DEFAULT_URL = "http://localhost:8000"


async def single_request(client: httpx.AsyncClient, url: str, payload: dict):
    start = time.perf_counter()
    r = await client.post(f"{url}/v1/chat/completions", json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000
    cache = r.headers.get("X-Cache", "UNKNOWN")
    return elapsed_ms, cache, r.status_code


async def benchmark(url: str, requests: int, concurrency: int):
    # сначала прогреваем кэш одним запросом
    warm_payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "привет как дела"}]}
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(f"{url}/cache/clear")
        await client.post(f"{url}/v1/chat/completions", json=warm_payload)
        print("✓ кэш прогрет одним запросом")

        # теперь меряем хиты (тот же запрос)
        print(f"\n--- HIT тест ({requests} запросов, concurrency={concurrency}) ---")
        hit_latencies = []
        sem = asyncio.Semaphore(concurrency)

        async def hit_task():
            async with sem:
                ms, cache, code = await single_request(client, url, warm_payload)
                return ms

        # батчами
        tasks = [hit_task() for _ in range(requests)]
        hit_latencies = await asyncio.gather(*tasks)

        # меряем миссы (уникальные запросы)
        print(f"\n--- MISS тест ({requests} уникальных запросов) ---")
        miss_latencies = []

        async def miss_task(i):
            async with sem:
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": f"уникальный бенчмарк запрос {i} {time.time()}"}],
                }
                ms, cache, code = await single_request(client, url, payload)
                return ms

        miss_tasks = [miss_task(i) for i in range(requests)]
        miss_latencies = await asyncio.gather(*miss_tasks)

    def stats(arr):
        return {
            "avg": statistics.mean(arr),
            "p50": statistics.median(arr),
            "p95": sorted(arr)[int(len(arr) * 0.95)] if len(arr) > 1 else arr[0],
            "p99": sorted(arr)[int(len(arr) * 0.99)] if len(arr) > 1 else arr[0],
            "min": min(arr),
            "max": max(arr),
        }

    hit_s = stats(hit_latencies)
    miss_s = stats(miss_latencies)
    speedup = miss_s["avg"] / hit_s["avg"] if hit_s["avg"] else 0

    print("\n" + "=" * 60)
    print(f"{'метрика':<12} {'HIT (кэш)':<15} {'MISS (upstream)':<15} speedup")
    print("-" * 60)
    for k in ["avg", "p50", "p95", "p99", "min", "max"]:
        print(f"{k:<12} {hit_s[k]:<15.1f} {miss_s[k]:<15.1f} x{speedup:.1f}")
    print("=" * 60)
    print(f"\n💰 экономия: при hit_rate 60% из 10k запросов ~6k * ~400 токенов = 2.4M токенов")
    print(f"   при $0.005/1k in + $0.015/1k out ~ $36 сэкономлено на 10k запросов")
    print(f"\n⚡ ускорение: {speedup:.1f}x (с {miss_s['avg']:.0f}ms до {hit_s['avg']:.0f}ms)")

    # сохраняем для ридми
    out = {
        "hit": hit_s,
        "miss": miss_s,
        "speedup": speedup,
        "requests": requests,
        "concurrency": concurrency,
        "timestamp": time.time(),
    }
    with open("benchmark_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n→ сохранено в benchmark_result.json")

    # ascii график для ридми
    print("\nГрафик latency (ms):")
    max_val = max(miss_s["max"], hit_s["max"])
    scale = 40 / max_val if max_val else 1
    print(f"HIT  : {'█' * int(hit_s['avg'] * scale)} {hit_s['avg']:.1f}ms")
    print(f"MISS : {'█' * int(miss_s['avg'] * scale)} {miss_s['avg']:.0f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(benchmark(args.url, args.requests, args.concurrency))
