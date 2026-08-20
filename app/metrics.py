import time
import threading
from dataclasses import dataclass, field


@dataclass
class Metrics:
    total_requests: int = 0
    hits: int = 0
    misses: int = 0

    # latency в миллисекундах
    total_latency_cached_ms: float = 0.0
    total_latency_miss_ms: float = 0.0

    total_input_tokens_saved: int = 0
    total_output_tokens_saved: int = 0

    # для вычисления avg
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_hit(self, latency_ms: float, input_tokens: int = 0, output_tokens: int = 0):
        with self._lock:
            self.total_requests += 1
            self.hits += 1
            self.total_latency_cached_ms += latency_ms
            self.total_input_tokens_saved += input_tokens
            self.total_output_tokens_saved += output_tokens

    def record_miss(self, latency_ms: float):
        with self._lock:
            self.total_requests += 1
            self.misses += 1
            self.total_latency_miss_ms += latency_ms

    def snapshot(self) -> dict:
        with self._lock:
            hit_rate = (self.hits / self.total_requests * 100) if self.total_requests else 0
            avg_cached = (self.total_latency_cached_ms / self.hits) if self.hits else 0
            avg_miss = (self.total_latency_miss_ms / self.misses) if self.misses else 0
            # грубая экономия в баксах
            from app.config import settings

            # если токены не считались - берем средние
            input_saved = self.total_input_tokens_saved or (self.hits * settings.avg_input_tokens)
            output_saved = self.total_output_tokens_saved or (self.hits * settings.avg_output_tokens)

            money_saved = (input_saved / 1000 * settings.price_input_per_1k) + (
                output_saved / 1000 * settings.price_output_per_1k
            )

            return {
                "total_requests": self.total_requests,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_percent": round(hit_rate, 2),
                "avg_latency_cached_ms": round(avg_cached, 2),
                "avg_latency_miss_ms": round(avg_miss, 2),
                "total_input_tokens_saved": input_saved,
                "total_output_tokens_saved": output_saved,
                "total_tokens_saved": input_saved + output_saved,
                "money_saved_usd": round(money_saved, 4),
                "cache_speedup": round(avg_miss / avg_cached, 1) if avg_cached and avg_miss else 0,
            }

    def reset(self):
        with self._lock:
            self.total_requests = 0
            self.hits = 0
            self.misses = 0
            self.total_latency_cached_ms = 0
            self.total_latency_miss_ms = 0
            self.total_input_tokens_saved = 0
            self.total_output_tokens_saved = 0


# глобальный синглтон
metrics = Metrics()
