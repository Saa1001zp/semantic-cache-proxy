"""
locust для нагрузочного теста
запуск: locust -f locustfile.py --host http://localhost:8000
или headless: locust -f locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 --run-time 30s
"""
from locust import HttpUser, task, between
import random

# набор запросов - часть повторяющихся чтобы увидеть хитрейт
QUERIES = [
    "привет как дела",
    "привет, как у тебя дела?",
    "что такое fastapi",
    "расскажи про fastapi фреймворк",
    "как приготовить борщ",
    "рецепт борща со свеклой",
    "напиши функцию на python для сортировки",
    "python sort function example",
    "что такое pgvector",
    "объясни что такое pgvector и зачем он нужен",
]

# уникальные чтобы создавать промахи
UNIQUE_QUERIES = [
    f"уникальный запрос номер {i} про {random.choice(['кота','собаку','машину','дом'])}"
    for i in range(100)
]


class CacheUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def chat_hit(self):
        # 70% - повторяющиеся запросы (будут хиты)
        q = random.choice(QUERIES)
        self.client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": q}]},
            name="/v1/chat/completions [hit?]",
        )

    @task(2)
    def chat_miss(self):
        q = random.choice(UNIQUE_QUERIES) + f" {random.randint(0, 9999)}"
        self.client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": q}]},
            name="/v1/chat/completions [miss]",
        )

    @task(1)
    def stats(self):
        self.client.get("/stats", name="/stats")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
