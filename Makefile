.PHONY: help up down build logs test lint benchmark clean

# цвета для красоты
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

help: ## показать помощь
	@echo "$(GREEN)Semantic Cache Proxy$(NC) - команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-12s$(NC) %s\n", $$1, $$2}'

up: ## поднять всё одной командой
	docker compose up --build -d
	@echo "$(GREEN)✓ поднялось!$(NC) проверь:"
	@echo "  curl http://localhost:8000/health"
	@echo "  open http://localhost:8000/dashboard"

down: ## остановить
	docker compose down

build: ## пересобрать образ
	docker compose build

logs: ## логи прокси
	docker compose logs -f proxy

dev: ## запустить локально без докера (для разработки)
	uvicorn app.main:app --reload --port 8000

install: ## поставить зависимости
	pip install -r requirements.txt

test: ## прогнать тесты
	pytest -v

test-cov: ## тесты с покрытием
	pytest -v --cov=app --cov-report=term-missing

lint: ## проверить форматирование (если есть ruff)
	python -m ruff check app tests || echo "ruff not installed, skip"

benchmark: ## нагрузочный тест (нужен запущенный сервис)
	python scripts/benchmark.py --requests 200 --concurrency 10

locust: ## запустить locust ui
	locust -f locustfile.py --host http://localhost:8000

clean: ## почистить кэш
	docker compose down -v 2>nul || docker-compose down -v
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>nul || true

# шорткаты
start: up
stop: down
restart: down up
