import os

# форсим dummy чтобы тесты не качали модель
os.environ["FORCE_DUMMY_EMBEDDING"] = "1"

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app import config as config_module
from app import main as main_module
from app.metrics import metrics


@pytest.fixture(autouse=True)
async def reset_state():
    # сбрасываем метрики и кэш перед каждым тестом
    metrics.reset()
    # дропаем кэш если уже создан
    if main_module.cache_store is not None:
        await main_module.cache_store.clear()
    yield
    metrics.reset()
    if main_module.cache_store is not None:
        await main_module.cache_store.clear()


@pytest.fixture
async def client():
    # lifespan не запускается автоматом в тестах - инициализируем руками
    # создаем минимальный окружение
    from app.cache.memory import MemoryStore
    from app.proxy.upstream import UpstreamClient
    from app.embeddings.dummy import DummyEmbedding

    # подменяем глобалы
    main_module.cache_store = MemoryStore(ttl_seconds=3600, max_size=1000)
    main_module.upstream = UpstreamClient(api_url="", api_key="")  # mock
    main_module.embedder = DummyEmbedding()

    # снижаем threshold для dummy - у dummy семантика слабее
    orig_threshold = config_module.settings.similarity_threshold
    config_module.settings.similarity_threshold = 0.85

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    config_module.settings.similarity_threshold = orig_threshold
