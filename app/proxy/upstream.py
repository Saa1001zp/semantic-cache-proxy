"""
клиент для upstream llm - дергает настоящую нейронку если настроена,
иначе отдает мок (чтобы можно было демо без ключей)
"""
import time
import uuid
import logging
import httpx

logger = logging.getLogger(__name__)


class UpstreamClient:
    def __init__(self, api_url: str = "", api_key: str = "", model: str = "gpt-4o-mini", timeout: float = 30.0):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def chat_completion(self, body: dict) -> dict:
        """
        делает запрос к upstream.
        если UPSTREAM_API_URL пустой - возвращает мок ответ (эхо)
        """
        start = time.time()

        # мок режим - удобно для тестов и демо без ключей
        if not self.api_url or not self.api_key:
            return self._mock_response(body)

        # реальный прокси
        try:
            client = await self._get_client()
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            # если body содержит model - оставляем, иначе подставляем дефолт
            if "model" not in body:
                body["model"] = self.model

            resp = await client.post(self.api_url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            logger.debug(f"upstream ok latency={(time.time()-start)*1000:.1f}ms")
            return data
        except httpx.HTTPStatusError as e:
            logger.error(f"upstream http error {e.response.status_code}: {e.response.text[:500]}")
            # пробрасываем как мок с ошибкой чтобы клиент видел что случилось
            # но лучше вернуть ошибку прокси
            raise
        except Exception as e:
            logger.error(f"upstream failed: {e}")
            # фолбэк на мок чтобы не падать полностью
            return self._mock_response(body, error=str(e))

    def _mock_response(self, body: dict, error: str | None = None) -> dict:
        messages = body.get("messages", [])
        # берем последний user message
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        if not last_user and messages:
            # fallback: склеить все
            last_user = " ".join([m.get("content", "") for m in messages])

        # если сообщения в старом формате completions
        if not last_user:
            last_user = body.get("prompt", "hello")

        mock_text = f"[MOCK] Эхо: {last_user[:200]}"
        if error:
            mock_text += f" (upstream error: {error})"

        # считаем токены грубо
        prompt_tokens = len(last_user.split()) * 2  # ~2 токена на слово грубо
        completion_tokens = len(mock_text.split()) * 2

        return {
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", self.model),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": mock_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "_mock": True,
        }

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
