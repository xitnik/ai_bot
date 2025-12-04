from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

# Ленивая и потокобезопасная инициализация клиента для повторного использования.
_client: Optional[OpenAI] = None
_client_lock = threading.Lock()

__all__ = ["get_client", "chat", "embeddings"]


def _require_env(name: str) -> str:
    """Возвращает обязательную переменную окружения или поднимает исключение."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def get_client() -> OpenAI:
    """Создает или возвращает singleton OpenAI клиента, указывающего на LiteLLM."""
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            base_url = _require_env("LITELLM_BASE_URL")
            api_key = _require_env("LITELLM_API_KEY")
            _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client


def chat(model: str, messages: List[Dict[str, Any]], **kwargs: Any) -> ChatCompletion:
    """Тонкая обертка над chat.completions.create."""
    client = get_client()
    return client.chat.completions.create(model=model, messages=messages, **kwargs)


def embeddings(model: str, inputs: List[str]) -> List[List[float]]:
    """Возвращает список эмбеддингов, скрывая детали ответа OpenAI SDK."""
    client = get_client()
    response = client.embeddings.create(model=model, input=inputs)
    vectors: List[List[float]] = []
    for item in response.data:
        vectors.append(list(item.embedding))
    return vectors


if __name__ == "__main__":
    try:
        completion = chat("gpt5", [{"role": "user", "content": "Say 'pong'."}])
        text = None
        if completion.choices:
            first_choice = completion.choices[0]
            text = first_choice.message.content if getattr(first_choice, "message", None) else None
            if text is None:
                text = getattr(first_choice, "text", None)
        print(f"Completion: {text}")
    except Exception as exc:  # pragma: no cover - ручная проверка
        print(f"Self-test failed: {exc}")
