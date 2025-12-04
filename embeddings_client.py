from __future__ import annotations

import os
from typing import List, Optional

from openai import AsyncOpenAI

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


class EmbeddingsClient:
    """Асинхронный клиент эмбеддингов поверх LiteLLM/OpenAI API."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self.model = model
        self._client = client

    def _require_env(self, name: str) -> str:
        # Перестраховка: четкая ошибка, если не хватает конфигурации.
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Environment variable {name} is required")
        return value

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            base_url = self._require_env("LITELLM_BASE_URL")
            api_key = self._require_env("LITELLM_API_KEY")
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return self._client

    async def embed_text(self, text: str) -> List[float]:
        """Возвращает единичный эмбеддинг строки."""
        client = self._get_client()
        response = await client.embeddings.create(model=self.model, input=[text])
        if not response.data:
            raise RuntimeError("Empty embeddings response")
        return list(response.data[0].embedding)
