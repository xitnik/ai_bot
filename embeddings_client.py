from __future__ import annotations

import asyncio
import os
from typing import Any, List, Optional

from openai import AsyncOpenAI

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
BGE_M3_MODEL = "bge-m3"


class EmbeddingsClient:
    """Unified embeddings client with OpenAI and BGE-M3 backends."""

    def __init__(
        self,
        default_model: str = DEFAULT_EMBEDDING_MODEL,
        client: Optional[AsyncOpenAI] = None,
        bge_model_name: str = BGE_M3_MODEL,
        cache_size: int = 256,
    ) -> None:
        self.default_model = default_model
        self._client = client
        self._bge_model_name = bge_model_name
        self._bge_model: Any = None
        self._cache_size = cache_size
        self._cache: dict[tuple[str, str], List[float]] = {}

    def _require_env(self, name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Environment variable {name} is required")
        return value

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._client is None:
            base_url = self._require_env("LITELLM_BASE_URL")
            api_key = self._require_env("LITELLM_API_KEY")
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return self._client

    def _load_bge_model(self, model_name: str) -> Any:
        if self._bge_model is not None:
            return self._bge_model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is required for BGE embeddings") from exc
        self._bge_model = SentenceTransformer(model_name)
        return self._bge_model

    async def get_text_embedding(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """
        Returns embedding for a single string using configured backend.

        Args:
            text: input text.
            model_name: optional override (e.g., "bge-m3" or OpenAI model id).
        """
        target = model_name or self.default_model
        cache_key = (target, text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)
        if target.startswith("bge"):
            vector = await self._embed_with_bge(text, target)
        else:
            vector = await self._embed_with_openai(text, target)
        if len(self._cache) >= self._cache_size:
            # Популярная эвристика: удалить первый вставленный элемент.
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = list(vector)
        return vector

    async def _embed_with_openai(self, text: str, model_name: str) -> List[float]:
        client = self._get_openai_client()
        response = await client.embeddings.create(model=model_name, input=[text])
        if not response.data:
            raise RuntimeError("Empty embeddings response")
        return list(response.data[0].embedding)

    async def _embed_with_bge(self, text: str, model_name: str) -> List[float]:
        model = self._load_bge_model(model_name)
        vector = await asyncio.to_thread(model.encode, [text], normalize_embeddings=True)
        if not vector:
            raise RuntimeError("Empty embeddings from BGE model")
        return list(vector[0])

    async def embed_text(self, text: str) -> List[float]:
        """Backward-compatible alias."""
        return await self.get_text_embedding(text)
