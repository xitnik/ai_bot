from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List, Optional, Sequence

from openai import AsyncOpenAI

from config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL") or "BAAI/bge-m3"
LOCAL_MODELS = {
    "baai/bge-m3",
    "intfloat/multilingual-e5-large",
}


def _normalize_model_name(name: str) -> str:
    return name.lower()


class EmbeddingsClient:
    """Unified embeddings client with selectable RU-friendly backends."""

    def __init__(
        self,
        default_model: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
        cache_size: int = 256,
    ) -> None:
        self.default_model = default_model or get_settings().rag.embedding_model or DEFAULT_EMBEDDING_MODEL
        self._client = client
        self._cache_size = cache_size
        self._cache: dict[tuple[str, str], List[float]] = {}
        self._local_models: dict[str, Any] = {}
        self._embedding_dim: Optional[int] = None
        logger.info("Embeddings client initialized", extra={"model": self.default_model})

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

    def _load_local_model(self, model_name: str) -> Any:
        normalized = _normalize_model_name(model_name)
        if normalized in self._local_models:
            return self._local_models[normalized]
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is required for local embeddings") from exc
        model = SentenceTransformer(model_name)
        self._local_models[normalized] = model
        return model

    def _is_local(self, model_name: str) -> bool:
        return _normalize_model_name(model_name) in LOCAL_MODELS

    async def get_embeddings(self, texts: Sequence[str], model_name: Optional[str] = None) -> List[List[float]]:
        """
        Returns embeddings for a batch of texts.

        Args:
            texts: list of strings to embed.
            model_name: optional override of embedding model.
        """
        if not texts:
            return []
        target = model_name or self.default_model
        vectors: List[List[float]] = []
        for text in texts:
            vectors.append(await self.get_text_embedding(text, target))
        return vectors

    async def get_text_embedding(self, text: str, model_name: Optional[str] = None) -> List[float]:
        target = model_name or self.default_model
        cache_key = (target, text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)
        if self._is_local(target):
            vector = await self._embed_with_local_model(text, target)
        else:
            vector = await self._embed_with_openai(text, target)
        self._validate_dimension(vector)
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = list(vector)
        return vector

    async def _embed_with_openai(self, text: str, model_name: str) -> List[float]:
        client = self._get_openai_client()
        response = await client.embeddings.create(model=model_name, input=[text])
        if not response.data:
            raise RuntimeError("Empty embeddings response")
        vector = list(response.data[0].embedding)
        return vector

    async def _embed_with_local_model(self, text: str, model_name: str) -> List[float]:
        model = self._load_local_model(model_name)
        vector = await asyncio.to_thread(model.encode, [text], normalize_embeddings=True)
        if not vector:
            raise RuntimeError("Empty embeddings from local model")
        return list(vector[0])

    def _validate_dimension(self, vector: Sequence[float]) -> None:
        dim = len(vector)
        if self._embedding_dim is None:
            self._embedding_dim = dim
            return
        if dim != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, got {dim}"
            )

    async def embed_text(self, text: str) -> List[float]:
        """Backward-compatible alias."""
        return await self.get_text_embedding(text)
