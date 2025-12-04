from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

# Загружаем .env при импорте, чтобы локальный запуск был воспроизводим.
load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass
class LLMSettings:
    base_url: str = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
    api_key: str = os.getenv("LITELLM_API_KEY", "dummy-key")
    yandex_api_key: str = os.getenv("YANDEX_API_KEY", "dummy-yandex")
    yc_folder_id: str = os.getenv("YC_FOLDER_ID", "demo-folder")
    lora_adapter_id: str = os.getenv("LORA_ADAPTER_ID", "demo-adapter")


@dataclass
class IntegrationSettings:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "fake-telegram-token")
    max_api_token: str = os.getenv("MAX_API_TOKEN", "fake-max-token")
    avito_api_token: str = os.getenv("AVITO_API_TOKEN", "fake-avito-token")
    bitrix24_webhook_url: str = os.getenv("BITRIX24_WEBHOOK_URL", "https://example.bitrix24.ru/rest/1/webhook")
    onec_base_url: str = os.getenv("ONEC_BASE_URL", "http://localhost:8200")
    agents_base_url: str = os.getenv("AGENTS_BASE_URL", "")
    integrations_base_url: str = os.getenv("INTEGRATIONS_BASE_URL", "")
    sales_agents_base_url: str = os.getenv("SALES_AGENTS_BASE_URL", "http://localhost:8000")


@dataclass
class ObservabilitySettings:
    service_name: str = os.getenv("SERVICE_NAME", "bot-suite")
    otlp_endpoint: Optional[str] = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    metrics_enabled: bool = _bool("METRICS_ENABLED", True)


@dataclass
class DatabaseSettings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./bot.db")
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "bot")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "bot")
    mysql_db: str = os.getenv("MYSQL_DB", "bot")
    mysql_pool_size: int = int(os.getenv("MYSQL_POOL_SIZE", "10"))
    mysql_echo: bool = _bool("MYSQL_ECHO", False)
    procurement_db: str = os.getenv("PROCUREMENT_MYSQL_DB", os.getenv("MYSQL_DB", "bot"))
    procurement_database_url: str = os.getenv(
        "PROCUREMENT_DATABASE_URL", "sqlite+aiosqlite:///./procurement.db"
    )

    def async_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"mysql+asyncmy://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
        )

    def procurement_async_url(self) -> str:
        if self.procurement_database_url:
            return self.procurement_database_url
        return (
            f"mysql+asyncmy://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.procurement_db}"
        )


@dataclass
class DeepResearchSettings:
    search_base_url: str = os.getenv("DEEP_RESEARCH_SEARCH_BASE_URL", "")
    search_api_key: str = os.getenv("DEEP_RESEARCH_SEARCH_API_KEY", "")
    search_timeout_seconds: float = float(os.getenv("DEEP_RESEARCH_SEARCH_TIMEOUT", "8.0"))
    search_top_k: int = int(os.getenv("DEEP_RESEARCH_SEARCH_TOP_K", "5"))
    search_max_results: int = int(os.getenv("DEEP_RESEARCH_SEARCH_MAX_RESULTS", "12"))
    model: str = os.getenv("DEEP_RESEARCH_MODEL", "gpt5")
    temperature: float = float(os.getenv("DEEP_RESEARCH_TEMPERATURE", "0.2"))
    max_output_tokens: int = int(os.getenv("DEEP_RESEARCH_MAX_OUTPUT_TOKENS", "1500"))


@dataclass
class RagVectorSettings:
    backend: str = os.getenv("RAG_VECTOR_BACKEND", "chroma")
    persist_path: str = os.getenv("RAG_VECTOR_PERSIST_PATH", "./data/chroma")
    collection_name: str = os.getenv("RAG_VECTOR_COLLECTION", "rag_chunks")
    qdrant_url: str = os.getenv("RAG_QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("RAG_QDRANT_API_KEY", "")
    pgvector_dsn: str = os.getenv("RAG_PGVECTOR_DSN", "")


@dataclass
class RAGSettings:
    default_mode: str = os.getenv("RAG_MODE", "basic")  # basic | self-rag | crag
    max_selfrag_iterations: int = int(os.getenv("RAG_SELF_RAG_MAX_ITERS", "2"))
    max_crag_retries: int = int(os.getenv("RAG_CRAG_MAX_RETRIES", "2"))
    retriever_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.0"))
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "qwen2.5-7b-instruct")
    llm_temperature: float = float(os.getenv("RAG_LLM_TEMPERATURE", "0.2"))
    llm_max_tokens: int = int(os.getenv("RAG_LLM_MAX_TOKENS", "512"))
    retriever_knn_top_k: int = int(os.getenv("RAG_RETRIEVER_KNN_TOP_K", "50"))
    retriever_final_top_k: int = int(os.getenv("RAG_RETRIEVER_FINAL_TOP_K", "8"))
    reranker_enabled: bool = _bool("RAG_RERANKER_ENABLED", True)
    reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    enable_knowledge_graph: bool = _bool("RAG_ENABLE_KG", True)
    simple_query_max_tokens: int = int(os.getenv("RAG_SIMPLE_QUERY_MAX_TOKENS", "3"))
    cache_enabled: bool = _bool("RAG_CACHE_ENABLED", True)
    ner_cache_enabled: bool = _bool("NER_CACHE_ENABLED", True)
    vector: RagVectorSettings = field(default_factory=RagVectorSettings)

@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    integrations: IntegrationSettings = field(default_factory=IntegrationSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    deep_research: DeepResearchSettings = field(default_factory=DeepResearchSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает кэшированный слепок конфигурации."""
    return Settings()
