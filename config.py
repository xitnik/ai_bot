from __future__ import annotations

import os
from dataclasses import dataclass
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
    procurement_url: str = os.getenv("PROCUREMENT_DATABASE_URL", "sqlite+aiosqlite:///./procurement.db")


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
class RAGSettings:
    default_mode: str = os.getenv("RAG_MODE", "basic")  # basic | self-rag | crag
    max_selfrag_iterations: int = int(os.getenv("RAG_SELF_RAG_MAX_ITERS", "2"))
    max_crag_retries: int = int(os.getenv("RAG_CRAG_MAX_RETRIES", "2"))
    retriever_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.0"))
    enable_knowledge_graph: bool = _bool("RAG_ENABLE_KG", True)
    simple_query_max_tokens: int = int(os.getenv("RAG_SIMPLE_QUERY_MAX_TOKENS", "3"))
    cache_enabled: bool = _bool("RAG_CACHE_ENABLED", True)
    ner_cache_enabled: bool = _bool("NER_CACHE_ENABLED", True)


@dataclass
class Settings:
    llm: LLMSettings = LLMSettings()
    integrations: IntegrationSettings = IntegrationSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    database: DatabaseSettings = DatabaseSettings()
    deep_research: DeepResearchSettings = DeepResearchSettings()
    rag: RAGSettings = RAGSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает кэшированный слепок конфигурации."""
    return Settings()
