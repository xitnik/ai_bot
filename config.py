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
class Settings:
    llm: LLMSettings = LLMSettings()
    integrations: IntegrationSettings = IntegrationSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    database: DatabaseSettings = DatabaseSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает кэшированный слепок конфигурации."""
    return Settings()
