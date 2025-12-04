from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from fastapi import FastAPI

from llm_client import get_client
from procurement.api import router as procurement_router
from procurement.db import init_db

app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    # Создаем таблицы при старте приложения для простоты запуска.
    await init_db()


@app.get("/llm/health")
async def llm_health() -> Dict[str, List[str]]:
    """Проверка доступности LiteLLM через список моделей."""
    client = get_client()
    # Вызываем синхронный SDK в отдельном потоке, чтобы не блокировать event loop.
    models_response = await asyncio.to_thread(client.models.list)
    names: List[str] = []
    for item in getattr(models_response, "data", []):
        model_id: Any = getattr(item, "id", None)
        if model_id is None and isinstance(item, dict):
            model_id = item.get("id")
        if model_id:
            names.append(str(model_id))
    return {"models": names}


app.include_router(procurement_router)
