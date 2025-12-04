from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from llm_client import chat

from . import schemas

logger = logging.getLogger(__name__)


async def parse_vendor_reply(raw_text: str) -> schemas.OfferCore:
    """
    Парсит свободный ответ поставщика с помощью LLM в структуру OfferCore.
    """

    system_prompt = (
        "Ты извлекаешь структуру цены/сроков из писем поставщиков."
        " Верни JSON, даже если часть полей отсутствует."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Извлеки цену за единицу, минимальную партию, срок поставки в днях"
                " и выпиши текст условий."
                " Если нет данных по полю — ставь null."
                " Письмо: "
                f"{raw_text}"
            ),
        },
    ]

    try:
        completion = await asyncio.to_thread(
            chat,
            "gpt5",
            messages,
            response_format={
                "type": "json_schema",
                "json_schema": schemas.offer_core_json_schema(),
            },
            temperature=0,
        )
    except Exception:
        # Логируем, чтобы было понятно, почему парсинг упал.
        logger.exception("LLM parsing failed")
        raise

    payload: Optional[dict[str, Any]] = None
    choice = completion.choices[0] if getattr(completion, "choices", None) else None
    message = getattr(choice, "message", None) if choice else None
    if message is not None:
        # В новых версиях SDK json_mode кладет результат в parsed, иначе в content.
        if hasattr(message, "parsed") and message.parsed:
            payload = message.parsed  # type: ignore[assignment]
        else:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    payload = None
            elif isinstance(content, dict):
                payload = content
    if payload is None:
        # Если модель вернула что-то неразбираемое, подставляем минимальный ответ.
        logger.warning("LLM returned no parseable payload")
        payload = {}

    data = {
        "price_per_unit": payload.get("price_per_unit"),
        "min_batch": payload.get("min_batch"),
        "lead_time_days": payload.get("lead_time_days"),
        # Если нет terms_text, сохраняем сырое письмо, чтобы не потерять контекст.
        "terms_text": payload.get("terms_text") or raw_text,
    }
    return schemas.OfferCore(**data)


__all__ = ["parse_vendor_reply"]
