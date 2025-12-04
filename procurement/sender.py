from __future__ import annotations

import logging
from typing import List, Protocol

from .schemas import Vendor

logger = logging.getLogger(__name__)


class RFQSender(Protocol):
    async def send_rfq(self, vendor: Vendor, payload: dict) -> None:  # pragma: no cover - интерфейс
        ...


class MemoryRFQSender:
    """Простая заглушка, сохраняющая отправленные заявки в памяти."""

    def __init__(self) -> None:
        self.sent: List[tuple[Vendor, dict]] = []

    async def send_rfq(self, vendor: Vendor, payload: dict) -> None:
        # Логируем и кладем в память для тестов.
        logger.info("Stub send RFQ to %s", vendor.name)
        self.sent.append((vendor, payload))


__all__ = ["RFQSender", "MemoryRFQSender"]
