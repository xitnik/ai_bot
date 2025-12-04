from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, Optional

from config import get_settings


def init_logging(service: str) -> None:
    handler = logging.StreamHandler(sys.stdout)

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            payload: Dict[str, Any] = {
                "service": service,
                "level": record.levelname,
                "message": record.getMessage(),
            }
            for key in ("trace_id", "session_id", "user_id"):
                value = getattr(record, key, None)
                if value:
                    payload[key] = value
            return json.dumps(payload, ensure_ascii=False)

    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def get_service_name() -> str:
    return get_settings().observability.service_name
