from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict


# Простая структурированная запись в stdout, чтобы потом сменить на OTEL/БД.
def log_event(event_type: str, payload: Dict[str, Any]) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
    }
    entry.update(payload)
    print(json.dumps(entry, ensure_ascii=False))
