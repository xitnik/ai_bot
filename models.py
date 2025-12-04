from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """Входящее сообщение от канала."""

    user_id: str
    channel: str
    text: str
    attachments: Optional[List[Any]] = Field(default=None)


class SessionDTO(BaseModel):
    """DTO сессии, синхронизированной с БД."""

    session_id: str
    user_id: str
    channel: str
    state: str
    started_at: datetime
    last_event_at: datetime

    model_config = ConfigDict(from_attributes=True)
