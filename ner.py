from __future__ import annotations

import re
from typing import Dict, List

PHONE_RE = re.compile(r"(?:\+7|8)?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
INN_RE = re.compile(r"\b\d{10}\b")
SKU_RE = re.compile(r"\bsku[-_]?\d+\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def extract_entities(text: str) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    for regex, etype in [
        (PHONE_RE, "phone"),
        (EMAIL_RE, "email"),
        (INN_RE, "inn"),
        (SKU_RE, "sku"),
        (NUMBER_RE, "number"),
    ]:
        for match in regex.finditer(text):
            entities.append({"type": etype, "value": match.group(0)})
    return entities
