from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from config import get_settings
from llm_client import chat

PHONE_RE = re.compile(r"(?:\+7|8)?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
INN_RE = re.compile(r"\b\d{10}\b")
SKU_RE = re.compile(r"\bsku[-_]?\d+\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
DIM_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)[\s×xX*](\d+(?:[.,]\d+)?)(?:[\s×xX*](\d+(?:[.,]\d+)?))?"
)
UNIT_RE = re.compile(r"\b(мм|cm|см|mm|m)\b", re.IGNORECASE)

WOOD_SPECIES = {
    "дуб": "oak",
    "oak": "oak",
    "сосна": "pine",
    "pine": "pine",
    "береза": "birch",
    "birch": "birch",
    "лиственница": "larch",
    "larch": "larch",
    "ель": "spruce",
    "spruce": "spruce",
}

APPLICATION_SYNONYMS = {
    "вагонка": "panel",
    "евровагонка": "panel",
    "доска пола": "floor_board",
    "половая доска": "floor_board",
    "брус": "timber",
    "брусок": "timber",
}

PROCESSING_SYNONYMS = {
    "строганая": "planed",
    "шлифованная": "sanded",
    "сухая": "kiln_dried",
    "сушка": "kiln_dried",
}


class NerMode(str, Enum):
    fast = "fast"
    llm = "llm"
    hybrid = "hybrid"


@dataclass
class NerEntity:
    text: str
    span: Tuple[int, int]
    type: str
    value: Optional[str] = None
    confidence: float = 1.0
    source: str = "fast"

    def to_dict(self) -> Dict[str, str]:
        data = {
            "type": self.type,
            "value": self.value or self.text,
            "start": str(self.span[0]),
            "end": str(self.span[1]),
            "source": self.source,
            "confidence": str(self.confidence),
        }
        return data


@dataclass
class NerResult:
    entities: List[NerEntity]

    def to_legacy(self) -> List[Dict[str, str]]:
        return [{"type": e.type, "value": e.value or e.text} for e in self.entities]


_NER_CACHE: Dict[Tuple[str, NerMode], NerResult] = {}
_NER_CACHE_LIMIT = 256


async def extract_entities_async(text: str, mode: NerMode = NerMode.fast) -> NerResult:
    """Main async NER entrypoint with modes fast/llm/hybrid."""
    settings = get_settings().rag
    cache_key = (text, mode)
    if settings.ner_cache_enabled and cache_key in _NER_CACHE:
        return _NER_CACHE[cache_key]

    if mode == NerMode.fast:
        result = _fast_entities(text)
    elif mode == NerMode.llm:
        result = await _llm_entities(text)
    else:
        result = await _hybrid_entities(text)

    if settings.ner_cache_enabled:
        if len(_NER_CACHE) >= _NER_CACHE_LIMIT:
            _NER_CACHE.pop(next(iter(_NER_CACHE)))
        _NER_CACHE[cache_key] = result
    return result


def extract_entities(text: str) -> List[Dict[str, str]]:
    """
    Legacy sync wrapper used across the codebase.
    Defaults to fast mode and returns list[dict] for backward compatibility.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Best effort: run fast extractor synchronously.
        return _fast_entities(text).to_legacy()
    return asyncio.run(extract_entities_async(text)).to_legacy()


async def _hybrid_entities(text: str) -> NerResult:
    """Fast first, fall back to LLM for missing/low-confidence entities."""
    fast_res = _fast_entities(text)
    types_seen = {ent.type for ent in fast_res.entities}
    if {"org", "product", "date"} <= types_seen:
        return fast_res
    llm_res = await _llm_entities(text)
    merged = _merge_entities(fast_res.entities, llm_res.entities)
    return NerResult(entities=merged)


def _fast_entities(text: str) -> NerResult:
    entities: List[NerEntity] = []
    for regex, etype in [
        (PHONE_RE, "phone"),
        (EMAIL_RE, "email"),
        (INN_RE, "inn"),
        (SKU_RE, "sku"),
        (NUMBER_RE, "number"),
    ]:
        for match in regex.finditer(text):
            entities.append(
                NerEntity(text=match.group(0), span=(match.start(), match.end()), type=etype, source="fast")
            )
    entities.extend(_dict_to_entities(text, WOOD_SPECIES, "wood_species"))
    entities.extend(_dict_to_entities(text, APPLICATION_SYNONYMS, "application"))
    entities.extend(_dict_to_entities(text, PROCESSING_SYNONYMS, "processing"))
    entities.extend(_dimensions_to_entities(text))
    return NerResult(entities=entities)


async def _llm_entities(text: str) -> NerResult:
    """LLM-based NER (prompt-style) with JSON output; falls back to empty on failure."""
    prompt = (
        "Извлеки сущности из текста. Типы: ORG, PRODUCT, DATE, PRICE, QUANTITY, ADDRESS, INN.\n"
        "Ответь JSON: [{\"text\":..., \"type\":..., \"value\":..., \"confidence\":0.0}]."
    )
    try:
        completion = await asyncio.to_thread(
            chat,
            "gpt5",
            [{"role": "system", "content": prompt}, {"role": "user", "content": text}],
            temperature=0,
            max_tokens=300,
        )
        raw = getattr(completion.choices[0].message, "content", "[]")
        parsed = json.loads(raw)
        entities: List[NerEntity] = []
        for item in parsed if isinstance(parsed, list) else []:
            try:
                ent = NerEntity(
                    text=item.get("text", ""),
                    span=(0, 0),
                    type=item.get("type", "unknown"),
                    value=item.get("value"),
                    confidence=float(item.get("confidence", 0.5)),
                    source="llm",
                )
                entities.append(ent)
            except Exception:
                continue
        return NerResult(entities=entities)
    except Exception:
        return NerResult(entities=[])


def _convert_unit(value: float, unit: str) -> float:
    unit_lower = unit.lower()
    if unit_lower in ("cm", "см"):
        return value * 10.0
    if unit_lower == "m":
        return value * 1000.0
    return value


def _dimensions_to_entities(text: str) -> List[NerEntity]:
    entities: List[NerEntity] = []
    for match in DIM_RE.finditer(text):
        nums = [n for n in match.groups() if n]
        if len(nums) < 2:
            continue
        unit_match = UNIT_RE.search(text)
        unit = unit_match.group(0) if unit_match else "mm"
        vals_mm = [_convert_unit(float(num.replace(",", ".")), unit) for num in nums]
        length, width, thickness = vals_mm[0], vals_mm[1], vals_mm[2] if len(vals_mm) > 2 else 25.0
        value = f"{int(length)}x{int(width)}x{int(thickness)}mm"
        entities.append(
            NerEntity(
                text=value,
                span=(match.start(), match.end()),
                type="dimensions",
                value=value,
                confidence=0.9,
                source="fast",
            )
        )
    return entities


def _dict_to_entities(text: str, mapping: Dict[str, str], etype: str) -> List[NerEntity]:
    found: List[NerEntity] = []
    lowered = text.lower()
    for alias, canonical in mapping.items():
        idx = lowered.find(alias)
        if idx >= 0:
            found.append(
                NerEntity(
                    text=alias,
                    span=(idx, idx + len(alias)),
                    type=etype,
                    value=canonical,
                    confidence=0.8,
                    source="fast",
                )
            )
    return found


def _merge_entities(primary: List[NerEntity], secondary: List[NerEntity]) -> List[NerEntity]:
    merged = list(primary)
    covered = {(e.span, e.type): e for e in primary}
    for ent in secondary:
        key = (ent.span, ent.type)
        if key in covered and covered[key].confidence >= ent.confidence:
            continue
        merged.append(ent)
    return merged
