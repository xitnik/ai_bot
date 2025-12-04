from __future__ import annotations

import re
from typing import Dict, List

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


def _convert_unit(value: float, unit: str) -> float:
    unit_lower = unit.lower()
    if unit_lower in ("cm", "см"):
        return value * 10.0
    if unit_lower == "m":
        return value * 1000.0
    return value


def _extract_dimensions(text: str) -> List[Dict[str, str]]:
    dims: List[Dict[str, str]] = []
    for match in DIM_RE.finditer(text):
        nums = [n for n in match.groups() if n]
        if len(nums) < 2:
            continue
        unit_match = UNIT_RE.search(text)
        unit = unit_match.group(0) if unit_match else "mm"
        vals_mm = [_convert_unit(float(num.replace(",", ".")), unit) for num in nums]
        length, width, thickness = vals_mm[0], vals_mm[1], vals_mm[2] if len(vals_mm) > 2 else 25.0
        dims.append(
            {
                "type": "dimensions",
                "value": f"{int(length)}x{int(width)}x{int(thickness)}mm",
                "length_mm": str(length),
                "width_mm": str(width),
                "thickness_mm": str(thickness),
            }
        )
    return dims


def _match_from_dict(text: str, mapping: Dict[str, str], etype: str) -> List[Dict[str, str]]:
    found: List[Dict[str, str]] = []
    lowered = text.lower()
    for alias, canonical in mapping.items():
        if alias in lowered:
            found.append({"type": etype, "value": canonical, "alias": alias})
    return found


def extract_entities(text: str) -> List[Dict[str, str]]:
    """
    Расширенный baseline NER:
    - контактные данные (phone/email/ИНН/SKU/числа)
    - дерево/применение/обработка по словарю
    - размеры с нормализацией единиц
    """
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
    entities.extend(_match_from_dict(text, WOOD_SPECIES, "wood_species"))
    entities.extend(_match_from_dict(text, APPLICATION_SYNONYMS, "application"))
    entities.extend(_match_from_dict(text, PROCESSING_SYNONYMS, "processing"))
    entities.extend(_extract_dimensions(text))
    return entities
