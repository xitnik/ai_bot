from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rag.pipeline import SessionContext as RagSessionContext
from rag.pipeline import rag_retrieve

CURRENCY = "USD"

SPECIES_MULTIPLIERS: Dict[str, float] = {
    "oak": 1.5,
    "дуб": 1.5,
    "pine": 1.0,
    "сосна": 1.0,
    "birch": 1.3,
    "береза": 1.3,
    "larch": 1.2,
    "лиственница": 1.2,
}

PROCESSING_MULTIPLIERS: Dict[str, float] = {
    "sanding": 0.15,
    "шлифовка": 0.15,
    "painting": 0.25,
    "окраска": 0.25,
    "custom_cut": 0.10,
    "распил": 0.10,
}

RUSH_MULTIPLIERS: Dict[str, float] = {
    "standard": 1.0,
    "fast": 1.15,
    "urgent": 1.25,
}

DEFAULT_PRICE_PER_BF = 25.0
DEFAULT_WASTE_PERCENT = 8.0
BULK_DISCOUNT_LADDER = [
    (50.0, 5.0),
    (100.0, 8.0),
    (200.0, 12.0),
]


class Dimensions(BaseModel):
    length_mm: float
    width_mm: float
    thickness_mm: float


class PricingItem(BaseModel):
    species: str = "pine"
    grade: Optional[str] = None
    dimensions: Optional[Dimensions] = None
    board_feet: Optional[float] = None
    quantity: float = 1.0
    price_per_board_foot: Optional[float] = None
    processing: List[str] = Field(default_factory=list)
    waste_percent: Optional[float] = None
    rush_level: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class PricingRequest(BaseModel):
    """Толерантный к формату запрос: order_spec с items или одиночным словарем."""

    order_spec: Dict[str, object] = Field(default_factory=dict)
    message: Optional[str] = None
    context: Dict[str, object] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class PricingItemBreakdown(BaseModel):
    item_index: int
    species: str
    grade: Optional[str]
    volume_board_feet: float
    base_price: float
    species_multiplier: float
    processing_multiplier: float
    waste_multiplier: float
    rush_multiplier: float
    subtotal: float
    discount_applied: float
    final_price: float
    notes: List[str] = Field(default_factory=list)


class PricingResponse(BaseModel):
    currency: str = CURRENCY
    total: float
    discounts: List[str] = Field(default_factory=list)
    items: List[PricingItemBreakdown] = Field(default_factory=list)
    context_snippets: List[str] = Field(default_factory=list)


def mm_to_inches(value: float) -> float:
    return value / 25.4


def calculate_board_feet(dim: Dimensions, qty: float) -> float:
    """Рассчитывает board feet при размерах в мм и количестве."""
    length_in = mm_to_inches(dim.length_mm)
    width_in = mm_to_inches(dim.width_mm)
    thickness_in = mm_to_inches(dim.thickness_mm)
    # Формула: (L * W * T) / 144 для дюймовой системы.
    return (length_in * width_in * thickness_in) / 144.0 * qty


def species_multiplier(species: str) -> float:
    return SPECIES_MULTIPLIERS.get(species.lower(), 1.0)


def processing_multiplier(processing: List[str]) -> float:
    mult = 1.0
    for op in processing:
        mult += PROCESSING_MULTIPLIERS.get(op.lower(), 0.0)
    return mult


def waste_multiplier(dim: Optional[Dimensions], explicit_percent: Optional[float]) -> float:
    percent = explicit_percent if explicit_percent is not None else DEFAULT_WASTE_PERCENT
    # Нестандартные размеры повышают отходы.
    if dim:
        if dim.width_mm > 200 or dim.thickness_mm > 60:
            percent += 5.0
        if dim.length_mm > 6000:
            percent += 3.0
    return 1.0 + max(0.0, percent) / 100.0


def rush_multiplier(level: Optional[str]) -> float:
    return RUSH_MULTIPLIERS.get((level or "standard").lower(), 1.0)


def bulk_discount_percent(total_board_feet: float, ladder: List[tuple[float, float]]) -> float:
    applicable = [percent for threshold, percent in ladder if total_board_feet >= threshold]
    return max(applicable) if applicable else 0.0


def _normalize_items(order_spec: Dict[str, object]) -> List[PricingItem]:
    # Поддерживаем два формата: order_spec["items"] = [...] или плоский словарь.
    items_raw = order_spec.get("items") if isinstance(order_spec, dict) else None
    if isinstance(items_raw, list) and items_raw:
        candidates = items_raw
    else:
        candidates = [order_spec]

    normalized: List[PricingItem] = []
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        dims_raw = raw.get("dimensions")
        dims = None
        if isinstance(dims_raw, dict):
            try:
                dims = Dimensions(
                    length_mm=float(dims_raw.get("length_mm") or dims_raw.get("length") or 0.0),
                    width_mm=float(dims_raw.get("width_mm") or dims_raw.get("width") or 0.0),
                    thickness_mm=float(
                        dims_raw.get("thickness_mm") or dims_raw.get("thickness") or 0.0
                    ),
                )
            except Exception:
                dims = None
        item = PricingItem(
            species=str(raw.get("species") or "pine"),
            grade=raw.get("grade"),
            dimensions=dims,
            board_feet=raw.get("board_feet"),
            quantity=float(raw.get("quantity") or raw.get("qty") or 1.0),
            price_per_board_foot=raw.get("price_per_board_foot")
            or raw.get("price_per_bf")
            or raw.get("price_per_unit"),
            processing=[str(p) for p in raw.get("processing") or []],
            waste_percent=(raw.get("waste_percent")),
            rush_level=raw.get("rush_level"),
        )
        normalized.append(item)
    return normalized


def calculate_pricing(order_spec: Dict[str, object]) -> PricingResponse:
    items = _normalize_items(order_spec)
    if not items:
        raise HTTPException(status_code=400, detail="order_spec_empty")

    ladder_raw = order_spec.get("bulk_discounts") if isinstance(order_spec, dict) else None
    ladder = BULK_DISCOUNT_LADDER
    if isinstance(ladder_raw, list):
        try:
            ladder = [
                (float(entry.get("threshold_bf")), float(entry.get("percent")))
                for entry in ladder_raw
                if isinstance(entry, dict)
                and entry.get("threshold_bf") is not None
                and entry.get("percent") is not None
            ]
        except Exception:
            ladder = BULK_DISCOUNT_LADDER

    total_bf = 0.0
    for item in items:
        if item.board_feet is None and item.dimensions:
            item.board_feet = calculate_board_feet(item.dimensions, item.quantity)
        if item.board_feet is None:
            # Фолбек: считаем 1 board foot на единицу.
            item.board_feet = 1.0 * item.quantity
        total_bf += item.board_feet

    discount_percent = bulk_discount_percent(total_bf, ladder)
    discounts_notes = []
    if discount_percent > 0:
        discounts_notes.append(f"Bulk discount {discount_percent}% for {total_bf:.1f} bf")

    breakdowns: List[PricingItemBreakdown] = []
    total_amount = 0.0
    for idx, item in enumerate(items):
        base_price_per_bf = float(item.price_per_board_foot or DEFAULT_PRICE_PER_BF)
        base_price = base_price_per_bf * item.board_feet
        sp_mult = species_multiplier(item.species)
        proc_mult = processing_multiplier(item.processing)
        waste_mult = waste_multiplier(item.dimensions, item.waste_percent)
        rush_mult = rush_multiplier(item.rush_level)
        subtotal = base_price * sp_mult * proc_mult * waste_mult * rush_mult
        discount_value = subtotal * discount_percent / 100.0
        final_price = subtotal - discount_value
        total_amount += final_price

        notes = list(item.notes)
        if item.waste_percent is None and item.dimensions:
            notes.append("waste inferred from dimensions")
        if discount_value > 0:
            notes.append(f"bulk discount applied {discount_percent}%")

        breakdowns.append(
            PricingItemBreakdown(
                item_index=idx,
                species=item.species,
                grade=item.grade,
                volume_board_feet=round(item.board_feet, 2),
                base_price=round(base_price, 2),
                species_multiplier=round(sp_mult, 2),
                processing_multiplier=round(proc_mult, 2),
                waste_multiplier=round(waste_mult, 3),
                rush_multiplier=round(rush_mult, 3),
                subtotal=round(subtotal, 2),
                discount_applied=round(discount_value, 2),
                final_price=round(final_price, 2),
                notes=notes,
            )
        )

    return PricingResponse(
        currency=CURRENCY,
        total=round(total_amount, 2),
        discounts=discounts_notes,
        items=breakdowns,
    )


app = FastAPI()


@app.post("/agents/pricing/run", response_model=PricingResponse)
async def run_pricing(request: PricingRequest) -> PricingResponse:
    """Пошаговый расчет стоимости с учетом породы, обработки, отходов и скидок."""
    try:
        response = calculate_pricing(request.order_spec)
        response.context_snippets = await _maybe_retrieve_pricing_context(request)
        return response
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - защитная ветка
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _maybe_retrieve_pricing_context(request: PricingRequest) -> List[str]:
    """
    Pulls RAG snippets when raw message is provided (for explanations).
    Kept optional to avoid extra latency when not needed.
    """
    if not request.message:
        return []
    try:
        session = RagSessionContext(
            client_id=request.context.get("client_id") if isinstance(request.context, dict) else None,
            product_id=request.context.get("product_id") if isinstance(request.context, dict) else None,
            lang=request.context.get("lang") if isinstance(request.context, dict) else None,
        )
        retrieved = await rag_retrieve(request.message, session=session)
        return [item.document.text for item in retrieved]
    except Exception:
        return []
