from __future__ import annotations

from typing import Iterable, Optional

from .db import Offer


def _normalize_numeric(values: list[Optional[float]], higher_is_better: bool) -> list[float]:
    """
    Нормирует числа в диапазон [0,1]; отсутствующие значения получают 0.5.
    Более низкая цена/срок = лучше => инвертируем шкалу.
    """

    present = [v for v in values if v is not None]
    if not present:
        return [0.5 for _ in values]
    min_v = min(present)
    max_v = max(present)
    spread = max(max_v - min_v, 1e-9)

    scores: list[float] = []
    for value in values:
        if value is None:
            scores.append(0.5)
            continue
        normalized = (value - min_v) / spread
        if higher_is_better:
            scores.append(max(0.0, min(1.0, normalized)))
        else:
            scores.append(max(0.0, min(1.0, 1.0 - normalized)))
    return scores


def score_offers(offers: Iterable[Offer], weights: dict[str, float]) -> list[tuple[Offer, float]]:
    """
    Считает итоговый скор для каждого оффера.
    Весовая формула: цена и срок чем ниже, тем лучше; vendor_score чем выше, тем лучше.
    """

    offers_list = list(offers)
    prices = [o.price_per_unit for o in offers_list]
    lead_times = [o.lead_time_days for o in offers_list]
    vendor_scores = [o.vendor_score for o in offers_list]

    price_scores = _normalize_numeric(prices, higher_is_better=False)
    lead_scores = _normalize_numeric(lead_times, higher_is_better=False)
    vendor_scores_norm = _normalize_numeric(vendor_scores, higher_is_better=True)

    w_price = weights.get("price_per_unit", 0.5)
    w_lead = weights.get("lead_time_days", 0.3)
    w_vendor = weights.get("vendor_score", 0.2)

    scored: list[tuple[Offer, float]] = []
    for offer, p_score, l_score, v_score in zip(
        offers_list, price_scores, lead_scores, vendor_scores_norm
    ):
        # Итоговый балл — взвешенная сумма нормированных показателей.
        total = w_price * p_score + w_lead * l_score + w_vendor * v_score
        scored.append((offer, float(total)))
    return scored


def select_best_offer(offers: list[Offer], weights: dict[str, float]) -> Offer:
    """
    Возвращает оффер с максимальным взвешенным скором.
    """

    scored = score_offers(offers, weights)
    # При равенстве баллов возвращаем самый ранний в списке, чтобы детерминировать поведение.
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[0][0]


__all__ = ["select_best_offer", "score_offers"]
