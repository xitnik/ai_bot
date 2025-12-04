from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from alternatives_models import AlternativeType


class AlternativesDeepResearchRequest(BaseModel):
    """Запрос на deep research с дополнительным контекстом."""

    query_text: str
    base_item_name: Optional[str] = None
    use_case: Optional[str] = None
    domain: Literal["material", "equipment", "software", "service", "other"] = "other"
    hard_filters: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)
    region: Optional[str] = None
    delivery_deadline: Optional[str] = None
    certifications: List[str] = Field(default_factory=list)
    budget_limit: Optional[str] = None
    price_band: Optional[Literal["cheaper", "premium"]] = None
    k: int = 5
    notes: Optional[str] = None
    trace_id: Optional[str] = None
    session_id: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class DeepResearchTaskUnderstanding(BaseModel):
    base_item_name: str
    domain: Literal["material", "equipment", "software", "service", "other"]
    key_requirements: List[str]
    hard_constraints: List[str]
    nice_to_have: List[str]
    notes: Optional[str] = None


class DeepResearchSource(BaseModel):
    url: str
    source_type: Literal["vendor", "review", "forum", "article", "marketplace", "other"]
    title: Optional[str] = None
    snippet: str
    sentiment: Literal["positive", "mixed", "negative", "neutral"] = "neutral"


class DeepResearchAlternative(BaseModel):
    name: str
    type: AlternativeType
    short_description: str
    when_to_choose: str
    pros: List[str]
    cons: List[str]
    price_info_currency: Optional[str] = None
    price_info_range: Optional[str] = None
    price_relative_to_base: Literal["unknown", "cheaper", "similar", "more_expensive"] = "unknown"
    fit_score_0_to_100: int = Field(ge=0, le=100)
    risk_notes: List[str] = Field(default_factory=list)
    sources: List[DeepResearchSource] = Field(default_factory=list)


class DeepResearchComparisonSummary(BaseModel):
    key_dimensions: List[str]
    best_for_strict_constraints: str
    best_overall: str
    tradeoffs: List[str]


class AlternativesDeepResearchResult(BaseModel):
    task_understanding: DeepResearchTaskUnderstanding
    research_plan: Dict[str, Any]
    alternatives: List[DeepResearchAlternative]
    comparison_summary: DeepResearchComparisonSummary
    final_recommendations: Dict[str, Any]


class AlternativesDeepResearchPayload(BaseModel):
    """Пэйлоад, который кладем в поле data ответа агента."""

    summary: str
    research: AlternativesDeepResearchResult


class AlternativesDeepResearchResponse(BaseModel):
    """Конверт для совместимости с orchestrator.call_agent."""

    status: Literal["ok", "error"] = "ok"
    data: AlternativesDeepResearchPayload
    error: Optional[Dict[str, Any]] = None

