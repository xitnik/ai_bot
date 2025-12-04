from __future__ import annotations

from agents.base import AgentInput
from agents.supervisor import Supervisor


def test_supervisor_routes_intents_and_variant_deterministic():
    sup = Supervisor()
    payload = AgentInput(message="Need price", user_id="u1", session_id="s1", intent="", context={})
    decision = sup.route(payload)
    assert decision.intent == "pricing"
    assert "pricing" in decision.agents
    # variant stable for same user
    assert sup.select_variant("u1") == sup.select_variant("u1")


def test_supervisor_adds_deep_research_for_alternatives():
    sup = Supervisor(variants=("control",))
    payload = AgentInput(message="Need alternative compare vs", user_id="u2", session_id="s2", intent="", context={})
    decision = sup.route(payload)
    assert decision.intent == "alternatives"
    assert "alternatives" in decision.agents
    assert "alternatives_deep_research" in decision.agents
