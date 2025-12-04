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
