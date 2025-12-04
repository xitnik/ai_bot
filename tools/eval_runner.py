from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from models import Message, SessionDTO
from orchestrator import enrich_message, route_message


@dataclass
class EvalResult:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


async def eval_intents(fixtures_path: Path) -> Dict[str, EvalResult]:
    samples = json.loads(fixtures_path.read_text())
    overall = EvalResult(total=0, correct=0)
    per_intent: Dict[str, EvalResult] = {}
    for sample in samples:
        session = SessionDTO(
            session_id="eval-session",
            user_id=sample.get("user_id") or "eval-user",
            channel="webchat",
            state="idle",
            started_at=sample.get("started_at") or __import__("datetime").datetime.utcnow(),  # type: ignore[arg-type]
            last_event_at=sample.get("last_event_at") or __import__("datetime").datetime.utcnow(),  # type: ignore[arg-type]
        )
        msg = Message(user_id=session.user_id, channel="webchat", text=sample["text"], attachments=None)
        context = await enrich_message(msg)
        decision = await route_message(session, msg, context)
        overall.total += 1
        expected = sample["intent"]
        if decision["intent"] == expected:
            overall.correct += 1
        per_result = per_intent.setdefault(expected, EvalResult(total=0, correct=0))
        per_result.total += 1
        if decision["intent"] == expected:
            per_result.correct += 1
    return {"overall": overall, **per_intent}


def main() -> int:
    fixtures_dir = Path("fixtures")
    intents_file = fixtures_dir / "golden_intents.json"
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(eval_intents(intents_file))
    overall = results.pop("overall")

    print(f"Overall accuracy: {overall.accuracy:.2f} ({overall.correct}/{overall.total})")
    min_overall = 0.8
    min_agent = 0.95
    status = 0 if overall.accuracy >= min_overall else 1

    for intent, res in results.items():
        acc = res.accuracy
        print(f"Intent {intent}: {acc:.2f} ({res.correct}/{res.total})")
        if acc < min_agent:
            status = 1
    if status != 0:
        print("Eval thresholds not met", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
