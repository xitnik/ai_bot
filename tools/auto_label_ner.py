from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ner import NerMode, extract_entities_async


async def auto_label(input_path: Path, output_path: Path, mode: NerMode) -> None:
    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text") or record.get("message") or ""
            if not text:
                continue
            ner_result = await extract_entities_async(text, mode=mode)
            payload = {
                "text": text,
                "suggested_entities": [e.to_dict() for e in ner_result.entities],
                "approved": False,
                "needs_review": True,
            }
            outfile.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Semi-automatic NER labeling pipeline.")
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to JSONL with raw messages."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("labeled_ner.jsonl"),
        help="Where to store suggestions.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in NerMode],
        default=NerMode.hybrid.value,
    )
    args = parser.parse_args()
    asyncio.get_event_loop().run_until_complete(
        auto_label(args.input, args.output, NerMode(args.mode))
    )
    print(f"Wrote suggestions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
