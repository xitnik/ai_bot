from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from rag.pipeline import SessionContext, rag_retrieve


async def label_rag(input_path: Path, output_path: Path, top_k: int = 3) -> None:
    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            record: Dict[str, Any] = json.loads(line)
            query = record.get("query") or record.get("text")
            if not query:
                continue
            session = SessionContext(client_id=record.get("client_id"), product_id=record.get("product_id"))
            docs = await rag_retrieve(query, session=session)
            suggestion = {
                "query": query,
                "suggested_docs": [
                    {"id": d.document.id, "score": d.score, "source": d.document.metadata.get("source")}
                    for d in docs[:top_k]
                ],
                "approved": False,
                "needs_review": True,
            }
            outfile.write(json.dumps(suggestion, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-label RAG relevant documents.")
    parser.add_argument("--input", type=Path, required=True, help="Path to JSONL with queries.")
    parser.add_argument("--output", type=Path, default=Path("labeled_rag.jsonl"), help="Output JSONL.")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    asyncio.get_event_loop().run_until_complete(label_rag(args.input, args.output, top_k=args.top_k))
    print(f"Wrote RAG suggestions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
