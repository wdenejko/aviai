"""Render execution-verified rows into Qwen3.5/3.6 chat-template training JSONL (ADR-001 B.4).

Separate from generation on purpose: the raw SFTRow JSONL is the auditable source of truth (it feeds
`decontaminate.py` and the licence audit); this step turns it into what the trainer eats. Choices
follow ADR-001 B.4:
  * ChatML messages (the trainer applies the Qwen template);
  * assistant-only loss -- marked with `loss_mask_roles` so the collator masks system/user tokens;
  * thinking on/off -- a reasoning row bakes a real <think>...</think> block into the assistant
    turn; a direct row emits an EMPTY <think></think> (the Qwen non-thinking convention), so the
    model keeps its thinking channel but learns these facts can be answered directly.
"""
from __future__ import annotations

import argparse
import json

from dsbench.sftgen.schema import SFTRow, row_from_dict


def _assistant_text(row: SFTRow) -> str:
    a = next(t for t in row.turns if t.role == "assistant")
    think = f"<think>\n{a.thinking}\n</think>" if a.thinking else "<think>\n\n</think>"
    return f"{think}\n\n{a.content}"


def to_chat_record(row: SFTRow) -> dict:
    """One training record: ChatML messages + assistant-only loss marker + audit meta."""
    messages = []
    for t in row.turns:
        content = _assistant_text(row) if t.role == "assistant" else t.content
        messages.append({"role": t.role, "content": content})
    return {
        "messages": messages,
        "loss_mask_roles": ["assistant"],
        "meta": {
            "id": row.id, "target": row.target, "family": row.family, "dialect": row.dialect,
            "tags": list(row.tags),
            "provenance": {
                "generator": row.provenance.generator, "method": row.provenance.method,
                "teacher": row.provenance.teacher, "licence": row.provenance.licence,
                "seed": row.provenance.seed,
            },
            "verification": {
                "engine": row.verification.engine, "truth": row.verification.truth,
                "agrees": row.verification.agrees,
            },
        },
    }


def render_file(raw_path: str, out_path: str) -> int:
    n = 0
    with open(raw_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = row_from_dict(json.loads(line))
            fout.write(json.dumps(to_chat_record(row)) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Render raw sftgen rows to Qwen chat-template JSONL")
    ap.add_argument("raw", help="raw SFTRow JSONL (from a generator)")
    ap.add_argument("out", help="training JSONL to write")
    args = ap.parse_args()
    n = render_file(args.raw, args.out)
    print(f"rendered {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
