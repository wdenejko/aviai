"""Build the combined METAR+TAF SFT set — one adapter, both products (Phase 6).

The multi-task experiment: can a SINGLE rank-16 LoRA decode both METAR (flat JSON) and TAF
(nested JSON)? Each source SFT example is already a self-describing {"messages":[user, assistant]}
pair whose user turn carries its own prompt ("decode the METAR…" vs "decode the TAF…") and exact
output schema, so the model routes on the prompt — no task tag needed. We union the two disjoint
train splits and INTERLEAVE them (deterministic shuffle) so training sees both tasks throughout,
not all of one then all of the other (which would let it forget the first).

Inputs are the existing single-task SFT files (each built from its own disjoint train split, so the
combined set is still disjoint from BOTH frozen evals). Output feeds the same train_lora.py; use
--max-seq 2048 to fit the long TAF targets (METAR examples are short and just train with headroom).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

_METAR = Path("data/processed/sft/train_v2.jsonl")
_TAF = Path("data/processed/sft/train_taf.jsonl")
_OUT = Path("data/processed/sft/train_combined.jsonl")


def _load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _kind(ex: dict) -> str:
    """Label an example by its task, read from the prompt text — for the mix report only."""
    return "taf" if "Decode the TAF" in ex["messages"][0]["content"] else "metar"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combine the METAR + TAF SFT sets into one training file."
    )
    ap.add_argument("--metar", type=Path, default=_METAR)
    ap.add_argument("--taf", type=Path, default=_TAF)
    ap.add_argument("--out", type=Path, default=_OUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    combined = _load(args.metar) + _load(args.taf)
    random.Random(args.seed).shuffle(combined)  # deterministic interleave of the two tasks

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for ex in combined:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    mix = Counter(_kind(ex) for ex in combined)
    print(f"wrote {len(combined)} combined SFT examples -> {args.out}")
    print(f"mix: {dict(mix)}")


if __name__ == "__main__":
    main()
