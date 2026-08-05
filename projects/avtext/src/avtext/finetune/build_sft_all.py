"""Build the all-products SFT set — one adapter for METAR + TAF + NOTAM (Phase 7).

The full multi-task bet: extend the proven METAR+TAF combined recipe to a THIRD product family
(NOTAM extraction + classification). Every source example is already a self-describing
{"messages":[user, assistant]} pair whose user turn names its own task ("Decode the METAR…",
"Decode the TAF…", "…NOTAM decoder…", "…NOTAM classifier…") and output schema, so the model routes
on the prompt — no task tag needed. We union the disjoint train splits and INTERLEAVE them
(deterministic shuffle) so training sees all tasks throughout rather than forgetting early ones.

Inputs are the existing single-product SFT files (each from its own disjoint train split), so the
union stays disjoint from ALL frozen evals (eval/v2, eval/taf/v1, eval/notam/v1, eval/notam_cls/v1).
Feeds the same train_lora.py; use --max-seq 2048 (TAF targets are the long pole).

CLI:  uv run python -m avtext.finetune.build_sft_all
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

_METAR = Path("data/processed/sft/train_v2.jsonl")
_TAF = Path("data/processed/sft/train_taf.jsonl")
_NOTAM = Path("data/processed/sft/train_notam.jsonl")
_OUT = Path("data/processed/sft/train_all.jsonl")


def _load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _kind(ex: dict) -> str:
    """Label an example by its task (from the prompt text) — for the mix report only."""
    u = ex["messages"][0]["content"]
    if "Decode the TAF" in u:
        return "taf"
    if "Decode the METAR" in u:
        return "metar"
    if "NOTAM classifier" in u:
        return "notam_cls"
    return "notam_ext"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combine METAR + TAF + NOTAM SFT sets into one training file."
    )
    ap.add_argument("--metar", type=Path, default=_METAR)
    ap.add_argument("--taf", type=Path, default=_TAF)
    ap.add_argument("--notam", type=Path, default=_NOTAM)
    ap.add_argument("--out", type=Path, default=_OUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    combined = _load(args.metar) + _load(args.taf) + _load(args.notam)
    random.Random(args.seed).shuffle(combined)  # deterministic interleave across all tasks

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for ex in combined:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    mix = Counter(_kind(ex) for ex in combined)
    print(f"wrote {len(combined)} all-products SFT examples -> {args.out}")
    print(f"mix: {dict(sorted(mix.items()))}")


if __name__ == "__main__":
    main()
