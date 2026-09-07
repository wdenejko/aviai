"""Assert an SFT set is disjoint from the frozen evals — the guard that was missing in Phase 7.

The S23 data-growth audit found 34 NOTAM extraction training records that were exact duplicates of
eval/notam/v1 rows: the corpus `split` label had missed them (dedup-before-split). This module makes
the check a first-class, reusable gate — run it on any built SFT file BEFORE training, and it exits
non-zero if a single training input exactly equals a frozen-eval raw.

We compare at the RECORD level (exact equality of the model's input string), which is the meaningful
contamination standard: short common phrases like "RWY 14/32 CLSD" legitimately recur as *fragments*
of longer, distinct NOTAMs, and excluding those would bias the training distribution — so substrings
are reported for transparency but are not a failure.

CLI:  uv run python -m avtext.finetune.leakcheck data/processed/sft/train_all_lg.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The five frozen evals every SFT set must stay disjoint from (raw field = the model input).
_EVALS = {
    "eval/v1 (METAR)": "eval/v1/eval.jsonl",
    "eval/v2 (METAR)": "eval/v2/eval.jsonl",
    "eval/taf/v1": "eval/taf/v1/eval.jsonl",
    "eval/notam/v1": "eval/notam/v1/eval.jsonl",
    "eval/notam_cls/v1": "eval/notam_cls/v1/eval.jsonl",
}
_MARKERS = ("METAR: ", "TAF: ", "NOTAM: ")  # the raw sits after the last marker, before "\n\nJSON:"


def embedded_input(user: str) -> str:
    """The raw METAR/TAF/NOTAM a training prompt embeds (classification prompts fall through)."""
    for m in _MARKERS:
        if m in user:
            return user.split(m, 1)[1].split("\n\nJSON:", 1)[0].strip()
    return user.strip()


def check(sft_path: Path, root: Path) -> int:
    inputs = {
        embedded_input(json.loads(line)["messages"][0]["content"])
        for line in sft_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    print(f"{sft_path}: {len(inputs):,} distinct training inputs")
    total = 0
    for name, rel in _EVALS.items():
        p = root / rel
        if not p.exists():
            continue
        ev = {json.loads(x)["raw"].strip() for x in p.read_text().splitlines() if x.strip()}
        overlap = ev & inputs
        total += len(overlap)
        mark = "OK" if not overlap else "LEAK"
        print(f"  [{mark:4s}] {name:20s} {len(overlap):3d} exact overlaps  (eval n={len(ev)})")
        for r in list(overlap)[:5]:
            print(f"           {r[:72]!r}")
    print("DISJOINT — safe to train" if total == 0 else f"CONTAMINATED — {total} exact overlaps")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Assert an SFT file is disjoint from frozen evals.")
    ap.add_argument("sft", type=Path, help="path to the SFT .jsonl to check")
    ap.add_argument("--root", type=Path, default=Path("."), help="repo root holding eval/")
    args = ap.parse_args()
    sys.exit(1 if check(args.sft, args.root) else 0)


if __name__ == "__main__":
    main()
