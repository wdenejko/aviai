"""Build the NOTAM SFT set — extraction + classification, from the train splits (Phase 7).

Two NOTAM tasks, one interleaved SFT file (each example self-describes via its prompt, so one
adapter handles both — the same multi-task recipe that worked for METAR+TAF):
  • extraction   → user = category-aware extract prompt, assistant = {"rows": [...]}  (OpenNOTAM)
  • classification → user = classify prompt, assistant = <class name>                  (DEEL-AI)
Only the train splits are used, so this is disjoint from the frozen eval/notam/* test splits.

CLI:  uv run python -m avtext.finetune.build_sft_notam
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from avtext.harness.prompt_notam import format_prompt_notam, format_prompt_notam_cls
from avtext.ingest.awc import DATA_DIR

EXTRACT_CORPUS = DATA_DIR / "processed" / "notam" / "notam_clean.jsonl"
CLASS_CORPUS = DATA_DIR / "processed" / "notam" / "notam_class.jsonl"
_OUT = DATA_DIR / "processed" / "sft" / "train_notam.jsonl"

# Frozen NOTAM evals to hold out. Belt-and-braces: the corpus `split` field SHOULD keep train
# disjoint from these, but it silently missed 34 extraction records that are exact duplicates of
# eval/notam/v1 rows (S23 leak audit — dedup-before-split). We now exclude any train record whose
# raw text exactly matches ANY frozen-eval raw, so disjointness is guaranteed against the actual
# eval files rather than trusting a label. (Short common phrases like "RWY 14/32 CLSD" still recur
# as *fragments* of longer, distinct NOTAMs — that's real-world distribution, not an eval-record
# leak, and is intentionally kept.)
_EVAL_FILES = (
    DATA_DIR.parent / "eval" / "notam" / "v1" / "eval.jsonl",
    DATA_DIR.parent / "eval" / "notam_cls" / "v1" / "eval.jsonl",
)


def _read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _eval_raws() -> set[str]:
    """Every raw string in the frozen NOTAM evals — the exact-match holdout blocklist."""
    blocked: set[str] = set()
    for f in _EVAL_FILES:
        if f.exists():
            blocked |= {
                json.loads(x)["raw"].strip() for x in f.read_text().splitlines() if x.strip()
            }
    return blocked


def _extraction_examples(blocked: set[str]) -> tuple[list[dict], int]:
    out, dropped = [], 0
    for r in _read(EXTRACT_CORPUS):
        if r.get("split") != "train":
            continue
        if r["raw_text"].strip() in blocked:  # exact dup of a frozen-eval row -> skip
            dropped += 1
            continue
        target = json.dumps({"rows": r["rows"]}, ensure_ascii=False)
        out.append(
            {
                "messages": [
                    {"role": "user", "content": format_prompt_notam(r["raw_text"], r["category"])},
                    {"role": "assistant", "content": target},
                ]
            }
        )
    return out, dropped


def _classification_examples(blocked: set[str]) -> tuple[list[dict], int]:
    out, dropped = [], 0
    for r in _read(CLASS_CORPUS):
        if r.get("split") != "train":
            continue
        if r["raw_text"].strip() in blocked:
            dropped += 1
            continue
        out.append(
            {
                "messages": [
                    {"role": "user", "content": format_prompt_notam_cls(r["raw_text"])},
                    {"role": "assistant", "content": r["label"]},
                ]
            }
        )
    return out, dropped


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the NOTAM SFT set (extraction + classification)."
    )
    ap.add_argument("--out", type=Path, default=_OUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    blocked = _eval_raws()
    extraction, ext_dropped = _extraction_examples(blocked)
    classification, cls_dropped = _classification_examples(blocked)
    examples = extraction + classification
    random.Random(args.seed).shuffle(examples)  # interleave the two tasks

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    mix = Counter({"extraction": len(extraction), "classification": len(classification)})
    print(f"wrote {len(examples)} NOTAM SFT examples -> {args.out}")
    print(f"mix: {dict(mix)}")
    print(f"eval-holdout drops (exact dup of a frozen-eval raw): "
          f"extraction={ext_dropped}, classification={cls_dropped}")


if __name__ == "__main__":
    main()
