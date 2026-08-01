"""Build the LoRA SFT dataset for TAF from the disjoint train split (Phase 6).

The TAF sibling of finetune/build_sft.py. Targets are the gold-validated consensus decodes, so the
model learns raw TAF -> CORRECT nested JSON (the change-group structure + the unit conversions
m/s->kt, SM->m). The split is exactly the train side of the freeze (ADR-006): stations NOT held out
for the eval AND before the unseen_time boundary — disjoint from every eval/taf/v1 record, so any
lift is generalization, not memorization.

Only CLEAN/DISSENT records become targets (their 2-voice consensus reference is trustworthy);
PARSE_FAIL is left out — a wrong target teaches the wrong thing. Output is conversational JSONL
({"messages":[user, assistant]}) with the assistant content == the exact reference the runner
scores against (target == reference == expected output).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import duckdb

from avtext.harness.freeze_taf import CORPUS, TIME_BOUNDARY, held_out_stations
from avtext.harness.hardcases import CLEAN, DISSENT
from avtext.harness.hardcases_taf import classify_taf
from avtext.harness.prompt_taf import format_prompt_taf

_OUT = Path("data/processed/sft/train_taf.jsonl")


def _target_json(ref: dict) -> str:
    """The consensus reference as the model should emit it (clouds tuples -> [cover, base])."""
    periods = []
    for p in ref.get("periods", []):
        pp = dict(p)
        c = p.get("clouds")  # None when the vote was undecided (dissent) -> keep null
        pp["clouds"] = None if c is None else [list(x) for x in c]
        periods.append(pp)
    return json.dumps({"header": ref.get("header", {}), "periods": periods})


def sample_train(corpus: Path, per_station: int, held: tuple[str, ...], boundary: str) -> list[str]:
    """Deterministic, station-stratified sample of the TRAIN split (disjoint from the eval)."""
    held_sql = ", ".join(f"'{s}'" for s in held)
    q = f"""
        SELECT raw FROM (
            SELECT raw_text AS raw, row_number() OVER (
                PARTITION BY station_id ORDER BY md5(raw_text || CAST(issue_time AS VARCHAR))
            ) AS rn
            FROM read_parquet('{corpus.as_posix()}')
            WHERE raw_text IS NOT NULL
              AND station_id NOT IN ({held_sql})                 -- not the held-out eval stations
              AND CAST(issue_time AS VARCHAR) < '{boundary}'      -- before the eval time boundary
        ) WHERE rn <= {per_station}
        ORDER BY md5(raw)   -- interleave stations so the cap doesn't starve rare formats
    """
    return [r[0] for r in duckdb.sql(q).fetchall()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the TAF LoRA SFT set from the train split.")
    ap.add_argument("--per-station", type=int, default=8)
    ap.add_argument("--max", type=int, default=8000)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args()

    held = held_out_stations(args.corpus)
    raws = sample_train(args.corpus, args.per_station, held, TIME_BOUNDARY)
    seen: set[str] = set()
    examples: list[dict] = []
    mix = Counter()
    for raw in raws:
        if raw in seen:
            continue
        seen.add(raw)
        v = classify_taf(raw)
        if v.label not in (CLEAN, DISSENT):
            continue  # trustworthy targets only
        if not v.reference.get("periods"):
            continue  # degenerate reference — nothing to learn
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": format_prompt_taf(raw)},
                    {"role": "assistant", "content": _target_json(v.reference)},
                ]
            }
        )
        mix["m/s" if "MPS" in raw else "kt"] += 1
        mix[
            f"{min(len(v.reference['periods']), 5)}+periods"
            if len(v.reference["periods"]) >= 5
            else f"{len(v.reference['periods'])}periods"
        ] += 1
        if len(examples) >= args.max:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} TAF SFT examples -> {args.out}")
    print(f"mix: {dict(mix)}")


if __name__ == "__main__":
    main()
