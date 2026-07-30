"""Build the LoRA SFT dataset from the disjoint train split (Phase 4).

Targets are the gold-validated consensus decodes, so the model learns raw -> CORRECT
canonical JSON — crucially, the unit conversions (inHg A-reports -> hPa, m/s -> knots) that
the size sweep showed *no* base model can do (§ size_sweep_v1). The split is exactly the
train side of ADR-006: training stations (not the eval's held-out four) AND before the
2026-05-01 time boundary — disjoint from every eval/v1 record, so a lift is generalization.

Only clean/dissent records become targets (their consensus reference is trustworthy);
parse_fail decodes are left out — a wrong target teaches the wrong thing. Output is a
conversational JSONL ({"messages":[user, assistant]}) that Unsloth templates with the
model's own chat format.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import duckdb

from avtext.consensus.vote import FIELDS
from avtext.harness.freeze import HELD_OUT_STATIONS, TIME_BOUNDARY, V2_CORPUS, V2_HELD_OUT
from avtext.harness.hardcases import CLEAN, DISSENT, classify
from avtext.harness.prompt import format_prompt

_CORPUS = Path("data/processed/metar/iem_metar.parquet")
_OUT = Path("data/processed/sft/train.jsonl")


def _target_json(ref: dict) -> str:
    """The reference decode in the prompt's exact schema (clouds as [cover, base] lists)."""
    out = {}
    for f in FIELDS:
        v = ref.get(f)
        out[f] = [list(c) for c in v] if f == "clouds" else v
    return json.dumps(out)


def sample_train(corpus: Path, per_station: int, held_out: tuple = HELD_OUT_STATIONS) -> list[str]:
    """Deterministic, station-stratified sample of the TRAIN split (disjoint from the eval)."""
    held = ", ".join(f"'{s}'" for s in held_out)
    q = f"""
        SELECT raw FROM (
            SELECT raw, row_number() OVER (
                PARTITION BY station ORDER BY md5(raw || CAST(valid_utc AS VARCHAR))
            ) AS rn
            FROM read_parquet('{corpus.as_posix()}')
            WHERE raw IS NOT NULL
              AND station NOT IN ({held})                        -- not the held-out eval stations
              AND CAST(valid_utc AS VARCHAR) < '{TIME_BOUNDARY}'  -- before the eval time boundary
        ) WHERE rn <= {per_station}
        ORDER BY md5(raw)   -- interleave stations so the cap doesn't starve rare formats (MPS)
    """
    return [r[0] for r in duckdb.sql(q).fetchall()]


def _is_inhg(raw: str) -> bool:
    return any(t[0] == "A" and t[1:].isdigit() for t in raw.split())


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the LoRA SFT set from the train split.")
    ap.add_argument("--per-station", type=int, default=600)
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--corpus", type=Path, default=_CORPUS)
    ap.add_argument("--out", type=Path, default=_OUT)
    ap.add_argument("--v2", action="store_true", help="v2 corpus + 78-station holdout, 8k examples")
    args = ap.parse_args()

    held = HELD_OUT_STATIONS
    if args.v2:  # 490-station geo-balanced corpus; disjoint from eval/v2's held-out + time split
        args.corpus, held, args.out, args.max = (
            V2_CORPUS, V2_HELD_OUT, Path("data/processed/sft/train_v2.jsonl"), 8000)
    raws = sample_train(args.corpus, args.per_station, held)
    seen: set[str] = set()
    examples: list[dict] = []
    mix = Counter()
    for raw in raws:
        if raw in seen:
            continue
        seen.add(raw)
        v = classify(raw)
        if v.label not in (CLEAN, DISSENT):
            continue  # trustworthy targets only
        if all(v.reference.get(f) is None for f in FIELDS if f != "clouds"):
            continue  # degenerate reference — nothing to learn
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": format_prompt(raw)},
                    {"role": "assistant", "content": _target_json(v.reference)},
                ]
            }
        )
        mix["inHg" if _is_inhg(raw) else "hPa"] += 1
        mix["m/s" if "MPS" in raw else "kt"] += 1
        if len(examples) >= args.max:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} SFT examples -> {args.out}")
    print(f"conversion coverage: {dict(mix)}")


if __name__ == "__main__":
    main()
