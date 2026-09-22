"""Build the Gate-1 behaviour-shift eval buckets (held-out vs trained).

The pilot mixture only drew a *slice* of each targeted pool (295 of 3840 targetA rows, 20 of 58
targetC trajectories), so the unused remainder is a free held-out set drawn from exactly the target
distribution — no separate generation run needed.

The split is exact rather than statistical: every generated record carries a stable `meta.id`, so we
collect the ids that actually landed in `pilot_mixture.jsonl` and partition the source slices by
membership. That is what lets the eval separate GENERALIZATION (held-out rows improve) from
MEMORIZATION (only trained rows improve) — the single question a loss-only eval can answer.

`tulu3_trained` is emitted as a replay reference, NOT a control: those rows were in training, so its
delta includes memorisation and says nothing about forgetting. A clean control needs a never-trained
general slice (see `breadth/acquire.py` with the pilot ids excluded).

Usage (from the dsbench project root):
    python -m dsbench.sftgen.eval_buckets --out eval_buckets.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

# chars/3.5 is the same token estimate the assembler uses, kept consistent on purpose.
CHARS_PER_TOKEN = 3.5
# Blocks are packed at the MMQ bundle's compiled geometry; 16 gives a stable per-bucket mean.
DEFAULT_BLOCKS = 16
SEQ = 2048
# Pack headroom so a bucket still yields DEFAULT_BLOCKS after template rendering overhead.
HEADROOM = 1.3


def _pilot_ids(pilot_path: str) -> tuple[dict[str, set[str]], list[dict]]:
    """Return {pool: {meta.id already trained on}} plus the pilot's tulu3 rows."""
    trained: dict[str, set[str]] = {"targetA": set(), "targetC": set()}
    tulu: list[dict] = []
    with open(pilot_path) as handle:
        for line in handle:
            record = json.loads(line)
            meta = record.get("meta", {})
            pool = meta.get("mix_pool")
            if pool in trained and meta.get("id"):
                trained[pool].add(meta["id"])
            if pool == "tulu3":
                tulu.append(record)
    return trained, tulu


def _split(path: str, trained_ids: set[str]) -> tuple[list[dict], list[dict]]:
    trained, heldout = [], []
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            bucket = trained if record.get("meta", {}).get("id") in trained_ids else heldout
            bucket.append(record)
    return trained, heldout


def _take(rows: list[dict], need_chars: float, rng: random.Random) -> list[dict]:
    """Sample rows until we have enough text to pack the requested number of blocks."""
    rows = list(rows)
    rng.shuffle(rows)
    out, total = [], 0
    for record in rows:
        out.append(record)
        total += len(json.dumps(record.get("messages")))
        if total >= need_chars:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gate-1 eval buckets")
    parser.add_argument("--data-zone", default="data/sft")
    parser.add_argument("--out", default="eval_buckets.jsonl")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    zone = args.data_zone.rstrip("/")
    trained_ids, tulu_rows = _pilot_ids(f"{zone}/pilot_mixture.jsonl")

    a_trained, a_heldout = _split(
        f"{zone}/targetA_dialect_conventions.train.jsonl", trained_ids["targetA"]
    )
    c_trained, c_heldout = _split(f"{zone}/targetC_ml_delivery.jsonl", trained_ids["targetC"])
    print(f"targetA: trained={len(a_trained)} heldout={len(a_heldout)}")
    print(f"targetC: trained={len(c_trained)} heldout={len(c_heldout)}")

    need = args.blocks * SEQ * CHARS_PER_TOKEN * HEADROOM
    buckets = {
        "targetA_heldout": _take(a_heldout, need, rng),
        "targetA_trained": _take(a_trained, need, rng),
        "targetC_heldout": _take(c_heldout, need, rng),
        "targetC_trained": _take(c_trained, need, rng),
        "tulu3_trained": _take(tulu_rows, need, rng),
    }
    with open(args.out, "w") as handle:
        for name, rows in buckets.items():
            for record in rows:
                handle.write(
                    json.dumps(
                        {
                            "bucket": name,
                            "messages": record["messages"],
                            "tools": record.get("tools"),
                        }
                    )
                    + "\n"
                )
            chars = sum(len(json.dumps(r["messages"])) for r in rows)
            print(f"  {name}: {len(rows)} rows ~{chars / CHARS_PER_TOKEN / 1000:.0f}k tok")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
