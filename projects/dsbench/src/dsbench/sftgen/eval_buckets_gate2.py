"""Build the Gate-2 eval buckets: held-out vs trained, plus the clean control Gate 1 lacked.

Three states get compared on these buckets -- base, the Gate-1 adapter and the Gate-2 adapter -- so
"held-out" has to mean never trained on by ANY of them. A row is held out only if it appears in
neither the Gate-1 pilot mixture nor the Gate-2 mixture. The Gate-2 pools were re-streamed from the
same sources as the pilot's, so they contain the pilot's rows; excluding only the Gate-2 mixture
would quietly hand Gate 1 an in-sample bucket and make it look better than it is.

Membership is decided by a hash of the canonical `messages` JSON. Targeted rows carry `meta.id` and
breadth rows `seq_id`, but only the content hash works across every pool. `pilot_recall()` checks
that the hash finds EVERY pilot row of each pool in the re-acquired Gate-2 file. If a normaliser
change had altered serialisation between the two acquisitions, trained rows would hash differently,
escape the exclusion and land in "held-out" -- recall below 100% is exactly that failure.

(A first-user-turn hash is NOT a usable cross-check: targetA reuses paraphrased questions, Target C
shares one prompt per family, and tulu3/opencoder repeat instructions, so it collides across
distinct rows and disagrees with the full hash even when membership is perfect.)

The new bucket that matters most is `general_heldout`: tulu3 replay rows that were acquired but
never trained on by either adapter. Gate 1 could only report a TRAINED tulu3 reference, whose
delta mixes memorisation into any forgetting signal. This one is a real forgetting control.

`targetC_heldout` is held out at the DATASET level, not the task level: the seven Target-C families
share their prompts across reps, so those rows are new data for seen task types. It measures
within-family generalisation, not transfer to new tasks.

Usage (from the dsbench project root):
    python -m dsbench.sftgen.eval_buckets_gate2 --out eval_buckets_gate2.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random

from dsbench.sftgen.eval_buckets import DEFAULT_BLOCKS, HEADROOM, SEQ, _take

CHARS_PER_TOKEN = 3.5

# bucket name -> (source file in the data zone, which split to draw)
HELDOUT = {
    "targetA_heldout": "targetA_dialect_conventions.train.jsonl",
    "targetC_heldout": "targetC_merged.jsonl",
    "general_heldout": "breadth_tulu3.jsonl",
    "sql_heldout": "breadth_gretel_sql.jsonl",
    "code_heldout": "breadth_opencoder_edu.jsonl",
    "dsnb_heldout": "breadth_jupyter_agent.jsonl",
    "swe_heldout": "breadth_swe_swiss.jsonl",
}
# Trained references: rows Gate 2 DID train on. A gap between these and the matching held-out bucket
# is the memorisation signal.
TRAINED = {
    "targetA_trained": "targetA_dialect_conventions.train.jsonl",
    "targetC_trained": "targetC_merged.jsonl",
}


def content_key(record: dict) -> str:
    """Stable identity for a record, whatever pool it came from."""
    blob = json.dumps(record.get("messages"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()


def read_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pilot_recall(pool: list[dict], pilot_rows: list[dict]) -> tuple[int, int]:
    """(distinct pilot rows found in the pool file, distinct pilot rows of this pool)."""
    in_pool = {content_key(r) for r in pool}
    wanted = {content_key(r) for r in pilot_rows}
    return len(wanted & in_pool), len(wanted)


def build(zone: str, blocks: int, seed: int) -> tuple[list[dict], dict]:
    pilot = read_jsonl(os.path.join(zone, "pilot_mixture.jsonl"))
    gate2 = read_jsonl(os.path.join(zone, "gate2_mixture.jsonl"))
    in_pilot = {content_key(r) for r in pilot}
    in_gate2 = {content_key(r) for r in gate2}
    need_chars = blocks * SEQ * CHARS_PER_TOKEN * HEADROOM
    rng = random.Random(seed)

    out: list[dict] = []
    report: dict = {"blocks_per_bucket": blocks, "seed": seed, "buckets": {},
                    "pilot_recall": {}}
    cache: dict[str, list[dict]] = {}
    for bucket, fname in {**HELDOUT, **TRAINED}.items():
        rows = cache.setdefault(fname, read_jsonl(os.path.join(zone, fname)))
        if bucket.endswith("_heldout"):
            eligible = [r for r in rows if content_key(r) not in in_pilot | in_gate2]
        else:
            eligible = [r for r in rows if content_key(r) in in_gate2]
        picked = _take(eligible, need_chars, rng)
        for record in picked:
            out.append({**record, "bucket": bucket})
        report["buckets"][bucket] = {"source": fname, "eligible": len(eligible),
                                     "picked": len(picked),
                                     "est_tokens": int(sum(len(json.dumps(r.get("messages")))
                                                           for r in picked) / CHARS_PER_TOKEN)}
    # pilot mix_pool name for each source file, so recall is measured pool by pool
    pool_of = {"targetA_dialect_conventions.train.jsonl": "targetA",
               "targetC_merged.jsonl": "targetC", "breadth_tulu3.jsonl": "tulu3",
               "breadth_gretel_sql.jsonl": "gretel_sql",
               "breadth_opencoder_edu.jsonl": "opencoder_edu",
               "breadth_jupyter_agent.jsonl": "jupyter_agent",
               "breadth_swe_swiss.jsonl": "swe_swiss"}
    by_pool = collections.defaultdict(list)
    for record in pilot:
        by_pool[record.get("meta", {}).get("mix_pool")].append(record)
    for fname in sorted(set(HELDOUT.values())):
        found, total = pilot_recall(cache[fname], by_pool[pool_of[fname]])
        report["pilot_recall"][fname] = {"found": found, "pilot_rows": total}
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gate-2 eval buckets")
    parser.add_argument("--data-zone", default="data/sft")
    parser.add_argument("--out", default="eval_buckets_gate2.jsonl")
    parser.add_argument("--report", default="")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--seed", type=int, default=4243)
    args = parser.parse_args()

    records, report = build(args.data_zone, args.blocks, args.seed)
    with open(args.out, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    if args.report:
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)
    counts = collections.Counter(r["bucket"] for r in records)
    for bucket, info in report["buckets"].items():
        print(f"{bucket:18s} eligible {info['eligible']:6,}  picked {counts[bucket]:5,}  "
              f"~{info['est_tokens']:8,} tok")
    print("pilot recall -- every row Gate 1 trained on must be found, or it leaks into held-out:")
    for fname, rec in report["pilot_recall"].items():
        found, total = rec["found"], rec["pilot_rows"]
        print(f"   {fname:40s} {found:5,} / {total:5,}  {'ok' if found == total else 'LEAK'}")


if __name__ == "__main__":
    main()
