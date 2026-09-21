"""Assemble the Gate-1 pilot SFT mixture (ADR-001 Gate 1 + ADR-004).

Combines our execution-verified targeted slices (A/C, `data/sft/`) with the acquired public
breadth/replay buckets (`breadth/`) into one training mixture, under the mixture discipline the ADRs
require:

  * **Proportions** follow ADR-004's reweighting (targeted ~14%, breadth ~61%, replay ~25%). Each
    pool is sampled DOWN to a token budget; a pool short of its budget contributes all it has and
    the shortfall is reported (never silently padded).
  * **Decontamination vs dsbench** runs again here over EVERY kept record (belt-and-braces over the
    whole mixture, not just per-source at acquisition) -- a post-fine-tune dsbench gain must be
    learning, not leakage.
  * **Licensing** is checked: every kept row must be redistributable (clean teacher). Study-only
    rows are counted and listed, so the mixture's shippability is explicit.

Records stay in the common `{messages, tools?, loss_mask_roles, meta}` shape; the Qwen chat template
is applied by the trainer at train time (ADR-001 template rules), so assembly does no rendering. The
output is the shuffled mixture JSONL + a manifest that is the reproducibility/audit record.

    uv run python -m dsbench.sftgen.assemble --breadth-dir <scratch>/sftdata_breadth \
        --out <scratch>/pilot_mixture.jsonl --report <scratch>/pilot_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import random

from dsbench.sftgen.breadth.acquire import approx_tokens
from dsbench.sftgen.decontaminate import _row_text, build_denylist, scan_text

# (pool, mix_bucket, kind, filename, target_tokens) -- targets are ADR-004 pilot proportions. The
# not-yet-built DE + exec-feedback buckets are left out (documented as a gap, not padded).
PILOT_POOLS: list[tuple[str, str, str, str, int]] = [
    ("targetA", "targeted:sql", "targeted", "targetA_dialect_conventions.train.jsonl", 50_000),
    ("targetC", "targeted:ml", "targeted", "targetC_ml_delivery.jsonl", 90_000),
    ("jupyter_agent", "ds_notebooks", "breadth", "breadth_jupyter_agent.jsonl", 150_000),
    ("datamind", "ds_notebooks", "breadth", "breadth_datamind.jsonl", 40_000),
    ("gretel_sql", "text_to_sql", "breadth", "breadth_gretel_sql.jsonl", 115_000),
    ("opencoder_edu", "general_code", "breadth", "breadth_opencoder_edu.jsonl", 65_000),
    ("swe_swiss", "swe", "breadth", "breadth_swe_swiss.jsonl", 108_000),
    ("tulu3", "replay", "breadth", "breadth_tulu3.jsonl", 250_000),
]


def _load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _prov(rec: dict) -> tuple[str, str, bool]:
    """(licence, teacher, redistributable) from either meta shape (breadth flat vs A/C prov)."""
    meta = rec.get("meta", {})
    if "licence" in meta:  # breadth/replay record
        return (meta.get("licence", "?"), meta.get("teacher", "?"),
                bool(meta.get("redistributable", False)))
    prov = meta.get("provenance", {})  # our targeted slice: teacher-free/exec-verified => clean
    return prov.get("licence", "Apache-2.0"), prov.get("teacher") or "own-generated", True


def _select(recs: list[dict], target: int, deny, rng: random.Random, pool: str,
            bucket: str) -> tuple[list[dict], dict]:
    """Shuffle, then greedily keep decontaminated records until the token budget is reached."""
    order = recs[:]
    rng.shuffle(order)
    kept: list[dict] = []
    tokens = 0
    stats = {"pool": pool, "bucket": bucket, "target_tokens": target, "available_rows": len(recs),
             "rows": 0, "tokens": 0, "dropped_contam": 0, "study_only_rows": 0,
             "licence": None, "teacher": None}
    for rec in order:
        if tokens >= target:
            break
        if scan_text(_row_text(rec), deny) is not None:
            stats["dropped_contam"] += 1
            continue
        lic, teach, redist = _prov(rec)
        stats["licence"], stats["teacher"] = lic, teach
        if not redist:
            stats["study_only_rows"] += 1
        rec.setdefault("meta", {})["mix_pool"] = pool
        rec["meta"]["mix_bucket"] = bucket
        n = approx_tokens(rec)
        kept.append(rec)
        tokens += n
    stats["rows"] = len(kept)
    stats["tokens"] = tokens
    return kept, stats


def assemble(*, data_zone: str, breadth_dir: str, scale: float,
             seed: int) -> tuple[list[dict], dict]:
    deny = build_denylist()
    rng = random.Random(seed)
    all_kept: list[dict] = []
    pools: list[dict] = []
    for pool, bucket, kind, fname, target in PILOT_POOLS:
        path = os.path.join(data_zone if kind == "targeted" else breadth_dir, fname)
        recs = _load(path)
        kept, stats = _select(recs, int(target * scale), deny, rng, pool, bucket)
        stats["kind"] = kind
        stats["shortfall"] = max(0, int(target * scale) - stats["tokens"])
        pools.append(stats)
        all_kept.extend(kept)
    rng.shuffle(all_kept)

    total_tokens = sum(p["tokens"] for p in pools)
    by_bucket: dict[str, int] = {}
    for p in pools:
        by_bucket[p["bucket"]] = by_bucket.get(p["bucket"], 0) + p["tokens"]
    manifest = {
        "seed": seed, "scale": scale,
        "total_rows": len(all_kept), "total_tokens": total_tokens,
        "study_only_rows": sum(p["study_only_rows"] for p in pools),
        "dropped_contam": sum(p["dropped_contam"] for p in pools),
        "by_bucket_tokens": by_bucket,
        "by_bucket_pct": ({b: round(100 * t / total_tokens, 1) for b, t in by_bucket.items()}
                          if total_tokens else {}),
        "pools": pools,
        "note": ("token counts are chars/3.5 estimates (consistent across pools). DE + "
                 "exec-feedback breadth buckets are own-generation tasks, not yet built."),
    }
    return all_kept, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble the Gate-1 pilot SFT mixture (ADR-004)")
    ap.add_argument("--data-zone", default="data/sft", help="dir with the targeted A/C slices")
    ap.add_argument("--breadth-dir", required=True, help="dir with breadth_<key>.jsonl")
    ap.add_argument("--out", default="", help="write the shuffled mixture JSONL here")
    ap.add_argument("--report", default="", help="write the manifest JSON here")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every pool's token target")
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    mixture, manifest = assemble(data_zone=args.data_zone, breadth_dir=args.breadth_dir,
                                 scale=args.scale, seed=args.seed)
    if args.out:
        with open(args.out, "w") as fh:
            for rec in mixture:
                fh.write(json.dumps(rec) + "\n")
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(manifest, fh, indent=2)

    print(f"pilot mixture: {manifest['total_rows']} rows / ~{manifest['total_tokens']:,} tokens "
          f"(seed {manifest['seed']}, scale {manifest['scale']})")
    print(f"decontam dropped: {manifest['dropped_contam']}  study-only rows: "
          f"{manifest['study_only_rows']}")
    for b, pct in manifest["by_bucket_pct"].items():
        print(f"  {b:26s} {manifest['by_bucket_tokens'][b]:>9,} tok  {pct:>5}%")
    for p in manifest["pools"]:
        flag = f"  SHORT by {p['shortfall']:,}" if p["shortfall"] else ""
        print(f"    [{p['pool']:14s}] {p['rows']:>4} rows / {p['tokens']:>8,} tok "
              f"({p['licence']}/{p['teacher']}){flag}")


if __name__ == "__main__":
    main()
