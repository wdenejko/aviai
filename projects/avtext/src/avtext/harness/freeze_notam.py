"""Freeze the immutable NOTAM eval sets (Phase 7).

Two evals, one per NOTAM task, each frozen from the SOURCE's own test split (OpenNOTAM / DEEL-AI
ship train/test, so we reuse it rather than inventing a holdout):
  • eval/notam/v1      — extraction: {id, category, raw, reference:{category, rows}}
  • eval/notam_cls/v1  — classification: {id, raw, reference:<class>}
Deterministic (sorted by id) → byte-identical eval.jsonl + sha256, frozen before training. Same
immutability discipline as the METAR/TAF freezes; no sampling needed (the split is given).

CLI:  uv run python -m avtext.harness.freeze_notam
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from avtext.ingest.awc import DATA_DIR

EXTRACT_CORPUS = DATA_DIR / "processed" / "notam" / "notam_clean.jsonl"
CLASS_CORPUS = DATA_DIR / "processed" / "notam" / "notam_class.jsonl"
EXTRACT_OUT = Path("eval/notam/v1")
CLASS_OUT = Path("eval/notam_cls/v1")


def _read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _freeze(records: list[dict], out: Path, version: str, kind: str, corpus: Path) -> dict:
    records = sorted(records, key=lambda r: r["id"])
    payload = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records) + "\n"
    set_hash = hashlib.sha256(payload.encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.jsonl").write_text(payload, encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "version": version,
                "kind": kind,
                "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sha256": set_hash,
                "n_records": len(records),
                "corpus": corpus.name,
                "split": "test",
            },
            indent=2,
        )
        + "\n"
    )
    return {"kind": kind, "n": len(records), "sha256": set_hash, "out": str(out)}


def build_extraction() -> dict:
    recs = [r for r in _read(EXTRACT_CORPUS) if r.get("split") == "test"]
    eval_recs = [
        {
            "id": r["id"],
            "category": r["category"],
            "raw": r["raw_text"],
            "reference": {"category": r["category"], "rows": r["rows"]},
        }
        for r in recs
    ]
    return _freeze(eval_recs, EXTRACT_OUT, "notam-v1", "extraction", EXTRACT_CORPUS)


def build_classification() -> dict:
    recs = [r for r in _read(CLASS_CORPUS) if r.get("split") == "test"]
    eval_recs = [{"id": r["id"], "raw": r["raw_text"], "reference": r["label"]} for r in recs]
    return _freeze(eval_recs, CLASS_OUT, "notam-cls-v1", "classification", CLASS_CORPUS)


if __name__ == "__main__":
    for stats in (build_extraction(), build_classification()):
        print(
            f"froze {stats['kind']:14} {stats['n']:>5} records -> {stats['out']}"
            f"  sha256={stats['sha256'][:16]}…"
        )
