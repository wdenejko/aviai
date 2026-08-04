"""OpenNOTAM ingest → clean canonical corpus (Phase 7). Supersedes ingest/knots.py.

OpenNOTAM (github.com/Estrellajer/OpenNOTAM) is the Knots EXPERT gold reformatted as instruction/
input/output SFT triples — verified label-identical to Knots (runway 538/538 match). We prefer it
because it (a) exposes `area_type` as a 6-value Chinese ENUM we remap to English (recovering the
whole area category with its type — vs Knots where it collapsed to 797), (b) adds `rvr` + `standard`
categories, and (c) ships a train/test split we reuse for the eval instead of inventing our own.

Same no-Chinese rule: remap area_type, then drop any record still holding a Chinese character in the
raw NOTAM or any field. OpenNOTAM's `output` is mixed-shape ({"rows":[…]} for multi-subject NOTAMs,
a flat dict for single ones), handled uniformly here.

Source: data/third_party/opennotam/ (gitignored; a copy of OpenNOTAM data/dataset/).
CLI:  uv run python -m avtext.ingest.opennotam        # build clean corpus + stats
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from avtext.ingest.awc import DATA_DIR
from avtext.schema.notam import NOTAM_FIELDS, has_chinese, normalize_row

OPENNOTAM_DIR = DATA_DIR / "third_party" / "opennotam"
PROCESSED = DATA_DIR / "processed" / "notam"
OUT = PROCESSED / "notam_clean.jsonl"

_FILE_CAT = {"light_1": "light"}  # OpenNOTAM's file name -> our canonical category


def _rows_from_output(output: str) -> list[dict]:
    """Parse an OpenNOTAM `output` into a list of row dicts. Handles {"rows":[…]} (multi-subject),
    a flat dict (single subject), and a bare list — the format varies within a category."""
    try:
        parsed = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, dict):
        rows = parsed.get("rows")
        return rows if isinstance(rows, list) else [parsed]
    return parsed if isinstance(parsed, list) else []


def normalize_all() -> list[dict]:
    """Every OpenNOTAM train/test file → canonical NOTAM records, Chinese-free."""
    out: list[dict] = []
    seen: set[str] = set()
    for f in sorted(OPENNOTAM_DIR.glob("*_train.json")) + sorted(OPENNOTAM_DIR.glob("*_test.json")):
        filecat, split = f.stem.rsplit("_", 1)
        cat = _FILE_CAT.get(filecat, filecat)
        if cat not in NOTAM_FIELDS:
            continue  # skip non-category files (e.g. segmentation_examples)
        for r in json.load(open(f, encoding="utf-8")):
            raw = (r.get("input") or "").strip()
            if not raw or has_chinese(raw):
                continue
            rid = f"{cat}-{hashlib.sha1(raw.encode()).hexdigest()[:12]}"
            if rid in seen:
                continue  # dedup identical (category, raw) across files
            rows = [
                normalize_row(cat, x)
                for x in _rows_from_output(r.get("output", ""))
                if isinstance(x, dict)
            ]
            if not rows or any(has_chinese(row) for row in rows):
                continue  # drop empty + any record with residual Chinese
            if all(all(v is None for v in row.values()) for row in rows):
                continue  # degenerate — nothing extracted
            seen.add(rid)
            out.append({"id": rid, "category": cat, "raw_text": raw, "rows": rows, "split": split})
    return out


def build(out: Path = OUT) -> dict:
    """Write the clean corpus JSONL; return per-category and per-split counts."""
    records = normalize_all()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {
        "total": len(records),
        "by_category": dict(Counter(r["category"] for r in records)),
        "by_split": dict(Counter(r["split"] for r in records)),
        "out": str(out),
    }


if __name__ == "__main__":
    s = build()
    print(f"clean NOTAM corpus: {s['total']} records ({s['by_split']}) -> {s['out']}")
    for cat, n in sorted(s["by_category"].items(), key=lambda kv: -kv[1]):
        print(f"  {cat:12} {n}")
    recs = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines()]
    zh = sum(has_chinese(r["raw_text"]) or has_chinese(r["rows"]) for r in recs)
    print(f"Chinese datapoints remaining: {zh}  (must be 0)")
