"""Knots NOTAM ingest → clean canonical corpus (Phase 7).

Unlike the METAR/TAF ingests (which pull raw reports and build a gold answer key), NOTAM gold is
GIVEN by the Knots expert annotations, so this module just normalises them into our canonical
`NotamExtraction` shape and enforces the project's **no-Chinese-datapoints** rule:

  • `area_type` (a 100%-Chinese field) is excluded by the schema (NOTAM_FIELDS), so area keeps
    only its English fields;
  • any record with a Chinese character in `raw_text` OR in any kept field value is dropped.

So the output corpus contains no Chinese anywhere. Source: data/third_party/knots/ (gitignored; a
copy of github.com/Estrellajer/Knots `data/output/`).

CLI:  uv run python -m avtext.ingest.knots            # build the clean corpus + print stats
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from avtext.ingest.awc import DATA_DIR
from avtext.schema.notam import NOTAM_FIELDS, ROW_CATEGORIES, has_chinese, normalize_row

KNOTS_DIR = DATA_DIR / "third_party" / "knots"
PROCESSED = DATA_DIR / "processed" / "notam"
OUT = PROCESSED / "notam_clean.jsonl"
CATEGORIES = tuple(NOTAM_FIELDS)  # the 9 canonical categories


def _knots_records(category: str) -> list[dict]:
    """Records for one Knots category file ({metadata, records} or a bare list)."""
    doc = json.loads((KNOTS_DIR / f"{category}.json").read_text(encoding="utf-8"))
    recs = doc.get("records", doc) if isinstance(doc, dict) else doc
    return recs if isinstance(recs, list) else []


def normalize_all() -> list[dict]:
    """Every Knots category → canonical {id, category, raw_text, rows}, Chinese-filtered."""
    out: list[dict] = []
    for cat in CATEGORIES:
        for r in _knots_records(cat):
            rid, raw = r.get("id"), r.get("raw_text")
            if not rid or not raw or has_chinese(raw):
                continue  # need an id + English raw text
            mf = r.get("manual_fields") or {}
            raw_rows = mf.get("rows", []) if cat in ROW_CATEGORIES else [mf]  # flat → single row
            rows = [normalize_row(cat, rr) for rr in raw_rows if isinstance(rr, dict)]
            if not rows or any(has_chinese(row) for row in rows):
                continue  # drop empty, and any record with Chinese in a kept field
            if all(all(v is None for v in row.values()) for row in rows):
                continue  # degenerate — nothing extracted
            out.append({"id": rid, "category": cat, "raw_text": raw, "rows": rows})
    return out


def build(out: Path = OUT) -> dict:
    """Write the clean corpus JSONL and return per-category counts."""
    records = normalize_all()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    by_cat = Counter(r["category"] for r in records)
    return {"total": len(records), "by_category": dict(by_cat), "out": str(out)}


if __name__ == "__main__":
    stats = build()
    print(f"clean NOTAM corpus: {stats['total']} records -> {stats['out']}")
    for cat, n in sorted(stats["by_category"].items(), key=lambda kv: -kv[1]):
        print(f"  {cat:12} {n}")
    # verify: no Chinese anywhere
    recs = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines()]
    zh = sum(has_chinese(r["raw_text"]) or has_chinese(r["rows"]) for r in recs)
    print(f"Chinese datapoints remaining: {zh}  (must be 0)")
