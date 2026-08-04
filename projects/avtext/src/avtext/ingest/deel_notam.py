"""DEEL-AI NOTAM classification ingest → clean corpus (Phase 7).

DEEL-AI/NOTAM (HuggingFace, **MIT**, French DEEL/IRT lab — independent of Knots/OpenNOTAM) is a
single-label classification set: raw NOTAM → one of 13 top-level classes. A different task from
extraction (no field extraction), added as a second NOTAM capability + eval. Source CSV is
`text;label` (label = the integer class id in label_mapping.json). Already English (0 Chinese
verified), with a train/test split we reuse.

Source: data/third_party/deel_notam/ (gitignored; downloaded from the HF dataset).
CLI:  uv run python -m avtext.ingest.deel_notam       # build clean classification corpus + stats
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from avtext.ingest.awc import DATA_DIR
from avtext.schema.notam import NOTAM_CLASSES, has_chinese

DEEL_DIR = DATA_DIR / "third_party" / "deel_notam"
PROCESSED = DATA_DIR / "processed" / "notam"
OUT = PROCESSED / "notam_class.jsonl"


def _id_to_class() -> dict[int, str]:
    """label_mapping.json is {class_name: int}; invert to {int: class_name}."""
    m = json.loads((DEEL_DIR / "label_mapping.json").read_text(encoding="utf-8"))
    return {int(v): k for k, v in m.items()}


def normalize_all() -> list[dict]:
    """DEEL-AI CSVs → canonical {id, raw_text, label, split}, English-only, deduped."""
    id2class = _id_to_class()
    out: list[dict] = []
    seen: set[str] = set()
    for split, fn in (("train", "train_data.csv"), ("test", "test_data.csv")):
        with (DEEL_DIR / fn).open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter=";")
            next(reader, None)  # header: text;label
            for row in reader:
                if len(row) < 2:
                    continue
                label_id, text = row[-1].strip(), ";".join(row[:-1]).strip()  # text may hold ';'
                if not text or has_chinese(text):
                    continue
                try:
                    label = id2class[int(label_id)]
                except (ValueError, KeyError):
                    continue  # malformed / unknown label
                if label not in NOTAM_CLASSES:
                    continue
                rid = f"cls-{hashlib.sha1(text.encode()).hexdigest()[:12]}"
                if rid in seen:
                    continue
                seen.add(rid)
                out.append({"id": rid, "raw_text": text, "label": label, "split": split})
    return out


def build(out: Path = OUT) -> dict:
    records = normalize_all()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {
        "total": len(records),
        "by_split": dict(Counter(r["split"] for r in records)),
        "by_class": dict(Counter(r["label"] for r in records)),
        "out": str(out),
    }


if __name__ == "__main__":
    s = build()
    print(f"clean NOTAM classification corpus: {s['total']} ({s['by_split']}) -> {s['out']}")
    for cls, n in sorted(s["by_class"].items(), key=lambda kv: -kv[1]):
        print(f"  {cls:28} {n}")
    recs = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines()]
    zh = sum(has_chinese(r["raw_text"]) for r in recs)
    print(f"Chinese datapoints remaining: {zh}  (must be 0)")
