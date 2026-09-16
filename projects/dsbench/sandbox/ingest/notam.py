"""Ingest the DEEL-AI/NOTAM corpus into aviation.notam.

Source: DEEL-AI/NOTAM (HF, MIT-licensed, English) — 13-class NOTAM classification, already vendored
in avtext at data/third_party/deel_notam/ (text;label CSV + label_mapping.json). MIT means this one
IS publishable (with attribution), unlike the quarantined opennotam/knots extraction sets.

Caveat (recorded in the manifest): this is a STATIC ~2024 corpus, NOT date-aligned to the flights/
weather month and carrying no airport/time keys, so it is a standalone NOTAM reference corpus (for
classification / text tasks), not something to join to flights on airport+time. See ADR-003 §4.

  uv run --package dsbench python projects/dsbench/sandbox/ingest/notam.py
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from _ch import get_client

SANDBOX = Path(__file__).resolve().parents[1]
MANIFESTS = SANDBOX / "manifests"
DEEL = SANDBOX.parents[2] / "projects" / "avtext" / "data" / "third_party" / "deel_notam"

DDL = """CREATE TABLE IF NOT EXISTS aviation.notam (
    text String,                        -- the NOTAM E-field text
    label_id UInt8,                     -- DEEL-AI class id (0-12)
    category LowCardinality(String),    -- class name (e.g. Runway, Taxiway, GPS, Obstacles)
    split LowCardinality(String),       -- train | test (the dataset's own split)
    source LowCardinality(String)
) ENGINE = MergeTree ORDER BY (category, label_id)"""


def _read_split(path: Path, split: str, id2cat: dict[int, str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")  # text;label, quoted multi-line E-fields
    df = df.rename(columns={"text": "text", "label": "label_id"})
    df["text"] = df["text"].fillna("").astype(str)
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce").fillna(0).astype("uint8")
    df["category"] = df["label_id"].map(id2cat).fillna("Unknown")
    df["split"] = split
    df["source"] = "deel-ai"
    return df[["text", "label_id", "category", "split", "source"]]


def load() -> dict:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((DEEL / "label_mapping.json").read_text())  # {name: id}
    id2cat = {int(v): k for k, v in mapping.items()}

    frames = []
    for split, fname in [("train", "train_data.csv"), ("test", "test_data.csv")]:
        df = _read_split(DEEL / fname, split, id2cat)
        frames.append(df)
        print(f"  {split}: {len(df)} NOTAMs")
    all_df = pd.concat(frames, ignore_index=True)

    client = get_client()
    client.command(DDL)
    client.command("TRUNCATE TABLE aviation.notam")
    for i in range(0, len(all_df), 20_000):
        client.insert_df("aviation.notam", all_df.iloc[i : i + 20_000])
    loaded = client.command("SELECT count() FROM aviation.notam")

    by_cat = client.query(
        "SELECT category, count() FROM aviation.notam GROUP BY category ORDER BY count() DESC"
    ).result_rows
    manifest = {
        "source": "DEEL-AI/NOTAM (HuggingFace)", "license": "MIT (publishable, with attribution)",
        "task": "single-label classification into 13 NOTAM classes", "classes": mapping,
        "rows_loaded": int(loaded), "by_category": {c: n for c, n in by_cat},
        "caveat": "static ~2024 corpus; not date-aligned; no airport/time key",
        "loaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    (MANIFESTS / "notam_deel_ai.json").write_text(json.dumps(manifest, indent=2))
    print(f"  loaded {loaded:,} NOTAMs -> aviation.notam ({len(mapping)} classes)")
    return manifest


if __name__ == "__main__":
    argparse.ArgumentParser(description="Load DEEL-AI NOTAM into ClickHouse").parse_args()
    load()
