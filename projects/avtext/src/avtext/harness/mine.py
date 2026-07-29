"""Mine the hard-case pool from the IEM corpus (Phase 3).

A one-off, reproducible scan: take a deterministic, station-stratified sample of the
backfilled corpus, run each report through `classify`, and write the labeled pool plus
a manifest. Two design choices worth studying:

  * Deterministic — the eval set built on top of this must be reproducible, so the
    sample must be too. We order rows by md5(raw+timestamp), a stable pseudo-random
    shuffle with no RNG seed to thread: the same corpus always yields the same rows.

  * Station-stratified — 6.4M rows are dominated by high-volume US airports. An equal
    per-station quota keeps the rare regional formats (Scandinavian AUTO, CIS m/s wind,
    the messy tail ADR-004 targets) from being drowned out by Kxxx clear-day repeats.

Output (under data/processed/, regenerable — the *frozen* eval set is the committed
artifact, built from this pool in the next step):
    hardcases/pool.jsonl      one JSON verdict per line
    hardcases/manifest.json   params, per-label counts, corpus fingerprint
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import duckdb

from avtext.harness.hardcases import CLEAN, LABELS, classify, summarize
from avtext.oracles import ORACLES

_CORPUS = Path("data/processed/metar/iem_metar.parquet")
_OUT_DIR = Path("data/processed/hardcases")


def sample(corpus: Path, per_station: int) -> list[tuple[str, str, str]]:
    """Deterministic, station-stratified sample -> [(station, valid_utc, raw), ...].

    row_number() over an md5 ordering takes the same `per_station` rows from each
    station every run. md5(raw + timestamp) rather than md5(raw) so that a station's
    genuinely-duplicate reports (identical raw at different times) still order stably
    instead of tie-breaking arbitrarily.
    """
    q = f"""
        SELECT station, CAST(valid_utc AS VARCHAR) AS valid_utc, raw
        FROM (
            SELECT station, valid_utc, raw,
                   row_number() OVER (
                       PARTITION BY station
                       ORDER BY md5(raw || CAST(valid_utc AS VARCHAR))
                   ) AS rn
            FROM read_parquet('{corpus.as_posix()}')
            WHERE raw IS NOT NULL
        )
        WHERE rn <= {per_station}
    """
    return duckdb.sql(q).fetchall()


def _corpus_rows(corpus: Path) -> int:
    """Total row count — a cheap corpus fingerprint (parquet metadata, no full scan)."""
    return duckdb.sql(f"SELECT count(*) FROM read_parquet('{corpus.as_posix()}')").fetchone()[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine the hard-case pool from the corpus.")
    ap.add_argument(
        "--per-station", type=int, default=500, help="reports sampled per station"
    )
    ap.add_argument("--corpus", type=Path, default=_CORPUS)
    ap.add_argument("--out", type=Path, default=_OUT_DIR)
    args = ap.parse_args()

    rows = sample(args.corpus, args.per_station)
    print(f"scanning {len(rows)} reports from {args.corpus} ...")

    verdicts = []
    lines = []
    for i, (station, valid_utc, raw) in enumerate(rows, 1):
        v = classify(raw)
        verdicts.append(v)
        lines.append(json.dumps({"station": station, "valid_utc": valid_utc, **asdict(v)}))
        if i % 5000 == 0:
            print(f"  {i}/{len(rows)}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pool.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n = len(verdicts)
    counts = summarize(verdicts)
    manifest = {
        "corpus": args.corpus.as_posix(),
        "corpus_rows": _corpus_rows(args.corpus),  # detects a changed corpus on re-run
        "per_station": args.per_station,
        "oracles": [o.name for o in ORACLES],
        "n_scanned": n,
        "counts": counts,
        "rates": {k: round(c / n, 4) for k, c in counts.items()} if n else {},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nhard-case distribution:")
    for lbl in LABELS:
        print(f"  {lbl:12s} {counts[lbl]:6d}  {counts[lbl] / n:6.2%}")
    hard = n - counts[CLEAN]
    print(f"  {'HARD total':12s} {hard:6d}  {hard / n:6.2%}   -> {args.out / 'pool.jsonl'}")


if __name__ == "__main__":
    main()
