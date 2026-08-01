"""Freeze eval/taf/v1 — the immutable, reproducible TAF evaluation set (Phase 6).

The TAF sibling of harness/freeze.py, same guardrail and same method: deterministic build →
byte-identical eval.jsonl → same sha256, frozen BEFORE any training. Same two-axis split policy
(ADR-006) so we measure two kinds of generalization separately:

    unseen_station  TAFs from stations the model will never train on (transfer to a new station
                    of a known format family — same-region siblings stay in training).
    unseen_time     TAFs from training stations in the held-out most-recent window (robustness to
                    new dates). The corpus is only ~30 days deep (the AWC API's hard ceiling), so
                    this window is 5 days, not METAR's months — a real, documented constraint.
    train           everything else — reserved for SFT; never scored here.

Differences from the METAR freeze, all corpus-shaped:
  • the corpus is the AWC global parquet (station_id / issue_time / raw_text columns);
  • labels come from classify_taf (2-voice consensus); TAF parse rarely fails, so PARSE_FAIL is
    small and DISSENT is ~2% — the eval is intentionally clean-heavy, because a TAF's hard part is
    whole-forecast EM across nested change groups, not the (easy) tail;
  • no cutoff floor is needed: every TAF here is 2026-07/08, well after Gemma-4's training data,
    so none could have been memorized.
  • held-out stations are selected DETERMINISTICALLY (md5 of the id, 15% of well-covered stations)
    rather than hand-listed — there are ~1,900 of them — and the resulting list is echoed into the
    manifest, so the set stays reproducible and auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from avtext.harness.hardcases import CLEAN, DISSENT, PARSE_FAIL
from avtext.harness.hardcases_taf import classify_taf

CORPUS = Path("data/processed/taf/awc_taf_global.parquet")
OUT = Path("eval/taf/v1")

# Most-recent 5 days of the ~30-day corpus (2026-07-01 .. 08-01) are the unseen_time window.
TIME_BOUNDARY = "2026-07-27"
# A station is a candidate for holdout only if it's well-covered (>= this many TAFs over 30 d),
# so a held-out station actually contributes enough eval records to matter.
MIN_TAFS = 20
HELD_OUT_PCT = 15  # deterministic md5-based holdout fraction of well-covered stations

# Target per (stratum, label). Clean-heavy on purpose (see module docstring); short tail cells are
# taken in full and logged, never padded.
TARGETS = {
    "unseen_station": {CLEAN: 2500, DISSENT: 200},
    "unseen_time": {CLEAN: 2500, DISSENT: 200, PARSE_FAIL: 300},
}
# Scan the full pools (not a sample) so the rare DISSENT/PARSE_FAIL tail is actually found.
SCAN_PER_STRATUM = 120_000


def held_out_stations(
    corpus: Path = CORPUS, pct: int = HELD_OUT_PCT, min_tafs: int = MIN_TAFS
) -> tuple[str, ...]:
    """Deterministic geo-uniform holdout: md5(id) %% 100 < pct, among well-covered stations.
    Pure function of the frozen corpus, so the eval stays reproducible; recorded in the manifest."""
    q = (
        f"SELECT station_id FROM read_parquet('{corpus.as_posix()}') "
        f"GROUP BY station_id HAVING count(*) >= {min_tafs}"
    )
    stations = [r[0] for r in duckdb.sql(q).fetchall()]
    held = [s for s in stations if int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 100 < pct]
    return tuple(sorted(held))


def _quote_list(items: tuple[str, ...]) -> str:
    return ", ".join(f"'{s}'" for s in items)


def _scan(where: str, corpus: Path, scan_per_stratum: int) -> list[tuple[str, str, str]]:
    """Deterministic, STATION-STRATIFIED pull of (station, valid_utc, raw) — same design as the
    METAR scanner (equal per-station quota so high-frequency stations don't swamp the tail; md5
    ordering for reproducibility), adapted to the AWC parquet's column names."""
    base = f"read_parquet('{corpus.as_posix()}') WHERE raw_text IS NOT NULL AND ({where})"
    n_stations = duckdb.sql(f"SELECT count(DISTINCT station_id) FROM {base}").fetchone()[0] or 1
    per_station = max(1, scan_per_stratum // n_stations)
    q = f"""
        SELECT station, valid_utc, raw FROM (
            SELECT station_id AS station, CAST(issue_time AS VARCHAR) AS valid_utc,
                   raw_text AS raw,
                   row_number() OVER (
                       PARTITION BY station_id
                       ORDER BY md5(raw_text || CAST(issue_time AS VARCHAR))
                   ) AS rn
            FROM {base}
        ) WHERE rn <= {per_station}
        ORDER BY md5(raw || CAST(valid_utc AS VARCHAR))
    """
    return duckdb.sql(q).fetchall()


def _select_stratum(
    stratum: str, rows: list[tuple[str, str, str]], targets: dict
) -> tuple[list[dict], dict]:
    """Classify a pool, dedup by raw, take up to each label's target in scan order."""
    tgt = targets[stratum]
    taken: dict[str, int] = {lbl: 0 for lbl in tgt}
    seen_raw: set[str] = set()
    records: list[dict] = []
    for station, valid_utc, raw in rows:
        if raw in seen_raw:
            continue  # distinct forecasts only
        seen_raw.add(raw)
        v = classify_taf(raw)
        if v.label not in tgt or taken[v.label] >= tgt[v.label]:
            continue
        taken[v.label] += 1
        rid = hashlib.sha1(f"{station}|{valid_utc}|{raw}".encode()).hexdigest()[:12]
        records.append(
            {
                "id": rid,
                "split": stratum,
                "station": station,
                "valid_utc": valid_utc,
                "raw": raw,
                "label": v.label,
                "reference": v.reference,
                "failed": v.failed,
                "dissent": v.dissent,
            }
        )
    return records, taken


def build(
    out: Path = OUT,
    *,
    corpus: Path = CORPUS,
    boundary: str = TIME_BOUNDARY,
    targets: dict = TARGETS,
    scan_per_stratum: int = SCAN_PER_STRATUM,
    version: str = "taf-v1",
) -> dict:
    """Freeze the immutable TAF eval set. Deterministic given the frozen corpus."""
    held = held_out_stations(corpus)
    held_sql = _quote_list(held)
    after_boundary = f"CAST(issue_time AS VARCHAR) >= '{boundary}'"
    pools = {
        "unseen_station": f"station_id IN ({held_sql})",
        "unseen_time": f"station_id NOT IN ({held_sql}) AND {after_boundary}",
    }

    records: list[dict] = []
    composition: dict[str, dict[str, int]] = {}
    for stratum, where in pools.items():
        rows = _scan(where, corpus, scan_per_stratum)
        recs, taken = _select_stratum(stratum, rows, targets)
        records.extend(recs)
        composition[stratum] = taken
        short = {
            lbl: targets[stratum][lbl] - n for lbl, n in taken.items() if n < targets[stratum][lbl]
        }
        note = f"  (short: {short})" if short else ""
        print(f"{stratum:16s} scanned {len(rows):6d} -> {taken}{note}")

    records.sort(key=lambda r: r["id"])
    payload = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records) + "\n"
    set_hash = hashlib.sha256(payload.encode()).hexdigest()

    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.jsonl").write_text(payload, encoding="utf-8")
    manifest = {
        "version": version,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": set_hash,
        "n_records": len(records),
        "corpus": corpus.as_posix(),
        "corpus_rows": duckdb.sql(
            f"SELECT count(*) FROM read_parquet('{corpus.as_posix()}')"
        ).fetchone()[0],
        "split_policy": "station+time holdout (ADR-006), TAF",
        "held_out_pct": HELD_OUT_PCT,
        "held_out_min_tafs": MIN_TAFS,
        "n_held_out_stations": len(held),
        "held_out_stations": list(held),
        "time_boundary_utc": boundary,
        "targets_per_stratum": targets,
        "scan_per_stratum": scan_per_stratum,
        "composition": composition,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nfroze {len(records)} records -> {out / 'eval.jsonl'}  sha256={set_hash[:16]}…")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the immutable TAF eval set (eval/taf/v1).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    build(args.out or OUT)


if __name__ == "__main__":
    main()
