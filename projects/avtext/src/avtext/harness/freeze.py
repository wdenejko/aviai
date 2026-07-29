"""Freeze eval/v1 — the immutable, reproducible evaluation set (Phase 3).

The harness's single most important guardrail: the eval set is FROZEN (content-hashed,
committed) BEFORE any training, so no result can be quietly tuned against it later. This
builder is deterministic — same corpus + same config -> byte-identical eval.jsonl and the
same sha256 — which is what "frozen" has to mean in practice.

Split policy (chosen 2026-07-29, ADR-006): hold out along BOTH axes so the eval measures
two distinct kinds of generalization, reported separately.

    unseen_station  reports from stations the model will NEVER train on. Tests transfer
                    to an unseen station of a *known format family* — each held-out
                    station keeps same-family siblings in the training set, so the format
                    was learnable; only this station is new.
    unseen_time     reports from training stations but in a held-out recent time window.
                    Tests robustness to new dates (new weather, same formats).
    train           everything else — reserved for Phase 4; never scored here.

The three are disjoint by construction (see `assign_split`). Each eval stratum OVER-WEIGHTS
the messy tail relative to its ~2% natural rate — ADR-004's thesis is that the tail is where
the interesting signal is, so a blind sample (98% clean) would waste the eval budget on a
solved problem. The two strata carry DIFFERENT tail types, and that asymmetry is forced by
the data, not chosen: post-cutoff, parse failures come almost entirely from one station
(UUWW, non-stationary), so abstention is testable only by TIME-holdout on it (unseen_time),
while unseen_station tests decode + disagreement (clean + dissent). See TARGETS / ADR-006.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from avtext.harness.hardcases import CLEAN, DISSENT, PARSE_FAIL, classify

# ── freeze configuration (echoed verbatim into the manifest) ──────────────────
# Held-out stations for the unseen_station stratum, chosen from the MEASURED per-station,
# post-cutoff tail distribution — the stations.yaml tags were wrong (see ADR-006 /
# LEARNING_LOG S10). A hard data constraint shapes this: post-cutoff, UUWW is the ONLY
# meaningful parse-fail source (38%); every other station is <1% (YSSY, the other candidate,
# had its format fixed at 2025-02 and now parses clean). So abstention CANNOT be tested by
# station-holdout — holding out UUWW would strand the sole source with nothing to train on.
# Instead UUWW stays in training and its parse-fails are tested by TIME-holdout (see
# unseen_time). The held-out stations therefore carry the DECODE test (clean + dissent):
#   KEKM  dissent source (RMK T-group); siblings KAIG/KOZW stay in training (format learnable).
#   EGCC/KSEA/EPKK  clean EU/US/PL baselines; each keeps same-family siblings in training.
HELD_OUT_STATIONS = ("KEKM", "EGCC", "KSEA", "EPKK")

# Reports on/after this UTC date (at training stations) are the unseen_time stratum. UUWW's
# parse-fail rate ROSE across it (37% -> 74%): a real distribution shift at a seen station.
TIME_BOUNDARY = "2026-05-01"

# Base-model memorization guard (eval/README rule 1): every eval item must post-date
# Gemma-4-E4B's Jan-2025 cutoff (ADR-003), so nothing here could have been memorized in
# pretraining. unseen_time's 2026-05-01 boundary already clears this; the floor binds the
# unseen_station pool, whose held-out stations would otherwise contribute pre-cutoff history.
CUTOFF_FLOOR = "2025-02-01"

# Target count per (stratum, label) — deliberately ASYMMETRIC, because the data forces it
# (above): only unseen_time can carry an abstention/parse_fail test (UUWW post-boundary);
# unseen_station has no post-cutoff parse-fail source and tests decode + dissent only. A
# ceiling, not a floor: a short cell is taken in full and logged, never padded.
TARGETS = {
    "unseen_station": {CLEAN: 200, DISSENT: 60},
    "unseen_time": {CLEAN: 200, PARSE_FAIL: 100, DISSENT: 60},
}

# How many reports to classify per stratum to fill the targets. The tail concentrates in
# a few stations, so scan generously; 30k stays a few seconds of CPU.
SCAN_PER_STRATUM = 30_000

_CORPUS = Path("data/processed/metar/iem_metar.parquet")
_OUT = Path("eval/v1")


def assign_split(station: str, valid_utc: str, held_out: frozenset[str], boundary: str) -> str:
    """Which disjoint bucket a report belongs to. Pure (no corpus) so it is unit-testable.

    valid_utc is an ISO string ('2026-05-01 12:30:00'); lexical >= against a 'YYYY-MM-DD'
    boundary is a correct date comparison because ISO-8601 sorts chronologically.
    """
    if station in held_out:
        return "unseen_station"  # held-out station -> eval, regardless of time
    if valid_utc >= boundary:
        return "unseen_time"  # training station, held-out window -> eval
    return "train"  # training station, training window -> reserved for finetune


def _quote_list(items: tuple[str, ...]) -> str:
    return ", ".join(f"'{s}'" for s in items)


def _scan(where: str) -> list[tuple[str, str, str]]:
    """Deterministic, STATION-STRATIFIED pull of (station, valid_utc, raw) for a pool.

    Equal per-station quota (not a global LIMIT): otherwise high-frequency US stations —
    all clean — swamp the sample and the rare parse-fail stations (YSSY, ~18%) barely
    appear, starving the tail. Same US-volume trap the miner already avoids. md5 ordering
    keeps it reproducible.
    """
    base = f"read_parquet('{_CORPUS.as_posix()}') WHERE raw IS NOT NULL AND ({where})"
    n_stations = duckdb.sql(f"SELECT count(DISTINCT station) FROM {base}").fetchone()[0]
    per_station = max(1, SCAN_PER_STRATUM // n_stations)
    # The OUTER ORDER BY is load-bearing: _select_stratum takes the first N of each label
    # in the order rows arrive, so that order must be fixed. Without it DuckDB streams rows
    # in nondeterministic parallel order and the frozen set's hash changes run-to-run.
    q = f"""
        SELECT station, valid_utc, raw FROM (
            SELECT station, CAST(valid_utc AS VARCHAR) AS valid_utc, raw,
                   row_number() OVER (
                       PARTITION BY station
                       ORDER BY md5(raw || CAST(valid_utc AS VARCHAR))
                   ) AS rn
            FROM {base}
        ) WHERE rn <= {per_station}
        ORDER BY md5(raw || CAST(valid_utc AS VARCHAR))
    """
    return duckdb.sql(q).fetchall()


def _select_stratum(stratum: str, rows: list[tuple[str, str, str]]) -> tuple[list[dict], dict]:
    """Classify a pool, dedup by raw, and take up to the stratum's target of each bucket
    in scan order. Returns (records, per-label counts actually taken)."""
    targets = TARGETS[stratum]
    taken: dict[str, int] = {lbl: 0 for lbl in targets}
    seen_raw: set[str] = set()
    records: list[dict] = []
    for station, valid_utc, raw in rows:
        if raw in seen_raw:
            continue  # distinct reports only — CAVOK repeats must not flood the set
        seen_raw.add(raw)
        v = classify(raw)
        if v.label not in targets or taken[v.label] >= targets[v.label]:
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
                # reference: gold-validated for clean/dissent; partial for parse_fail
                "reference": v.reference,
                "failed": v.failed,
                "dissent": v.dissent,
            }
        )
    return records, taken


def build(out: Path) -> dict:
    held_sql = _quote_list(HELD_OUT_STATIONS)
    after_floor = f"CAST(valid_utc AS VARCHAR) >= '{CUTOFF_FLOOR}'"
    after_boundary = f"CAST(valid_utc AS VARCHAR) >= '{TIME_BOUNDARY}'"

    pools = {
        # held-out station, but still post-cutoff so the base model can't have memorized it
        "unseen_station": f"station IN ({held_sql}) AND {after_floor}",
        "unseen_time": f"station NOT IN ({held_sql}) AND {after_boundary}",
    }

    records: list[dict] = []
    composition: dict[str, dict[str, int]] = {}
    for stratum, where in pools.items():
        rows = _scan(where)
        recs, taken = _select_stratum(stratum, rows)
        records.extend(recs)
        composition[stratum] = taken
        tgt = TARGETS[stratum]
        short = {lbl: tgt[lbl] - n for lbl, n in taken.items() if n < tgt[lbl]}
        note = f"  (short: {short})" if short else ""
        print(f"{stratum:16s} scanned {len(rows):6d} -> {taken}{note}")

    # Stable order (by id) so the file — and its hash — are reproducible.
    records.sort(key=lambda r: r["id"])
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records]
    payload = "\n".join(lines) + "\n"
    set_hash = hashlib.sha256(payload.encode()).hexdigest()

    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.jsonl").write_text(payload, encoding="utf-8")

    manifest = {
        "version": "v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),  # metadata, not hashed
        "sha256": set_hash,  # over eval.jsonl — the immutability anchor
        "n_records": len(records),
        "corpus": _CORPUS.as_posix(),
        "corpus_rows": duckdb.sql(
            f"SELECT count(*) FROM read_parquet('{_CORPUS.as_posix()}')"
        ).fetchone()[0],
        "split_policy": "station+time holdout (ADR-006)",
        "held_out_stations": list(HELD_OUT_STATIONS),
        "time_boundary_utc": TIME_BOUNDARY,
        "cutoff_floor_utc": CUTOFF_FLOOR,
        "targets_per_stratum": TARGETS,
        "scan_per_stratum": SCAN_PER_STRATUM,
        "composition": composition,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nfroze {len(records)} records -> {out / 'eval.jsonl'}  sha256={set_hash[:16]}…")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the immutable eval/v1 set.")
    ap.add_argument("--out", type=Path, default=_OUT)
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
