"""Gold-seed calibration — tier-2 (Phase 2, Skill B).

The trust ladder's answer to the mutant audit's blind spot: consensus + invariants
catch disagreement and impossibility, never a plausible-but-wrong value — only an
independent AUTHORITY catches those. `calibrate` scores the parser consensus against
authoritative reference records and reports per-field accuracy + every divergence
(a divergence indicts the parsers OR the reference — both are worth a look).

Reference sources, in ascending authority:
  - **AWC official decode** (NOAA's own, machine-readable, scalable) — used to run a
    real calibration now (`gold_from_awc_rows`).
  - **Hand-transcribed worked examples** from AC 00-45H / FMH-1 / NWS decode key
    (copy-only, zero judgment) — the true Tier-2 gold. A human copy-task that plugs
    into the same `GoldRecord` format; deliberately NOT fabricated here, because
    correctness must come from an authority, never from our own aviation judgment.
"""

from __future__ import annotations

from dataclasses import dataclass

from avtext.consensus.vote import FIELDS, consensus
from avtext.oracles import ORACLES

_INHG_TO_HPA = 33.8639
_SM_TO_M = 1609.344


@dataclass
class GoldRecord:
    raw: str
    fields: dict[str, object]  # authoritative canonical values (a subset of FIELDS)
    source: str


@dataclass
class CalibrationReport:
    per_field: dict[str, tuple[int, int, float | None]]  # field -> (n, matches, accuracy)
    divergences: list[tuple[str, str, object, object]]  # (raw, field, consensus, gold)


def calibrate(gold: list[GoldRecord]) -> CalibrationReport:
    """Score the 3-parser consensus against authoritative `gold` records."""
    tally = {f: [0, 0] for f in FIELDS}  # field -> [n_compared, n_match]
    divergences: list[tuple[str, str, object, object]] = []
    for g in gold:
        obs = [r.obs for o in ORACLES if (r := o.decode(g.raw)).ok]
        if not obs:
            continue
        con = consensus(obs).values
        for f, gold_val in g.fields.items():
            con_val = con.get(f)
            if gold_val is None or con_val is None:
                continue  # only compare where both have an opinion
            tally[f][0] += 1
            if con_val == gold_val:
                tally[f][1] += 1
            else:
                divergences.append((g.raw, f, con_val, gold_val))
    per_field = {f: (n, m, (m / n if n else None)) for f, (n, m) in tally.items()}
    return CalibrationReport(per_field, divergences)


def gold_from_awc_rows(rows: list[dict[str, str]]) -> list[GoldRecord]:
    """Map AWC cache rows (raw_text + NOAA's decoded columns) into GoldRecords,
    canonicalized to the schema's units. Scalars only — the cache's repeated cloud
    columns need positional parsing (a later refinement)."""

    def num(x: str | None) -> float | None:
        try:
            return float(x)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    gold: list[GoldRecord] = []
    for row in rows:
        raw = row.get("raw_text")
        if not raw:
            continue
        alt_in = num(row.get("altim_in_hg"))
        vis_mi = num(row.get("visibility_statute_mi"))
        fields = {
            "wind_dir": _round(num(row.get("wind_dir_degrees"))),
            "wind_speed": _round(num(row.get("wind_speed_kt"))),
            "wind_gust": _round(num(row.get("wind_gust_kt"))),
            "temperature_c": _round(num(row.get("temp_c"))),
            "dewpoint_c": _round(num(row.get("dewpoint_c"))),
            "altimeter_hpa": _round(alt_in * _INHG_TO_HPA) if alt_in is not None else None,
            "visibility_m": round(vis_mi * _SM_TO_M / 100) * 100 if vis_mi is not None else None,
        }
        gold.append(
            GoldRecord(raw, {k: v for k, v in fields.items() if v is not None}, "awc-official")
        )
    return gold


def _round(x: float | None) -> int | None:
    return round(x) if x is not None else None
