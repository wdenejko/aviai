"""Phase 1 scope probe — is the fine-tuning premise justified?

Question (finetune briefing §7.1): on a REAL feed, what fraction of METARs does a
good parser fail on, or where do independent parsers disagree? That fraction is
the messy-tail prize (ADR-004). If it's ~0, an LLM has no headroom here; if it's
meaningful, that failure/disagreement set is the fine-tune's target and the seed
of the hard-case queue.

Deliberately a *probe*, not the oracle layer: we call each parser's NATIVE API on
a few key fields, with no canonical schema yet (that's Phase 2). The goal is one
decision-informing number + a seed hard-case sample, fast.

Four "voices":
  - python-metar, avwx-engine, mivek  (3 independent parsers)
  - AWC's own decoded columns          (bundled ground truth, free 4th voice)

Failure semantics differ and MUST be handled per-parser (learned empirically):
  - python-metar : raises metar.Metar.ParserError on bad input
  - mivek        : raises ValueError / others on bad input
  - avwx-engine  : NEVER raises — returns nulls; we treat all-core-fields-null as
                   an effective failure.
"""

from __future__ import annotations

import json

from avtext.ingest.awc import DATA_DIR, _utcstamp, fetch_metar_cache, parse_metar_cache

# Four scalar fields that every voice reports in the same units (deg, kt, °C),
# so cross-voice comparison needs no unit juggling — perfect for a fast probe.
CORE_FIELDS = ("wind_dir", "wind_speed", "temp", "dewpoint")


def _to_kt(speed: float | None, unit: object, raw: str | None = None) -> float | None:
    """Normalize a wind speed to knots. METAR reports wind in KT or MPS (m/s), and
    parsers vary in whether they auto-convert — so we normalize here. (A preview of
    the unit-handling the Phase-2 oracle layer must own; the probe first surfaced
    this as spurious wind_speed 'disagreements' on Russian/Chinese MPS reports.)"""
    if speed is None:
        return None
    is_mps = str(unit).upper() in ("MPS", "M/S") or (
        unit is None and raw is not None and "MPS" in raw
    )
    return speed * 1.94384 if is_mps else speed


# ── per-parser field extractors (native APIs) ────────────────────────────────
def _pm(raw: str) -> dict[str, float | None]:
    from metar.Metar import Metar

    m = Metar(raw)
    return {
        "wind_dir": m.wind_dir.value() if m.wind_dir else None,
        "wind_speed": m.wind_speed.value("KT") if m.wind_speed else None,
        "temp": m.temp.value("C") if m.temp else None,
        "dewpoint": m.dewpt.value("C") if m.dewpt else None,
    }


def _avwx(raw: str) -> dict[str, float | None]:
    import avwx

    report = avwx.Metar.from_report(raw)  # units live on the report, not on .data
    data = report.data
    if data is None:
        raise ValueError("avwx returned no data")
    unit = getattr(report.units, "wind_speed", None)
    fields = {
        "wind_dir": getattr(data.wind_direction, "value", None),
        "wind_speed": _to_kt(getattr(data.wind_speed, "value", None), unit),
        "temp": getattr(data.temperature, "value", None),
        "dewpoint": getattr(data.dewpoint, "value", None),
    }
    if all(v is None for v in fields.values()):
        raise ValueError("avwx parsed nothing (all core fields null)")
    return fields


def _mivek(raw: str) -> dict[str, float | None]:
    from metar_taf_parser.parser.parser import MetarParser

    parts = raw.split(None, 1)
    body = parts[1] if parts and parts[0] in ("METAR", "SPECI") else raw
    m = MetarParser().parse(body)
    w = m.wind
    return {
        "wind_dir": getattr(w, "degrees", None) if w else None,
        "wind_speed": _to_kt(getattr(w, "speed", None), getattr(w, "unit", None), raw)
        if w
        else None,
        "temp": m.temperature,
        "dewpoint": m.dew_point,
    }


PARSERS = {"python-metar": _pm, "avwx": _avwx, "mivek": _mivek}


def parse_one(raw: str) -> dict[str, dict]:
    out = {}
    for name, fn in PARSERS.items():
        try:
            out[name] = {"ok": True, "fields": fn(raw), "error": None}
        except Exception as e:  # noqa: BLE001 — a probe: any failure is a data point
            out[name] = {"ok": False, "fields": None, "error": f"{type(e).__name__}: {e}"}
    return out


def _awc_voice(row: dict[str, str]) -> dict[str, float | None]:
    def num(x: str | None) -> float | None:
        try:
            return float(x)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    return {
        "wind_dir": num(row.get("wind_dir_degrees")),
        "wind_speed": num(row.get("wind_speed_kt")),
        "temp": num(row.get("temp_c")),
        "dewpoint": num(row.get("dewpoint_c")),
    }


def _norm(field: str, v: float | None) -> int | None:
    """Round to a comparable integer. Treat None as 'no opinion' (not a value), so
    differing null conventions (e.g. VRB wind: parser=None vs AWC=0) don't count
    as disagreements — only genuine value conflicts do."""
    if v is None:
        return None
    if field == "wind_dir":
        return int(round(v)) % 360  # 360 and 0 are the same heading
    return int(round(v))


def _distinct(values: list[float | None], field: str) -> set[int]:
    return {n for n in (_norm(field, v) for v in values) if n is not None}


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "  n/a"


# ── the probe ────────────────────────────────────────────────────────────────
def run(limit: int | None = None) -> dict:
    rows = parse_metar_cache(fetch_metar_cache())
    if limit:
        rows = rows[:limit]
    n = len(rows)

    fail = dict.fromkeys(PARSERS, 0)
    any_fail = all_fail = 0
    auto_n = auto_fail = manned_n = manned_fail = 0
    dis_field = dict.fromkeys(CORE_FIELDS, 0)
    dis_any = dis_any_awc = comparable = 0
    hardcases: list[dict] = []

    for row in rows:
        raw = row["raw_text"]
        pr = parse_one(raw)
        failed = [name for name in PARSERS if not pr[name]["ok"]]
        for name in failed:
            fail[name] += 1

        is_auto = row.get("auto") == "TRUE"
        auto_n, manned_n = auto_n + is_auto, manned_n + (not is_auto)
        reasons: list[str] = []

        if failed:
            any_fail += 1
            auto_fail += is_auto
            manned_fail += not is_auto
            reasons.append("parse_fail:" + ",".join(failed))
            if len(failed) == len(PARSERS):
                all_fail += 1

        else:  # all three parsed → check whether they AGREE
            comparable += 1
            field_conflict = False
            for f in CORE_FIELDS:
                vals = [pr[name]["fields"][f] for name in PARSERS]
                if len(_distinct(vals, f)) > 1:
                    dis_field[f] += 1
                    field_conflict = True
                    reasons.append(f"disagree:{f}={sorted(_distinct(vals, f))}")
            dis_any += field_conflict
            # add AWC's decoded voice: does the bundled ground truth break ties?
            awc = _awc_voice(row)
            awc_conflict = any(
                len(_distinct([pr[name]["fields"][f] for name in PARSERS] + [awc[f]], f)) > 1
                for f in CORE_FIELDS
            )
            dis_any_awc += awc_conflict

        if reasons:
            hardcases.append(
                {"raw": raw, "station": row.get("station_id"), "auto": is_auto, "reasons": reasons}
            )

    messy = any_fail + dis_any  # a report is "messy" if it failed OR the parsers disagreed

    # ── report ───────────────────────────────────────────────────────────────
    print("\n=== SCOPE PROBE — AWC bulk METAR cache ===")
    print(f"sample: N={n}   (auto={auto_n}, manned={manned_n})\n")
    print("-- hard-failure rate (parser raised / produced nothing) --")
    for name in PARSERS:
        print(f"  {name:<13}: {_pct(fail[name], n)}  ({fail[name]})")
    print(f"  {'any parser':<13}: {_pct(any_fail, n)}  ({any_fail})")
    print(f"  {'all parsers':<13}: {_pct(all_fail, n)}  ({all_fail})")
    af, mf = _pct(auto_fail, auto_n), _pct(manned_fail, manned_n)
    print(f"\n  failure by type: AUTO {af}  vs  manned {mf}")
    print(f"\n-- disagreement among the 3 parsers (of {comparable} all-parsed) --")
    for f in CORE_FIELDS:
        print(f"  {f:<13}: {_pct(dis_field[f], comparable)}  ({dis_field[f]})")
    print(f"  {'any field':<13}: {_pct(dis_any, comparable)}  ({dis_any})")
    aw = _pct(dis_any_awc, comparable)
    print(f"  {'+ AWC voice':<13}: {aw}  ({dis_any_awc})   <- + bundled ground truth")
    mp = _pct(messy, n)
    print(f"\n=> MESSY-TAIL FRACTION (any fail OR disagreement): {mp}  ({messy}/{n})")

    # ── persist: seed the hard-case queue + a summary (gitignored bulk) ────────
    outdir = DATA_DIR / "processed" / "scope_probe" / f"dt={_utcstamp()}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "hardcases.jsonl").write_text("".join(json.dumps(hc) + "\n" for hc in hardcases))
    summary = {
        "n": n,
        "auto_n": auto_n,
        "manned_n": manned_n,
        "hard_failure": {name: fail[name] for name in PARSERS},
        "any_fail": any_fail,
        "all_fail": all_fail,
        "auto_fail": auto_fail,
        "manned_fail": manned_fail,
        "comparable": comparable,
        "disagree_field": dis_field,
        "disagree_any": dis_any,
        "disagree_any_with_awc": dis_any_awc,
        "messy_fraction": messy / n if n else None,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rel = outdir.relative_to(DATA_DIR.parent)
    print(f"\nhard cases: {len(hardcases)} saved -> {rel}/hardcases.jsonl")
    return summary


if __name__ == "__main__":
    run()
