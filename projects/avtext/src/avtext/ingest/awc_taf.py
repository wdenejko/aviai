"""AWC ingest — one-time global TAF collector from the Aviation Weather Center.

Why this module exists (see docs/research/non-us-taf-collection.md): IEM's deep TAF archive is
**US/NWS-only**, but a geo-balanced corpus needs the rest of the world — and the METAR
geo-balancing lesson (Session 19) showed the messy *non-US* tail is where the model's value and
its failures live. There is **no free deep global TAF archive** (the one that exists, Ogimet, is
scraping-hostile), so this is a **one-time / on-demand collection, NOT a scheduled pipeline**
(a deliberate project decision — we collect once, we don't run a daily cron). Two modes, both
hitting NOAA public-domain endpoints:

  • --collect  : one current global SNAPSHOT of the AWC bulk cache (~2,800 TAFs, ~70% non-US).
  • --backfill : the last ~30 DAYS from the AWC Data API — the deepest clean global history there
                 is. Verified live 2026-08-01: `date=` at 30 d back returns data, 31 d+ is empty,
                 so ~30 days is a hard ceiling (NOT 90 — no clean source goes deeper for non-US).

The sources (verified live 2026-08-01):
  cache : https://aviationweather.gov/data/cache/tafs.cache.xml.gz   (XML, not CSV; .csv.gz 404s)
  api   : https://aviationweather.gov/api/data/taf?ids=<batch>&date=YYYYMMDD&format=xml
          — keyless; accepts ~300 station ids per call (1000 fails); one call returns one TAF per
          station (the one valid at that date). So all non-US stations × 30 days is only ~250
          requests total. Both endpoints return the SAME <TAF> schema (raw_text + decoded
          <forecast> voice + lat/lon), so `parse_taf_cache` handles both uniformly.

Discipline (same as the rest of ingest/): immutable raw zone + checksummed manifest; the raw
XML is the source of truth and the parquet is a derived, DEDUPED view (DISTINCT station_id,
issue_time, raw_text) we can rebuild anytime — so the AWC <forecast> voice can be re-extracted
later without a re-fetch. Polite client: descriptive User-Agent, paced calls.

CLI:
  uv run python -m avtext.ingest.awc_taf --collect            # one current global snapshot
  uv run python -m avtext.ingest.awc_taf --backfill           # last ~30 d, all non-US stations
  uv run python -m avtext.ingest.awc_taf --backfill --all     # ...including US stations too
  uv run python -m avtext.ingest.awc_taf --parquet            # rebuild deduped corpus parquet
  uv run python -m avtext.ingest.awc_taf --stats              # coverage of the accumulated corpus
"""

from __future__ import annotations

import gzip
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import httpx

from avtext.ingest.awc import DATA_DIR, RAW_DIR, USER_AGENT, _utcstamp, record_artifact

TAF_CACHE_URL = "https://aviationweather.gov/data/cache/tafs.cache.xml.gz"
TAF_API_URL = "https://aviationweather.gov/api/data/taf"
AWC_TAF_RAW = RAW_DIR / "awc" / "tafs_cache"  # --collect snapshots (pre-gzipped XML)
AWC_TAF_API_RAW = RAW_DIR / "awc" / "taf_api"  # --backfill responses (we gzip them)
PROCESSED = DATA_DIR / "processed" / "taf"

# The AWC Data API serves at most ~30 days of history (verified: 30 d ok, 31 d empty). We cap
# any larger request at this ceiling rather than silently returning nothing for the older dates.
API_MAX_DAYS = 30
# ~300 ids per call works, 1000 fails (URL length / server cap); 200 leaves a safe margin.
API_BATCH = 200

# Top-level <TAF> fields we lift into the parquet. The nested <forecast> decode is intentionally
# NOT flattened here — it stays in the immutable raw XML and becomes the AWC "voice" at consensus
# time, so the collector never has to guess the decode schema up front.
_FIELDS = (
    "station_id",
    "raw_text",
    "issue_time",
    "bulletin_time",
    "valid_time_from",
    "valid_time_to",
    "latitude",
    "longitude",
    "elevation_m",
    "remarks",
)


def parse_taf_cache(raw_gz: bytes) -> list[dict[str, str]]:
    """Decode a gzipped AWC TAF XML payload (cache OR api, same schema) into one dict per <TAF>.

    Robust to missing children (findtext -> None) so an odd record never sinks the parse; the
    full raw XML is preserved regardless, so nothing is lost by keeping this lean."""
    root = ET.fromstring(gzip.decompress(raw_gz))
    rows: list[dict[str, str]] = []
    for taf in root.findall(".//TAF"):
        row = {f: taf.findtext(f) for f in _FIELDS}
        if row.get("raw_text"):  # a TAF with no raw text is unusable as a decode target
            rows.append(row)
    return rows


# ── mode 1: current global snapshot (the bulk cache) ─────────────────────────────────────────
def fetch_taf_cache(*, save: bool = True) -> bytes:
    """Fetch the AWC bulk TAF cache (already-gzipped XML); return the raw gzip bytes.

    With save=True, writes the bytes verbatim into a timestamped partition of the immutable raw
    zone and records provenance — the same discipline as awc.fetch_metar_cache."""
    resp = httpx.get(
        TAF_CACHE_URL, headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True
    )
    resp.raise_for_status()
    raw = resp.content
    if save:
        out = AWC_TAF_RAW / f"dt={_utcstamp()}" / "tafs.cache.xml.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
        record_artifact(
            TAF_CACHE_URL, out, raw, source="awc", product="taf", n_tafs=len(parse_taf_cache(raw))
        )
    return raw


def collect(*, save: bool = True) -> dict[str, object]:
    """Take one current snapshot: fetch + parse + report coverage."""
    rows = parse_taf_cache(fetch_taf_cache(save=save))
    cov = _coverage(rows)
    print(f"snapshot: {cov['n']} TAFs ({cov['non_us']} non-US, {cov['non_us_pct']}%)")
    return cov


# ── mode 2: ~30-day backfill (the Data API) ──────────────────────────────────────────────────
def latest_snapshot_stations(*, non_us_only: bool = True) -> list[str]:
    """Station list for the backfill = who currently issues TAFs, read from the newest --collect
    snapshot. Non-US by default (US is already covered deeply by IEM; this corpus is about the
    rest of the world). Raises if no snapshot exists yet — run --collect first."""
    snaps = sorted(AWC_TAF_RAW.rglob("*.xml.gz"))
    if not snaps:
        raise FileNotFoundError("no snapshot yet — run --collect before --backfill")
    ids = sorted(
        {r["station_id"] for r in parse_taf_cache(snaps[-1].read_bytes()) if r["station_id"]}
    )
    return [s for s in ids if s[0] not in "KPT"] if non_us_only else ids


def _fetch_api(ids: list[str], date: str, *, retries: int = 4) -> bytes:
    """One AWC Data API call: TAFs valid at `date` (YYYYMMDD) for a batch of stations, as XML.

    The AWC API returns intermittent 502/504s under load, so retry 5xx and transport errors
    (timeouts) with exponential backoff (2/4/8/16 s); 4xx and a final failure propagate."""
    for attempt in range(retries + 1):
        try:
            resp = httpx.get(
                TAF_API_URL,
                params={"ids": ",".join(ids), "date": date, "format": "xml"},
                headers={"User-Agent": USER_AGENT},
                timeout=90.0,
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 or attempt == retries:
                raise
        except httpx.TransportError:
            if attempt == retries:
                raise
        time.sleep(2 ** (attempt + 1))  # 2, 4, 8, 16 s backoff before the next attempt
    raise RuntimeError("unreachable")  # pragma: no cover — loop always returns or raises


def backfill_api(
    stations: list[str], *, days: int = API_MAX_DAYS, batch: int = API_BATCH, save: bool = True
) -> dict[str, object]:
    """Backfill the last `days` (capped at the API's ~30-day ceiling) of daily TAFs for `stations`.

    For each date in [today, today-days], call the API in station-batches; one call yields one TAF
    per station (the one valid that day). Responses are gzipped into the immutable raw zone
    (raw/awc/taf_api/date=YYYYMMDD/) so `parse_taf_cache`/`to_parquet` consume them like snapshots.
    Polite pacing keeps us well under the ~100/min guideline."""
    if days > API_MAX_DAYS:
        print(
            f"note: AWC API serves at most ~{API_MAX_DAYS} days; capping {days} -> {API_MAX_DAYS}"
        )
        days = API_MAX_DAYS
    end = datetime.now(UTC).date()
    dates = [(end - timedelta(days=i)).strftime("%Y%m%d") for i in range(days + 1)]
    batches = [stations[i : i + batch] for i in range(0, len(stations), batch)]
    n_calls = len(dates) * len(batches)
    print(
        f"backfill: {len(stations)} stations × {len(dates)} days = {n_calls} calls "
        f"({len(batches)} batches/day)"
    )
    seen_tafs = 0
    done = 0
    for date in dates:
        for bi, ids in enumerate(batches):
            out = AWC_TAF_API_RAW / f"date={date}" / f"batch{bi:02d}.xml.gz"
            if save and out.exists() and out.stat().st_size > 0:
                done += 1  # resume: this (date, batch) already fetched — skip (fills only gaps)
                continue
            try:
                raw = _fetch_api(ids, date)
            except Exception as e:  # noqa: BLE001 — one bad call must not abort a multi-hundred-call run
                print(f"  {date} batch {bi}: FETCH FAILED {type(e).__name__}: {e}")
                time.sleep(1.0)
                continue
            gz = gzip.compress(raw, compresslevel=9, mtime=0)  # API returns plain XML; we gzip
            n = len(parse_taf_cache(gz))
            seen_tafs += n
            if save:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(gz)
                record_artifact(
                    TAF_API_URL, out, gz, source="awc", product="taf", date=date, n_tafs=n
                )
            done += 1
            time.sleep(0.7)  # ~85/min — polite, under the 100/min guideline
        print(f"  {date}: {done}/{n_calls} calls done, {seen_tafs} TAF-rows fetched (pre-dedup)")
    print(f"backfill complete: {n_calls} calls, {seen_tafs} TAF-rows fetched (dedup at --parquet)")
    return {"calls": n_calls, "rows": seen_tafs}


# ── corpus assembly (dedup across snapshots + backfill) ──────────────────────────────────────
def _coverage(rows: list[dict[str, str]]) -> dict[str, object]:
    """Quick geo-coverage summary of a batch of TAFs (for --collect / --stats output)."""
    pref = Counter((r["station_id"] or "?")[0] for r in rows)
    us = sum(v for k, v in pref.items() if k in "KPT")
    return {
        "n": len(rows),
        "non_us": len(rows) - us,
        "non_us_pct": round(100 * (len(rows) - us) / len(rows)) if rows else 0,
        "by_region": dict(sorted(pref.items())),
    }


def _all_raw_files() -> list[Path]:
    """Every stored AWC TAF XML payload — snapshots AND api backfill — for corpus assembly."""
    return sorted(AWC_TAF_RAW.rglob("*.xml.gz")) + sorted(AWC_TAF_API_RAW.rglob("*.xml.gz"))


def to_parquet(out: Path | None = None) -> Path:
    """Rebuild the deduped global-TAF corpus parquet from ALL stored payloads (snapshot+backfill).

    Dedup by (station_id, issue_time, raw_text) across everything via a dict, then let DuckDB
    write the parquet from a temp NDJSON — no pandas/numpy dependency (that combo bit this venv)."""
    files = _all_raw_files()
    if not files:
        raise FileNotFoundError(
            f"no AWC TAF payloads under {RAW_DIR / 'awc'} — run --collect first"
        )
    dedup: dict[tuple[str, str, str], dict[str, str]] = {}
    for fp in files:
        for r in parse_taf_cache(fp.read_bytes()):
            key = (r["station_id"] or "", r["issue_time"] or "", r["raw_text"] or "")
            dedup.setdefault(key, r)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = out or (PROCESSED / "awc_taf_global.parquet")
    tmp = PROCESSED / "_awc_taf.ndjson"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in dedup.values()), encoding="utf-8")
    try:
        duckdb.connect().execute(
            f"COPY (SELECT * FROM read_json_auto('{tmp}')) TO '{out}' (FORMAT PARQUET)"
        )
    finally:
        tmp.unlink(missing_ok=True)
    print(f"deduped {len(files)} payloads -> {len(dedup)} unique TAFs -> {out}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="AWC one-time global TAF collector")
    ap.add_argument(
        "--collect", action="store_true", help="one current global snapshot -> raw zone"
    )
    ap.add_argument("--backfill", action="store_true", help="last ~30 days via the Data API")
    ap.add_argument(
        "--all", action="store_true", help="backfill US stations too (default: non-US only)"
    )
    ap.add_argument(
        "--days", type=int, default=API_MAX_DAYS, help=f"backfill depth (capped {API_MAX_DAYS})"
    )
    ap.add_argument("--parquet", action="store_true", help="rebuild deduped corpus parquet")
    ap.add_argument("--stats", action="store_true", help="coverage of the accumulated corpus")
    args = ap.parse_args()

    if args.collect:
        collect()
    elif args.backfill:
        stations = latest_snapshot_stations(non_us_only=not args.all)
        backfill_api(stations, days=args.days)
        to_parquet()
    elif args.parquet:
        to_parquet()
    elif args.stats:
        allrows = [r for fp in _all_raw_files() for r in parse_taf_cache(fp.read_bytes())]
        dedup = {(r["station_id"], r["issue_time"], r["raw_text"]): r for r in allrows}
        cov = _coverage(list(dedup.values()))
        print(f"unique TAFs: {cov['n']} ({cov['non_us_pct']}% non-US)")
        print(f"payloads: {len(_all_raw_files())} | by region: {cov['by_region']}")
    else:
        ap.error("one of --collect / --backfill / --parquet / --stats is required")
