"""IEM ingest — historical METAR backfill from the Iowa Environmental Mesonet.

Phase 1 (Skill A). IEM's ASOS archive is the best open bulk source of raw METAR
history: global, deep, free, one request per station over any date range. Same
discipline as ingest/awc.py — immutable raw zone + checksummed manifest — then
normalize to Parquet for DuckDB.

Endpoints (verified 2026-07-29):
  request:  cgi-bin/request/asos.py?station=..&data=metar&<range>&format=onlycomma
            -> CSV "station,valid,metar"  (raw report; METAR/SPECI keyword stripped,
            unlike the AWC feed — the schema/oracle layer must tolerate both)
  network:  geojson/network/<NET>.geojson  (station metadata; not needed to backfill)

CLI:
  uv run python -m avtext.ingest.iem --validate       # which stations return data?
  uv run python -m avtext.ingest.iem --days 30        # bounded test backfill
  uv run python -m avtext.ingest.iem --years 3        # full backfill + parquet
  uv run python -m avtext.ingest.iem --parquet        # rebuild parquet from raw
"""

from __future__ import annotations

import gzip
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import httpx
import yaml

from avtext.ingest.awc import DATA_DIR, RAW_DIR, USER_AGENT, record_artifact

ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
CONFIG = DATA_DIR.parent / "configs" / "stations.yaml"
IEM_RAW = RAW_DIR / "iem" / "metar"
PROCESSED = DATA_DIR / "processed" / "metar"


def load_station_ids() -> list[str]:
    """ICAO ids from configs/stations.yaml."""
    return [s["icao"] for s in yaml.safe_load(CONFIG.read_text())["stations"]]


def fetch_station(station: str, start: date, end: date) -> bytes:
    """Fetch raw METAR CSV for one station over [start, end] from IEM."""
    params = {
        "station": station, "data": "metar", "tz": "Etc/UTC", "format": "onlycomma",
        "latlon": "no", "elev": "no", "missing": "M", "trace": "T",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
    }  # fmt: skip
    resp = httpx.get(
        ASOS_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=180.0,
        follow_redirects=True,
    )  # fmt: skip
    resp.raise_for_status()
    return resp.content


def _row_count(csv_bytes: bytes) -> int:
    """Data rows in an IEM CSV (line 1 is the header)."""
    lines = csv_bytes.decode("utf-8", "replace").splitlines()
    return len([ln for ln in lines[1:] if ln.strip()])


def validate_stations(stations: list[str], sample_days: int = 3) -> dict[str, int]:
    """Sample a recent window per station; {station: row_count} (0/-1 => drop)."""
    end = datetime.now(UTC).date()
    start = end - timedelta(days=sample_days)
    out: dict[str, int] = {}
    for s in stations:
        try:
            out[s] = _row_count(fetch_station(s, start, end))
        except Exception as e:  # noqa: BLE001 — validation: any error = drop candidate
            print(f"  {s}: ERROR {type(e).__name__}: {e}")
            out[s] = -1
        time.sleep(0.4)  # polite to IEM
    return out


def backfill(stations: list[str], start: date, end: date, *, save: bool = True) -> dict[str, int]:
    """Fetch each station's full range -> immutable raw .csv.gz + manifest row."""
    counts: dict[str, int] = {}
    for i, s in enumerate(stations, 1):
        try:
            raw = fetch_station(s, start, end)
        except Exception as e:  # noqa: BLE001 — one bad station must not abort a long run
            print(f"  [{i}/{len(stations)}] {s}: FETCH FAILED {type(e).__name__}: {e}")
            counts[s] = -1
            time.sleep(1.0)
            continue
        counts[s] = _row_count(raw)
        if save:
            # mtime=0 -> deterministic bytes, so re-running backfill is idempotent
            gz = gzip.compress(raw, compresslevel=9, mtime=0)
            out = IEM_RAW / f"station={s}" / f"{start.isoformat()}__{end.isoformat()}.csv.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(gz)
            record_artifact(
                ASOS_URL, out, gz, source="iem", product="metar", station=s, rows=counts[s]
            )
        print(f"  [{i}/{len(stations)}] {s}: {counts[s]} obs")
        time.sleep(1.0)  # polite between stations
    return counts


def to_parquet(out: Path | None = None) -> Path:
    """Normalize every raw IEM CSV -> one Parquet (station, valid_utc, raw) via DuckDB."""
    files = [str(p) for p in sorted(IEM_RAW.rglob("*.csv.gz"))]
    if not files:
        raise FileNotFoundError(f"no raw IEM files under {IEM_RAW} — run a backfill first")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = out or (PROCESSED / "iem_metar.parquet")
    files_sql = "[" + ", ".join(f"'{p}'" for p in files) + "]"
    duckdb.connect().execute(f"""
        COPY (
          SELECT station,
                 CAST(valid AS TIMESTAMP) AS valid_utc,
                 metar AS raw
          FROM read_csv({files_sql}, header=true, all_varchar=true)
          WHERE metar IS NOT NULL AND length(metar) > 0 AND metar <> 'M'
        ) TO '{out}' (FORMAT PARQUET)
    """)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="IEM METAR backfill")
    ap.add_argument("--validate", action="store_true", help="sample each station; report coverage")
    ap.add_argument("--years", type=int, default=3, help="backfill this many years back")
    ap.add_argument("--days", type=int, default=0, help="if >0, backfill only the last N days")
    ap.add_argument("--parquet", action="store_true", help="(re)build parquet from existing raw")
    args = ap.parse_args()

    sids = load_station_ids()
    if args.validate:
        counts = validate_stations(sids)
        ok = sorted(k for k, v in counts.items() if v > 0)
        bad = sorted(k for k, v in counts.items() if v <= 0)
        print(f"\nvalid: {len(ok)}/{len(sids)}")
        if bad:
            print("NO DATA (drop or fix):", ", ".join(bad))
    elif args.parquet:
        print("wrote", to_parquet())
    else:
        end = datetime.now(UTC).date()
        start = end - timedelta(days=args.days if args.days else 365 * args.years)
        print(f"backfill {len(sids)} stations {start} -> {end}")
        backfill(sids, start, end)
        print("parquet:", to_parquet())
