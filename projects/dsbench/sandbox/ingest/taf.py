"""Ingest one month of TAF for the hub airports from the IEM archive into aviation.taf.

Reuses avtext's IEM TAF approach (mesonet taf.py endpoint). Unlike METAR, a TAF bulletin expands to
ONE ROW PER FORECAST PERIOD (initial Observation + each FM/TEMPO/BECMG group), and IEM returns the
DECODED fields (wind, visibility, sky) for free — so aviation.taf is already query-friendly. A
bulletin is identified by product_id; "how many TAFs" = count(DISTINCT product_id), not row count.

  uv run --package dsbench python projects/dsbench/sandbox/ingest/taf.py --year 2026 --month 6
"""
from __future__ import annotations

import argparse
import io
import json
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
from _ch import get_client

SANDBOX = Path(__file__).resolve().parents[1]
MANIFESTS = SANDBOX / "manifests"
UA = "aviai-dsbench/0.1 (research; contact wojciech.denejko@gmail.com)"
TAF_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py"
HUBS = ["KATL", "KORD", "KDFW", "KDEN", "KLAX"]  # major hubs all issue TAFs

DDL = """CREATE TABLE IF NOT EXISTS aviation.taf (
    station LowCardinality(String),
    issued DateTime,                 -- bulletin issue time (IEM 'valid')
    fx_from DateTime,                -- forecast period start ('fx_valid')
    fx_to Nullable(DateTime),        -- period end; null on the initial Observation row
    ftype LowCardinality(String),    -- Observation (initial) | Forecast (FM/TEMPO/BECMG group)
    is_tempo UInt8,
    is_amendment UInt8,
    raw String,                      -- the change-group text (header-stripped, per IEM)
    sknt Nullable(Float32),          -- wind speed (kt)
    drct Nullable(Float32),          -- wind direction (deg)
    gust Nullable(Float32),
    visibility Nullable(Float32),    -- statute miles
    presentwx String,
    skyc String,                     -- e.g. "['BKN']" (IEM's decoded list, kept as text)
    skyl String,                     -- e.g. "[10000]"
    product_id String                -- the TAF bulletin id
) ENGINE = MergeTree ORDER BY (station, issued, fx_from)"""

_TABLE_COLS = ["station", "issued", "fx_from", "fx_to", "ftype", "is_tempo", "is_amendment",
               "raw", "sknt", "drct", "gust", "visibility", "presentwx", "skyc", "skyl",
               "product_id"]


def _fetch(station: str, start: date, end: date, *, attempts: int = 5) -> str:
    params = {"station": station, "sts": f"{start.isoformat()}T00:00Z",
              "ets": f"{end.isoformat()}T00:00Z", "format": "comma"}
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = httpx.get(TAF_URL, params=params, headers={"User-Agent": UA}, timeout=180.0,
                          follow_redirects=True)
            r.raise_for_status()
            return r.text
        except (httpx.HTTPStatusError, httpx.TransportError) as e:  # transient IEM 5xx / timeout
            last = e
            wait = 3 * (i + 1)
            print(f"    {station}: {type(e).__name__} attempt {i + 1}/{attempts}, retry {wait}s")
            time.sleep(wait)
    raise last  # type: ignore[misc]


def to_frame(icao: str, text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return df
    out = pd.DataFrame({
        "station": icao,
        "issued": pd.to_datetime(df["valid"], errors="coerce"),
        "fx_from": pd.to_datetime(df["fx_valid"], errors="coerce"),
        "fx_to": pd.to_datetime(df["fx_valid_end"], errors="coerce"),  # empty -> NaT -> NULL
        "ftype": df["ftype"].fillna("").astype(str),
        "is_tempo": (df["is_tempo"].astype(str) == "True").astype("uint8"),
        "is_amendment": (df["is_amendment"].astype(str) == "True").astype("uint8"),
        "raw": df["raw"].fillna("").astype(str),
        "sknt": pd.to_numeric(df["sknt"], errors="coerce"),
        "drct": pd.to_numeric(df["drct"], errors="coerce"),
        "gust": pd.to_numeric(df["gust"], errors="coerce"),
        "visibility": pd.to_numeric(df["visibility"], errors="coerce"),
        "presentwx": df["presentwx"].fillna("").astype(str),
        "skyc": df["skyc"].fillna("").astype(str),
        "skyl": df["skyl"].fillna("").astype(str),
        "product_id": df["product_id"].fillna("").astype(str),
    })
    return out.dropna(subset=["issued", "fx_from"])[_TABLE_COLS]


def load(year: int, month: int) -> dict:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)

    client = get_client()
    client.command(DDL)
    client.command("TRUNCATE TABLE aviation.taf")

    counts, bulletins = {}, {}
    for icao in HUBS:
        df = to_frame(icao, _fetch(icao, start, end))
        if len(df):
            client.insert_df("aviation.taf", df)
        counts[icao] = int(len(df))
        bulletins[icao] = int(df["product_id"].nunique()) if len(df) else 0
        print(f"  {icao}: {counts[icao]} periods across {bulletins[icao]} bulletins")
        time.sleep(1.0)  # polite to IEM

    loaded = client.command("SELECT count() FROM aviation.taf")
    manifest = {
        "source": "IEM TAF archive", "endpoint": TAF_URL, "year": year, "month": month,
        "periods_per_station": counts, "bulletins_per_station": bulletins,
        "rows_loaded": int(loaded),
        "license": "US NWS public domain (IEM: polite use + attribution)",
        "note": "one row per forecast period; count(DISTINCT product_id) = number of TAF bulletins",
        "loaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    (MANIFESTS / f"taf_{year}_{month:02d}.json").write_text(json.dumps(manifest, indent=2))
    print(f"  loaded {loaded:,} TAF periods -> aviation.taf")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load one month of hub TAFs into ClickHouse")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    args = ap.parse_args()
    load(args.year, args.month)
