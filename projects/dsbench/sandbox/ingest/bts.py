"""Ingest one month of BTS 'Reporting Carrier On-Time Performance' into aviation.flights.

Collection (ADR-003 Appendix A): a keyless GET of the PREZIP monthly file (US-gov public domain,
so publishable). Downloads to sandbox/.data/raw/bts/ (gitignored), loads the operational subset
into ClickHouse, and writes a committed manifest (url + sha256 + rowcount) so the snapshot is
pinned and reproducible.

  uv run --package dsbench python projects/dsbench/sandbox/ingest/bts.py --year 2026 --month 6
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
from _ch import get_client

SANDBOX = Path(__file__).resolve().parents[1]
RAW = SANDBOX / ".data" / "raw" / "bts"
MANIFESTS = SANDBOX / "manifests"
UA = "aviai-dsbench/0.1 (research; contact wojciech.denejko@gmail.com)"
PREZIP = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"
)

# Column order MUST match clickhouse/init/01_flights.sql.
COLS = [
    "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "TaxiOut", "WheelsOff", "WheelsOn",
    "TaxiIn", "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes", "Cancelled",
    "CancellationCode", "Diverted", "AirTime", "Distance", "CarrierDelay", "WeatherDelay",
    "NASDelay", "SecurityDelay", "LateAircraftDelay",
]  # fmt: skip
STR_COLS = [
    "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime",
    "DepTime", "WheelsOff", "WheelsOn", "CRSArrTime", "ArrTime", "CancellationCode",
]  # fmt: skip
FLOAT_COLS = [
    "DepDelay", "DepDelayMinutes", "TaxiOut", "TaxiIn", "ArrDelay", "ArrDelayMinutes", "AirTime",
    "Distance", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]  # fmt: skip
U8_COLS = ["Cancelled", "Diverted"]


def download(year: int, month: int) -> tuple[bytes, str]:
    url = PREZIP.format(y=year, m=month)
    # BTS historically needs relaxed TLS (see ADR-003 Appendix A) + a descriptive UA.
    with httpx.Client(verify=False, headers={"User-Agent": UA}, timeout=180.0,
                      follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
    return r.content, url


def to_frame(zip_bytes: bytes) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
    want = set(COLS)
    with zf.open(csv_name) as f:
        df = pd.read_csv(f, usecols=lambda c: c in want, dtype=str, low_memory=False)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.date
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # NaN -> NULL in Nullable(Float32)
    for c in U8_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("uint8")
    for c in STR_COLS:
        df[c] = df[c].fillna("").astype(str)
    return df[COLS], csv_name


def load(year: int, month: int) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    print(f"downloading BTS {year}-{month:02d} ...")
    zip_bytes, url = download(year, month)
    sha = hashlib.sha256(zip_bytes).hexdigest()
    (RAW / f"{year}_{month}.zip").write_bytes(zip_bytes)
    print(f"  {len(zip_bytes):,} bytes  sha256={sha[:16]}...")

    df, csv_name = to_frame(zip_bytes)
    print(f"  parsed {len(df):,} rows x {len(df.columns)} cols from {csv_name}")

    client = get_client()
    client.command("TRUNCATE TABLE IF EXISTS aviation.flights")  # single-month load = idempotent
    # chunk the insert to keep memory flat
    n = len(df)
    for i in range(0, n, 100_000):
        client.insert_df("aviation.flights", df.iloc[i : i + 100_000])
        print(f"  inserted {min(i + 100_000, n):,}/{n:,}")
    loaded = client.command("SELECT count() FROM aviation.flights")

    manifest = {
        "source": "BTS Reporting Carrier On-Time Performance",
        "url": url, "year": year, "month": month, "csv": csv_name,
        "sha256": sha, "zip_bytes": len(zip_bytes), "rows_parsed": int(len(df)),
        "rows_loaded": int(loaded), "license": "US government public domain",
        "loaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    (MANIFESTS / f"bts_{year}_{month:02d}.json").write_text(json.dumps(manifest, indent=2))
    print(f"  loaded {loaded:,} rows -> aviation.flights")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load one month of BTS flights into ClickHouse")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    args = ap.parse_args()
    load(args.year, args.month)
