"""Ingest one month of METAR for the hub airports from the IEM ASOS archive into aviation.metar,
and a tiny aviation.airports dimension (IATA<->ICAO) so flights (IATA codes) can join weather
(ICAO station ids).

Reuses avtext's IEM approach (mesonet ASOS request endpoint -> 'station,valid,metar' CSV),
replicated here to keep the sandbox self-contained. Underlying obs are US NWS public domain.

  uv run --package dsbench python projects/dsbench/sandbox/ingest/metar.py --year 2026 --month 6
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
ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# hub set (ADR-003 §4): (IATA as flights use it, ICAO as METAR uses it, name)
HUBS = [
    ("ATL", "KATL", "Hartsfield-Jackson Atlanta Intl"),
    ("ORD", "KORD", "Chicago O'Hare Intl"),
    ("DFW", "KDFW", "Dallas/Fort Worth Intl"),
    ("DEN", "KDEN", "Denver Intl"),
    ("LAX", "KLAX", "Los Angeles Intl"),
]


def _fetch(station: str, start: date, end: date, *, attempts: int = 5) -> str:
    params = {
        "station": station, "data": "metar", "tz": "Etc/UTC", "format": "onlycomma",
        "latlon": "no", "elev": "no", "missing": "M", "trace": "T",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
    }  # fmt: skip
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = httpx.get(ASOS, params=params, headers={"User-Agent": UA}, timeout=180.0,
                          follow_redirects=True)
            r.raise_for_status()
            return r.text
        except (httpx.HTTPStatusError, httpx.TransportError) as e:  # transient IEM 5xx / timeout
            last = e
            wait = 3 * (i + 1)
            print(f"    {station}: {type(e).__name__} attempt {i + 1}/{attempts}, retry {wait}s")
            time.sleep(wait)
    raise last  # type: ignore[misc]


def fetch_hub(icao: str, start: date, end: date) -> str:
    # IEM's US ASOS id may be the 3-letter local (ATL) or 4-letter ICAO (KATL); try both.
    for sid in ([icao, icao[1:]] if icao.startswith("K") and len(icao) == 4 else [icao]):
        text = _fetch(sid, start, end)
        if [ln for ln in text.splitlines()[1:] if ln.strip()]:
            return text
        time.sleep(0.5)
    return text  # last attempt (possibly header-only)


def to_frame(icao: str, text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df = df.rename(columns={"valid": "valid_utc", "metar": "raw"})
    df = df[df["raw"].notna() & (df["raw"].astype(str).str.len() > 0) & (df["raw"] != "M")]
    df["station"] = icao  # normalize to the ICAO we asked for, regardless of IEM's internal id
    df["valid_utc"] = pd.to_datetime(df["valid_utc"], errors="coerce")
    df = df.dropna(subset=["valid_utc"])
    return df[["station", "valid_utc", "raw"]]


def load(year: int, month: int) -> dict:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)  # exclusive upper

    client = get_client()
    client.command(
        "CREATE TABLE IF NOT EXISTS aviation.metar (station LowCardinality(String), "
        "valid_utc DateTime, raw String) ENGINE = MergeTree ORDER BY (station, valid_utc)"
    )
    client.command(
        "CREATE TABLE IF NOT EXISTS aviation.airports (iata String, icao String, name String) "
        "ENGINE = TinyLog"
    )
    client.command("TRUNCATE TABLE aviation.metar")
    client.command("TRUNCATE TABLE aviation.airports")
    client.insert("aviation.airports", [list(h) for h in HUBS],
                  column_names=["iata", "icao", "name"])

    counts = {}
    for iata, icao, _ in HUBS:
        text = fetch_hub(icao, start, end)
        df = to_frame(icao, text)
        if len(df):
            client.insert_df("aviation.metar", df)
        counts[icao] = int(len(df))
        print(f"  {icao} ({iata}): {counts[icao]} obs")
        time.sleep(1.0)  # polite to IEM

    loaded = client.command("SELECT count() FROM aviation.metar")
    manifest = {
        "source": "IEM ASOS archive (METAR)", "endpoint": ASOS, "year": year, "month": month,
        "stations": {icao: counts[icao] for _, icao, _ in HUBS},
        "rows_loaded": int(loaded),
        "license": "US NWS public domain (IEM: polite use + attribution)",
        "loaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    (MANIFESTS / f"metar_{year}_{month:02d}.json").write_text(json.dumps(manifest, indent=2))
    print(f"  loaded {loaded:,} METARs; {len(HUBS)} airports -> aviation.airports")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load one month of hub METARs into ClickHouse")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    args = ap.parse_args()
    load(args.year, args.month)
