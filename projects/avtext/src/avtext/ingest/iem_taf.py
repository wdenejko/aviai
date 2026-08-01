"""IEM ingest — historical TAF backfill from the Iowa Environmental Mesonet.

TAF is "METAR 2" (see docs/research/aviation-data-sources.md): the same free bulk
history and the same auto-gradable answer key, so this module is a deliberate near-mirror
of ingest/iem.py — same immutable-raw + checksummed-manifest discipline, same parquet
normalization. The differences are all TAF-shaped and worth stating up front, because
they drive every design choice downstream:

  1. ENDPOINT + PARAMS DIFFER. taf.py takes ISO `sts`/`ets` (not asos.py's year1/month1…).
  2. ONE ROW PER FORECAST PERIOD, not per report. A single TAF bulletin expands into an
     `Observation` row (initial conditions) plus one row per FM / TEMPO / BECMG / PROB
     change group. So "how many TAFs" = COUNT(DISTINCT product_id), never the row count.
     The columns already carry IEM's *decoded* view (sknt/drct/gust, visibility in SM,
     skyc/skyl lists, ws_*), which is our first "voice" in the TAF consensus — for free.
  3. THE per-row `raw` IS THE CHANGE GROUP ONLY, header-stripped (e.g. `FM011500 …`, and
     TEMPO rows drop even the `TEMPO` keyword — it rides in `is_tempo`). The full raw
     bulletin (what a user actually pastes) — station + `DDHHMMZ` + `DDHH/DDHH` validity +
     all groups + `=` — is NOT cleanly reconstructable from these columns because the TAF
     *validity period* is not a per-row field. So the full raw is fetched canonically from
     `/api/1/nwstext/{product_id}` at eval/SFT build time (bounded subset), while THIS
     bulk parquet stores the cheap per-period decoded rows. (Verified live 2026-08-01.)
  4. FEWER STATIONS ISSUE TAFs than METARs. `--validate` here counts distinct TAFs per
     station over a recent window so we can filter our 526-station set before a full
     backfill — most small non-US fields in the geo-balanced v2 set won't have TAFs.

Endpoints (verified 2026-08-01):
  request:  cgi-bin/request/taf.py?station=..&sts=ISO&ets=ISO&format=comma
            -> CSV with columns:
               station,valid,fx_valid,raw,is_tempo,fx_valid_end,sknt,drct,gust,
               visibility,presentwx,skyc,skyl,ws_level,ws_drct,ws_sknt,
               product_id,ftype,is_amendment
  full raw: api/1/nwstext/{product_id}   (canonical bulletin; used later, not here)

CLI:
  uv run python -m avtext.ingest.iem_taf --validate      # which stations issue TAFs?
  uv run python -m avtext.ingest.iem_taf --days 30       # bounded test backfill
  uv run python -m avtext.ingest.iem_taf --years 3       # full backfill + parquet
  uv run python -m avtext.ingest.iem_taf --parquet       # rebuild parquet from raw
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

TAF_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py"
CONFIG_V1 = DATA_DIR.parent / "configs" / "stations.yaml"
CONFIG_V2 = DATA_DIR.parent / "configs" / "stations_v2.yaml"
IEM_TAF_RAW = RAW_DIR / "iem" / "taf"
PROCESSED = DATA_DIR / "processed" / "taf"


def _ids(config: Path) -> list[str]:
    """ICAO ids from a stations.yaml (either v1 or the geo-balanced v2 file)."""
    doc = yaml.safe_load(config.read_text())
    return [s["icao"] for s in doc["stations"]]


def all_station_ids() -> list[str]:
    """The deduped v1 ∪ v2 universe — every station we might want a TAF from.
    Order-preserving so a run is reproducible; v1 first, then new v2 picks."""
    seen: dict[str, None] = {}
    for cfg in (CONFIG_V1, CONFIG_V2):
        for s in _ids(cfg):
            seen.setdefault(s, None)
    return list(seen)


def fetch_station(station: str, start: date, end: date) -> bytes:
    """Fetch raw TAF CSV for one station over [start, end] from IEM.

    taf.py wants ISO timestamps (`sts`/`ets`), unlike asos.py's split year/month/day —
    the one API-shape difference from the METAR sibling."""
    params = {
        "station": station,
        "sts": f"{start.isoformat()}T00:00Z",
        "ets": f"{end.isoformat()}T00:00Z",
        "format": "comma",
    }
    resp = httpx.get(
        TAF_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=180.0,
        follow_redirects=True,
    )  # fmt: skip
    resp.raise_for_status()
    return resp.content


def _distinct_tafs(csv_bytes: bytes) -> int:
    """Count DISTINCT TAF bulletins (by product_id) — NOT rows. Rows are per-period, so a
    single 6-group TAF is 6 rows; the meaningful "does this station issue TAFs" signal is
    the bulletin count. product_id is the last-but-two column; parse by header index so we
    don't hard-code position if IEM reorders."""
    lines = csv_bytes.decode("utf-8", "replace").splitlines()
    if not lines or not lines[0].startswith("station,"):
        return 0
    header = lines[0].split(",")
    try:
        pid = header.index("product_id")
    except ValueError:
        return 0
    ids = set()
    for ln in lines[1:]:
        if not ln.strip():
            continue
        cols = ln.split(",")
        if len(cols) > pid:
            ids.add(cols[pid])
    return len(ids)


def validate_stations(stations: list[str], sample_days: int = 14) -> dict[str, int]:
    """Sample a recent window per station; {station: distinct_TAF_count} (0 => no TAFs).

    Wider window than the METAR validator (14 d vs 3 d): routine TAFs come only ~4×/day at
    issuing fields, and many stations amend irregularly, so a short window can false-0 a
    real TAF station. 0 here means "filter out before backfill"."""
    end = datetime.now(UTC).date()
    start = end - timedelta(days=sample_days)
    out: dict[str, int] = {}
    for i, s in enumerate(stations, 1):
        try:
            out[s] = _distinct_tafs(fetch_station(s, start, end))
        except Exception as e:  # noqa: BLE001 — validation: any error = drop candidate
            print(f"  [{i}/{len(stations)}] {s}: ERROR {type(e).__name__}: {e}")
            out[s] = -1
            time.sleep(0.5)
            continue
        time.sleep(0.3)  # polite to IEM
    return out


def backfill(stations: list[str], start: date, end: date, *, save: bool = True) -> dict[str, int]:
    """Fetch each station's full range -> immutable raw .csv.gz + manifest row.

    Identical discipline to iem.py.backfill: deterministic gzip (mtime=0) so re-running is
    idempotent, one manifest row per artifact, one bad station never aborts the run. `counts`
    is distinct-TAF, not rows, to stay comparable with --validate."""
    counts: dict[str, int] = {}
    for i, s in enumerate(stations, 1):
        try:
            raw = fetch_station(s, start, end)
        except Exception as e:  # noqa: BLE001 — one bad station must not abort a long run
            print(f"  [{i}/{len(stations)}] {s}: FETCH FAILED {type(e).__name__}: {e}")
            counts[s] = -1
            time.sleep(1.0)
            continue
        counts[s] = _distinct_tafs(raw)
        if save:
            gz = gzip.compress(raw, compresslevel=9, mtime=0)
            out = IEM_TAF_RAW / f"station={s}" / f"{start.isoformat()}__{end.isoformat()}.csv.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(gz)
            record_artifact(
                TAF_URL, out, gz, source="iem", product="taf", station=s, tafs=counts[s]
            )
        print(f"  [{i}/{len(stations)}] {s}: {counts[s]} TAFs")
        time.sleep(1.0)  # polite between stations
    return counts


def to_parquet(out: Path | None = None) -> Path:
    """Normalize every raw IEM TAF CSV -> one Parquet via DuckDB.

    Kept as all-varchar on purpose: the decoded columns skyc/skyl/presentwx are Python-
    literal *strings* (`['SCT', 'OVC']`, `[1500, 4000]`) that the oracle layer will
    literal_eval — normalizing types here would launder IEM's decode choices into the
    store before the consensus even runs. We keep the row grain (one period per row) and
    the product_id so a bulletin can be reassembled downstream."""
    files = [str(p) for p in sorted(IEM_TAF_RAW.rglob("*.csv.gz"))]
    if not files:
        raise FileNotFoundError(f"no raw IEM TAF files under {IEM_TAF_RAW} — run a backfill first")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = out or (PROCESSED / "iem_taf.parquet")
    files_sql = "[" + ", ".join(f"'{p}'" for p in files) + "]"
    duckdb.connect().execute(f"""
        COPY (
          SELECT * FROM read_csv({files_sql}, header=true, all_varchar=true, union_by_name=true)
          WHERE raw IS NOT NULL AND length(raw) > 0
        ) TO '{out}' (FORMAT PARQUET)
    """)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="IEM TAF backfill (mirror of ingest.iem)")
    ap.add_argument(
        "--validate", action="store_true", help="sample each station; which issue TAFs?"
    )
    ap.add_argument("--years", type=int, default=3, help="backfill this many years back")
    ap.add_argument("--days", type=int, default=0, help="if >0, backfill only the last N days")
    ap.add_argument("--parquet", action="store_true", help="(re)build parquet from existing raw")
    ap.add_argument(
        "--stations", default="union", choices=["v1", "v2", "union"],
        help="which station set: v1 (51), v2 (475), or their union (~526, default)",
    )  # fmt: skip
    ap.add_argument("--out", default=None, help="explicit path for a filtered TAF-station yaml")
    args = ap.parse_args()

    sids = {"v1": _ids(CONFIG_V1), "v2": _ids(CONFIG_V2), "union": all_station_ids()}[args.stations]

    if args.validate:
        counts = validate_stations(sids)
        ok = sorted(k for k, v in counts.items() if v > 0)
        none = sorted(k for k, v in counts.items() if v == 0)
        err = sorted(k for k, v in counts.items() if v < 0)
        print(f"\nTAF-issuing: {len(ok)}/{len(sids)}  (none: {len(none)}, errors: {len(err)})")
        total = sum(v for v in counts.values() if v > 0)
        print(f"total TAFs sampled (14 d): {total}")
        # Persist the filtered list so the backfill/eval use only TAF-issuing stations.
        out = Path(args.out) if args.out else (CONFIG_V1.parent / "stations_taf.yaml")
        body = "".join(f"- icao: {s}\n  taf_sample: {counts[s]}\n" for s in ok)
        out.write_text(
            "# TAF-issuing stations, validated live against IEM taf.py.\n"
            "# Generated by `python -m avtext.ingest.iem_taf --validate`.\n"
            "# taf_sample = distinct TAFs in a 14-day window.\n"
            f"# {len(ok)} of {len(sids)} {args.stations} stations issue TAFs.\n"
            "stations:\n" + body
        )
        print(f"wrote {out}")
    elif args.parquet:
        print("wrote", to_parquet())
    else:
        end = datetime.now(UTC).date()
        start = end - timedelta(days=args.days if args.days else 365 * args.years)
        print(f"backfill {len(sids)} stations {start} -> {end}")
        backfill(sids, start, end)
        print("parquet:", to_parquet())
