"""AWC ingest — fetch current data from the Aviation Weather Center.

Phase 1. For now this provides the bulk METAR-cache fetch used by the scope probe;
the scheduled collector (`make collect`) will reuse the same fetch + manifest
helpers, just on a cron.

Two rules from the plan/briefings baked in here:
  - **Be a polite client** (AWC ToU): send a descriptive User-Agent; identify via
    the repo URL, not a personal email.
  - **Immutable raw zone + manifest.** Raw bytes are written verbatim and never
    edited; every fetch appends a checksummed row to data/manifest.json. That's
    what makes `make backfill` reproducible (Skill A).

Why the cache and not the JSON API here: the bulk cache returns the raw report
string *and* AWC's own decoded columns (temp_c, wind_*, altim_in_hg,
flight_category, an `auto` flag, …). That decoded data is our "bundled ground
truth" — a free 4th oracle voice — which is exactly what the scope probe compares
the Python parsers against.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

# Descriptive UA per AWC ToU; contact point is the repo, not a personal address.
USER_AGENT = "aviai-avtext/0.1 (+https://github.com/wdenejko/aviai)"
METAR_CACHE_URL = "https://aviationweather.gov/data/cache/metars.cache.csv.gz"

# data/ layout (see data/README.md). __file__ = .../projects/avtext/src/avtext/ingest/awc.py
PKG_ROOT = Path(__file__).resolve().parents[3]  # -> .../projects/avtext
DATA_DIR = PKG_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST = DATA_DIR / "manifest.json"


def _utcstamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def record_artifact(url: str, path: Path, raw: bytes, **extra: object) -> None:
    """Append a provenance row to the committed manifest ledger."""
    manifest = (
        json.loads(MANIFEST.read_text())
        if MANIFEST.exists()
        else {"schema_version": 1, "artifacts": []}
    )
    manifest.setdefault("artifacts", [])
    manifest["artifacts"].append(
        {
            "url": url,
            "path": str(path.relative_to(PKG_ROOT)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "fetched_at": _utcstamp(),
            **extra,
        }
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def fetch_metar_cache(*, save: bool = True) -> bytes:
    """Fetch the AWC bulk METAR cache (gzipped CSV); return the raw gzip bytes.

    With save=True, writes the bytes verbatim into the immutable raw zone and
    records provenance — the exact path the collector will also use.
    """
    resp = httpx.get(
        METAR_CACHE_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    )
    resp.raise_for_status()
    raw = resp.content
    if save:
        out = RAW_DIR / "awc" / "metars_cache" / f"dt={_utcstamp()}" / "metars.cache.csv.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
        record_artifact(METAR_CACHE_URL, out, raw, source="awc", product="metar")
    return raw


def parse_metar_cache(raw_gz: bytes) -> list[dict[str, str]]:
    """Decode the gzipped CSV into row dicts (only rows carrying a raw_text).

    The AWC header repeats sky_cover/cloud_base_ft_agl up to 4× (one per cloud
    layer); csv.DictReader collapses duplicate keys, which is fine — the probe
    only needs raw_text plus a few decoded scalars. We defensively skip any
    preamble before the real header row.
    """
    text = gzip.decompress(raw_gz).decode("utf-8", errors="replace")
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("raw_text,")), 0)
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return [row for row in reader if row.get("raw_text")]
