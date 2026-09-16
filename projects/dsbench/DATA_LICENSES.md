# Data licenses & redistribution rules (dsbench v2 sandbox)

**This file is part of code review.** Before any source is added to `sandbox/ingest/` or any
artifact is committed/uploaded, its row must exist here with a redistribution decision. The sandbox
loads real aviation data into a **local** ClickHouse; what lands in git is only the **loader scripts**
and the **checksummed manifests** (`sandbox/manifests/*.json`) — **never the bulk records**. Bulk
downloads live in `sandbox/.data/` (gitignored) and the ClickHouse volume, both local-only.

Legend for **Redistribute?**: ✅ yes · ⚠️ derived/with-attribution only (commit manifests, not bulk)
· ⛔ never (eval-only quarantine).

## Warehouse sources (loaded into `aviation.*`)

| Source | Table(s) | License / terms | Redistribute? | Notes |
|---|---|---|---|---|
| **BTS Reporting Carrier On-Time Performance** (PREZIP, June 2026) | `flights` (607,577) | US-gov **public domain** | ✅ | Direct keyless GET; `bts.py`. Manifest pins the month + sha256 of the zip. Records are public domain but we still commit only the manifest to keep the repo lean. |
| **IEM ASOS archive — METAR** (June 2026, 5 hubs) | `metar` (45,796), `airports` (5) | Underlying obs US NWS **public domain**; IEM asks for polite use + attribution | ⚠️ | `metar.py` sets a descriptive User-Agent + retry/backoff. Commit manifest only. `airports` is a tiny derived IATA↔ICAO dimension (safe). |
| **IEM TAF archive** (June 2026, 5 hubs, decoded per period) | `taf` (9,060 periods / 1,618 bulletins) | Underlying obs US NWS **public domain**; IEM polite use + attribution | ⚠️ | `taf.py`. Commit manifest only. |
| **DEEL-AI/NOTAM** (HuggingFace, 13-class) | `notam` (8,478) | **MIT** per the HF dataset card (publishable, with attribution) | ✅ | `notam.py` reads the CSVs from a local mirror under `avtext/data/third_party/deel_notam/`. The MIT license — not the directory — governs it; even so we commit only the manifest + loader, not the text records. Static ~2024 corpus, not date-aligned (see caveat in the manifest). |

## Rules

- **Only manifests + loaders are committed.** No source's raw rows (CSV, JSON, text) enter git. If a
  future problem needs a *derived* artifact committed (e.g. a tiny frozen fixture), it gets its own
  row here first with an explicit decision.
- **`avtext/data/third_party/` quarantine still applies to avtext's own sets.** `Knots`, `OpenNOTAM`
  and `ERAU` there are eval-only and must never be committed or trained on (see
  `projects/avtext/DATA_LICENSES.md`). DEEL-AI/NOTAM is a *separate*, MIT-licensed dataset that
  merely happens to be cached alongside them; do not conflate it with the quarantined sets.
- **If a source's license is unverified, it does not ship.** Every row above has a decided basis; add
  new sources as ⚠️/⛔ until their terms are confirmed.
