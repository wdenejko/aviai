# Data sources for TAF, SIGMET & NOTAM — sourcing study

**Researched & verified live: 2026-07-30** (three parallel research sweeps, endpoints fetched, not
recalled). Purpose: the METAR decode study works because we get **(1) free bulk history + a reusable
per-station ingest** and **(2) a free, auto-gradable decoded reference** (parser consensus + NOAA gold).
Before extending to TAF/SIGMET/NOTAM — the stated goal is *all four at "METAR level"* — we need to know
whether each product hands us those same two things. This document records the answer.

> **The one idea that governs everything below:** the products get *more valuable for an LLM* exactly as
> they get *harder to source and harder to grade*. TAF is a solved parser problem (like METAR); NOTAM's
> free-text is where an LLM could genuinely beat a parser — which is the same reason you can't cheaply
> auto-grade it. The answer key, not the modeling, is the binding constraint.

## Summary — the two axes that decide "can it reach METAR level"

| Product | Free bulk history + pipeline reuse | Auto-gradable answer key | The part that resists | Verdict |
|---|---|---|---|---|
| **METAR** *(done)* | IEM `asos.py`, deep | 3-parser consensus + NOAA gold | — | baseline |
| **TAF** | **IEM `taf.py`, to 1996** — reuse `iem.py` near-verbatim | **4 voices**: IEM + AWC decoded, `avwx`, `mivek` | change-group nesting (harder whole-record EM) | **Easy** |
| **SIGMET / AIRMET** | IEM GIS/text archive, to 2005 — mirror `iem.py` | AWC decoded JSON (hazard/alt/movement/**polygon**) — *current-only* | free-text geographic **extent** (grade by polygon IoU or human) | **Medium** |
| **NOTAM** | ✗ **no free bulk archive** (live API, ~22K academic sets, or paid) | Q-line: parsers ✓ · E-field: ✗ | free-text **E-field** — *but expert gold already exists* | **Hard sourcing, gold pre-built** |

---

## 1. TAF — the clean win

Essentially "METAR 2": same pipeline, same free answer key, deep history.

- **History (backfill):** `https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py` — the exact sibling of
  our METAR `asos.py`. Params `station, sts, ets` (ISO `2024-01-15T00:00Z`), `format=comma`. Archive to
  **1996-01-01**, global, per-station date-range. Reuse `iem.py`'s `backfill()`/manifest/parquet verbatim.
- **Decoded reference, bundled:** `taf.py` returns one **row per forecast period**, already split into
  change groups with decoded fields (`ftype` ∈ Observation/FM/TEMPO/BECMG/PROB30/40; `sknt/drct/gust`,
  `visibility`, `skyc/skyl` as lists, wind-shear `ws_*`). Full raw bulletin via
  `https://mesonet.agron.iastate.edu/api/1/nwstext/{product_id}`.
- **Live collector + 2nd oracle:** AWC `https://aviationweather.gov/api/data/taf?ids=KORD&format=json` —
  `rawTAF` + decoded `fcsts[]` (fcstChange, probability, wdir/wspd/wgst, visib, clouds[], wxString…).
  Current + 15 days only; 100 req/min; no key. Bulk-current twin: `.../data/cache/tafs.cache.csv.gz`.
- **Answer key = 4 voices:** IEM-decoded + AWC-decoded + `avwx` (`avwx/current/taf.py`) + `mivek` (strong
  TAF support). `python-metar` is METAR-only → drops out of the TAF consensus (2 code parsers + 2 bundled
  decodes; still a real ladder).
- **Licensing:** US Gov / NWS **public domain** (IEM & AWC). Polite-use only (descriptive User-Agent,
  ≤100/min) — already honored by our `awc.py`/`iem.py`.
- **Integration effort: LOW.** Swap `ASOS_URL`→`taf.py`, adjust the ISO date params, handle
  multi-row-per-TAF (+ `literal_eval` the cloud lists), extend `schema/` + `oracles/{avwx,mivek}.py` for TAF.
- **Caveat:** far fewer sites issue TAFs (~600–700 US) than METARs — run `--validate` against `taf.py` to
  filter our station list before the full backfill.

## 2. SIGMET / AIRMET — the medium build

Structured half auto-grades for free; the free-text extent is the part that needs work.

- **History (backfill):** IEM GIS + text archive, mirrors `iem.py`:
  - SIGMET geometry + raw text: `https://mesonet.agron.iastate.edu/cgi-bin/request/gis/sigmets.py`
    (`sts/ets`, `format=csv|shp|kml`) — cols `NAME,LABEL,TYPE,ISSUE,EXPIRE,PROD_ID,TEXT`; **to 2005**
    (raw text added 2025-05, backfilled).
  - G-AIRMET: `.../cgi-bin/request/gis/awc_gairmets.py` — **to 2021-03-03**.
  - Raw text by AFOS PIL (direct `asos.py` analog): `.../cgi-bin/afos/retrieve.py?pil=<PIL>&sdate=&edate=`
    — PILs `SIGE/SIGC/SIGW` (convective), `WS*`/`WSNT*` (non-convective + international), AIRMET
    Sierra/Tango/Zulu, `GMT` (G-AIRMET). Single product: `/api/1/nwstext/{PROD_ID}`.
- **Decoded reference (current only):** AWC `.../api/data/airsigmet` (US) and `.../api/data/isigmet`
  (international, **truly global**, 100+ live records) → decoded JSON: hazard, `airSigmetType`, valid
  from/to, altitude base/top, movement dir/speed, severity, **polygon `coords[]`**. Also `format=iwxxm`
  (ICAO XML). Bulk-current: `.../data/cache/airsigmets.cache.csv.gz`. ~400 rows/req, ~15-day history,
  100/min. **Not a deep decoded archive** — decoded history accrues only via a forward collector, or you
  re-decode the raw archive yourself (AWC's decoder is not retroactive).
- **Answer-key split:**
  - *Auto-gradable now:* FIR, series/label, hazard type, SIGMET/AIRMET/OUTLOOK, validity, altitude,
    movement, severity, and the polygon `coords[]`. WMO/AFOS header parses reliably.
  - *Needs human/IoU:* the free-text extent as phrased ("FROM 90NW GRB TO 30SE…", VOR radials) and any
    record where AWC's decoder abstains (`coords` null). Grade extent by **polygon IoU vs AWC coords**
    (strong but fallible — treat like the mutant audit, not gold) or human labels.
  - No dominant Python raw-text→structured SIGMET parser exists. Use AWC JSON/IWXXM as primary reference +
    `@opengeoweb/sigmet-airmet` (npm, KNMI, TS) as a second independent decode.
- **Licensing:** public domain (NWS); IEM open.
- **Volume:** **1–2 orders of magnitude below METAR** — episodic. Convective SIGMET hourly (E/C/W),
  AIRMET every 6 h; rough order 10k–150k issuances/yr *(agent estimate, confirm from IEM PIL counts)*.
- **Integration:** new `ingest/iem_sigmet.py` mirroring `iem.py`; extend `awc.py` for the decoded cache;
  new SIGMET schema with a raw+parsed split for the extent field.

## 3. NOTAM — hard to source, but the gold already exists

This is the frontier (parsers fail the free-text E-field) and the surprise: the expert-labeled gold that
would have been the blocker is **already published, Apache-2.0**.

- **No free bulk historical archive** (unlike IEM METAR). Confirmed dead-ends: FAA API is active-only;
  DINS/EAD are real-time + anti-scraping; FAA NOTAM Search is per-location lookup, no bulk export.
- **Live raw stream:** FAA NOTAM API `https://external-api.faa.gov/notamapi/v1` — free self-registration
  (`client_id`/`client_secret`), US + international (FNS), formats `icao` (raw text) / `aixm` / `geoJson`,
  **active-only** (collect forward for fresh volume). *(Since 2021–24 the FAA moved US NOTAMs to ICAO
  format, so modern ones carry a parseable Q-line.)*
- **The realistic free historical path — two academic datasets (Apache-2.0, 2024 NOTAMs, raw + expert
  gold; same lab, Beihang/BUAA + ADCC):**
  - **Knots** — `https://github.com/Estrellajer/Knots` (arXiv 2511.12630): **12,347** expert-annotated
    NOTAMs, 194 FIRs, split by Q-code category, `raw_text` + `manual_fields` structured gold.
  - **NOTAM-Evolve** — `https://github.com/Estrellajer/NOTAM-Evolve` (AAAI 2026, arXiv 2511.07982):
    **9,993** rows, instruction format `{raw E-field → structured JSON}` + chain-of-thought, α = 0.96.
  - *(AirsideLabs/NOTAM on HF is a small RAG/QA set from guidance docs — marginal.)*
- **Answer-key split:**
  - *Q-line = auto-gradable* (structured: FIR, Q-code, traffic/purpose/scope, limits, coords/radius) via
    open-source parsers — **PyNotam** (`https://github.com/slavak/PyNotam`, Python, best fit for our
    oracle-adapter pattern), dbrgn/notam-parse, svoop/notam.
  - *E-field = needs expert gold* — no deterministic decoder; **Knots + NOTAM-Evolve provide exactly this
    gold**, so the human-labeling investment is largely pre-paid.
- **Licensing:** FAA API free/public-domain; Knots/NOTAM-Evolve **Apache-2.0** (attribution); deep paid
  history via **ICAO Stored NOTAMs** (~USD 550 / 2K calls) or commercial (Notamify, etc.).
- **Caveats:** the academic sets are **2024-only, ~22K, Asia-skewed, ADCC/China-annotated**; no free
  multi-year archive; live FAA API is active-only.

---

## Recommended sequencing

1. **TAF — next, right after the v2 METAR run.** Same pipeline, 4-voice auto-grading, deep public-domain
   history. Change-group nesting makes whole-record EM a *better* size discriminator than METAR. Low
   effort, high certainty. (Filter to TAF-issuing stations first.)
2. **SIGMET — the medium build.** Mirror `iem.py` for the 2005+ archive; extend `awc.py` for the decoded
   cache. Structured fields grade for free; grade the free-text extent by polygon IoU (or defer). Completes
   the auto-gradable weather trio; low volume.
3. **NOTAM — the flagship, now tractable.** Start from Knots + NOTAM-Evolve (git clone, Apache-2.0) as gold
   eval + SFT; PyNotam for the Q-line; FAA API for a live raw stream. Here "METAR level" means something
   *stronger* — not "match a parser" (parsers fail the E-field) but **"match expert humans on what parsers
   can't do."** The one place the LLM's value is categorical, not fidelity-to-a-parser.

## Licensing for an eventual combined publish

- **TAF / SIGMET:** public domain (US Gov / NWS) — freely publishable.
- **NOTAM academic sets:** Apache-2.0 (attribution).
- **Model weights:** carry the **Gemma Terms of Use** regardless (see the publishing discussion) — the
  adapter can't be relicensed; the harness/eval code can be MIT/Apache.

## Appendix — verified endpoints (2026-07-30)

```
TAF     IEM backfill   https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py
        IEM raw text   https://mesonet.agron.iastate.edu/api/1/nwstext/{product_id}
        AWC decoded    https://aviationweather.gov/api/data/taf?ids=KORD&format=json
        AWC bulk       https://aviationweather.gov/data/cache/tafs.cache.csv.gz
SIGMET  IEM geometry   https://mesonet.agron.iastate.edu/cgi-bin/request/gis/sigmets.py
        IEM G-AIRMET   https://mesonet.agron.iastate.edu/cgi-bin/request/gis/awc_gairmets.py
        IEM text/PIL   https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py
        AWC US         https://aviationweather.gov/api/data/airsigmet?format=json
        AWC intl       https://aviationweather.gov/api/data/isigmet?format=json
        AWC bulk       https://aviationweather.gov/data/cache/airsigmets.cache.csv.gz
NOTAM   FAA API        https://external-api.faa.gov/notamapi/v1
        Knots          https://github.com/Estrellajer/Knots         (arXiv 2511.12630)
        NOTAM-Evolve   https://github.com/Estrellajer/NOTAM-Evolve  (arXiv 2511.07982, AAAI 2026)
        Q-line parser  https://github.com/slavak/PyNotam
        ICAO stored    https://dataservices.icao.int/  (paid)
```
