# Collecting non-US (global) TAF data — sourcing study

**Researched & verified live: 2026-08-01** (every HTTP code, record count, and licence below is from a
live fetch this session, not recalled). Companion to [aviation-data-sources.md](aviation-data-sources.md),
which it **corrects on one point**: that doc assumed TAF backfill was "global, to 1996" via IEM. That is
true for METAR (ASOS) but **false for TAF** — IEM's TAF archive is US/NWS-only. This study answers: how do
we get the *rest of the world's* TAFs, given the METAR geo-balancing lesson that the non-US tail is where
the model's value lives?

> **Bottom line:** there is **no clean deep global TAF archive**. The only deep global archive (Ogimet) is
> scraping-hostile (`robots: Disallow: /`). So the architecture is **forward-collection** from NOAA's global,
> public-domain feeds, plus a **~30-day catch-up buffer** from the AWC API, plus **MET Norway** for the one
> backfillable, licence-clean non-US archive (Norway, to ~1998). All primary sources are publication-safe.

## Comparison

| Source | Coverage | History | Auth | Format | Licence (store + redistribute?) | Verdict |
|---|---|---|---|---|---|---|
| **AWC bulk cache** `tafs.cache.xml.gz` | **Global** — 2,823 stns, **70% non-US**, 257 ICAO prefixes | current snapshot (`max-age=30`) | none | XML: `raw_text` **+ bundled decode** + lat/lon | **NOAA public domain** — yes | **PRIMARY (breadth)** |
| **AWC Data API** `/api/data/taf` | Global, per-station (`ids=` required) | **~30 days** via `date=YYYYMMDD` | none, ~100/min | raw / json / xml (+decode) | NOAA public domain — yes | **catch-up / backfill** |
| **NOAA tgftp** `cycles/{00,06,12,18}Z.TXT` | Global — 2,134 stns/cycle, 64% non-US | current only | none | raw text (concatenated bulletins) | NOAA public domain — yes | **redundant bulk (simplest cron)** |
| **MET Norway** `tafmetar` | Global last ~24 h **+ Norway to ~1998** | Norway deep; ROW ~24 h | UA header | raw text (multi-issuance) | **CC BY 4.0 / NLOD 2.0** — yes, attribute | **best licence; only non-US deep history** |
| **EC MSC datamart** | Global WMO relay, ~11 centres/day | ~30-day rolling | none | WMO bulletins + IWXXM | **ECCC End-use 2.1** — yes, attribute | secondary relay (narrower) |
| CheckWX | global | current (+paid history) | **key**, 200/day free | raw/decoded JSON | **ToS silent on redistribution** | avoid for a published corpus |
| avwx.rest | global (NOAA reparser) | current | **token** | JSON/XML | no gain over NOAA PD | avoid (no advantage) |
| **Ogimet** | global historical | deep | none | HTML scrape | unclear + `robots: Disallow: /` | **AVOID (scraping-hostile)** |
| DWD / UK Met Office / BoM / Meteostat / NCEI | — | — | — | — | — | **not raw-TAF sources** |

## The clean NOAA paths (primary architecture)

All three are **NOAA/NWS public domain** (`weather.gov/disclaimer`: NWS content is public domain unless
noted) — freely storable *and publishable* with attribution, provided we don't imply endorsement or present
modified data as official.

1. **AWC bulk cache — `https://aviationweather.gov/data/cache/tafs.cache.xml.gz`** *(the one we build on)*.
   The bulk TAF analog is **XML, not CSV** — `tafs.cache.csv.gz` → 404, `.xml.gz` → 200 (`max-age=30`, a
   live rolling snapshot; the dir index is 403 but the file is directly fetchable; no `robots.txt`). One
   request → 2,823 stations, **1,992 non-US (70%)**, 257 prefixes spanning every region. Each `<TAF>` has
   `raw_text` (the full bulletin) **and** a decoded `<forecast>` block (a free answer-key voice) **and**
   lat/lon (geo-balancing metadata). This single file builds a geo-balanced non-US corpus over time.

2. **AWC Data API — `/api/data/taf?ids=<ICAO>&date=YYYYMMDD&format=raw|json`.** Keyless, ~100/min. `date=`
   retrieves history: `date=` 30 days back returns data; 34+ days → empty, so the window is **≈30–33 days**.
   **One call returns one TAF** (the one valid at that date), so a per-station daily backfill is ~30 calls
   per station. This is the *catch-up buffer*: if the collector is down, or to seed history at launch, pull
   up to ~30 days per station. (An all-stations call with no `ids` → 400; use the cache for breadth, the API
   for depth.)

3. **NOAA tgftp — `tgftp.nws.noaa.gov/data/forecasts/taf/cycles/{00,06,12,18}Z.TXT`.** Four flat text files
   per day, ~2,134 global stations each (64% non-US), same PD licence. A redundant/alternative ingestor —
   the simplest possible cron if plain text is preferred over XML. (Per-station `stations/<ICAO>.TXT` also
   works, but the `stations/` listing is polluted with junk codewords — trust the cycle files.)

## The history problem, and the two partial fixes

Deep global TAF backfill from clean sources **does not exist**. The only deep global archive is **Ogimet**,
which disallows automated access for everyone but Googlebot and is known to ban scrapers — **off-limits** for
a corpus we intend to store and publish. The two partial fixes:

- **AWC API ~30-day catch-up** (above) — a rolling month of per-station daily history, public domain.
- **MET Norway `tafmetar`** (`api.met.no/weatherapi/tafmetar/1.0/taf.txt?icao=<ICAO>`) — the **only clean
  non-US deep archive**: Norwegian stations (EN**) go back to ~1998, and foreign stations return the last
  ~24 h. **Best licence of any source** (CC BY 4.0 / NLOD 2.0 — explicit redistribution, attribute "Data
  from MET Norway"); requires a descriptive User-Agent (403 otherwise); per-ICAO only.

So genuine temporal depth for the non-US set comes only from (a) running the forward-collector for a while
and (b) the AWC 30-day catch-up; only Norway can be backfilled to real archive depth.

## Recommended architecture

- **Primary (breadth), forward-collection:** poll `tafs.cache.xml.gz` **hourly** → immutable raw zone →
  deduped corpus. Global, 70% non-US, raw + bundled decode + geo. *(Built — `ingest/awc_taf.py`.)*
- **Seed / catch-up (depth ~30 d):** AWC Data API `date=` loop over a geo-balanced non-US station subset —
  a one-time seed (e.g. ~500 stations × 30 d ≈ 15K TAFs, ~2.5 h at 100/min) plus gap-recovery when the
  collector misses a window.
- **Redundancy:** tgftp cycle files as a backup ingestor (same PD licence).
- **Licence-safe supplement + Norway archive:** MET Norway `tafmetar` (CC BY 4.0) — a redistribution-
  bulletproof slice, and the only real non-US deep history.
- **Do NOT use for a publishable corpus:** Ogimet (robots-blocked), CheckWX (ToS silent on redistribution
  when the same data is NOAA PD), avwx.rest (token-gated, no gain).

## Licensing for eventual publication

- **NOAA/NWS (AWC cache, AWC API, tgftp):** public domain — publishable; attribute NOAA/NWS, don't imply
  endorsement, don't present modified data as official.
- **MET Norway:** CC BY 4.0 / NLOD 2.0 — attribute "Data from MET Norway".
- **Environment Canada:** ECCC End-use Licence 2.1 — attribute "Data Source: Environment and Climate Change
  Canada".
- **Model weights** still carry the Gemma Terms of Use regardless (see the publishing discussion); the
  harness/eval code stays MIT/Apache.

## Appendix — verified endpoints (2026-08-01)

```
AWC cache (PRIMARY)  https://aviationweather.gov/data/cache/tafs.cache.xml.gz     (XML; .csv.gz 404s)
AWC API (backfill)   https://aviationweather.gov/api/data/taf?ids=EGLL&date=YYYYMMDD&format=raw   (~30d)
NOAA tgftp cycles    https://tgftp.nws.noaa.gov/data/forecasts/taf/cycles/00Z.TXT  (06/12/18 too)
MET Norway           https://api.met.no/weatherapi/tafmetar/1.0/taf.txt?icao=ENGM  (UA required; CC BY 4.0)
EC MSC datamart      https://dd.weather.gc.ca/YYYYMMDD/WXO-DD/bulletins/alphanumeric/YYYYMMDD/FC|FT/CCCC/HH/
Ogimet               AVOID — robots.txt Disallow: / for all but Googlebot
```
