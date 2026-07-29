# Learning log

The real output of a learning project. One entry per working session: what was
done, what was *learned* (concepts, not just tasks), and open questions.

---

## Session 1 — 2026-07-29 · Scaffold

**Done**

- Verified the target model against primary sources: `google/gemma-4-E4B-it`,
  4.5B effective params, **Jan-2025 data cutoff**, Apache-2.0. The plan's
  post-cutoff test split is correct.
- Chose the repo shape: `aviai` = uv-workspace sandbox, `projects/avtext/` =
  first subproject (ADR-002).
- Scaffolded the full skeleton: package layers with responsibility docstrings,
  configs, data-zone rules, `DATA_LICENSES.md`, Makefile, CI, ADR-001/002/003.
- Synced a lean local env (no torch/transformers — fine-tuning is cloud-side).

**Learned**

- *Verify the linchpin fact before building on it.* Gemma 4 released after this
  assistant's knowledge cutoff, so "does this model even exist / what's its
  cutoff" was a real question, not a formality — and the answer defines the
  entire eval time-split. Cheap check, load-bearing result.
- *A time split is necessary but not sufficient.* METAR/TAF are so templated that
  a base model can decode post-cutoff strings from pre-cutoff format knowledge.
  So "did fine-tuning help" is best measured on **hard cases / rare tokens /
  abstention**, not clean decodes — those risk a ceiling effect that hides the
  signal. This should shape station selection *and* eval weighting. (→ ADR-003)
- *uv workspaces* are the idiomatic answer to "umbrella repo of self-contained
  subprojects sharing one env" — one lock, one `.venv`, per-project pyprojects;
  the cost is a single shared dependency resolution.

**Addendum — companion research docs integrated** (now in `docs/research/`)

Reconciling both briefings against the plan + scaffold:

- **Reframe (→ ADR-004).** Clean decode is a solved parser problem (~99–100%);
  the measurable prize is the **messy tail + faithful briefing + abstention**.
  Headline metrics + README results table updated to lead with these; clean
  decode demoted to a reference ceiling.
- **The oracle worry is answerable** — it's the *test-oracle problem*.
  Correctness becomes a property of authoritative artifacts + machine checks,
  never my judgment: AWC/IEM ship the **decoded JSON alongside the raw** (ground
  truth largely bundled); where it isn't, it's a table lookup (Q-codes,
  contractions); for prose, check **faithfulness, not taste**. One paid expert
  pass + the **rule of three** (0 errors in 300 ⇒ <1% error rate) bounds the rest.
- **E4B is dense, not MoE** → stable fine-tuning, no router to destabilize.
- **Stack divergence is intentional:** the datasources doc sketches the day-job
  Airflow/ClickHouse/GE/dbt stack; the plan deliberately uses DuckDB + Hypothesis
  + plain scripts to keep the learning repo self-contained. We follow the plan.

**Open questions**

- Confirm exact PyPI names for the oracle parsers; weigh `metaf` (C++/MIT) as a
  cross-language 4th lineage vs its integration cost (Phase 2).

**Next session — Phase 1 (data)**

0. **Scope probe** — run 2–3 parsers over a real sample, measure the
   parser-failure rate. Validates the premise and sizes the messy-tail prize.
1. Finalize ~50 stations in `stations.yaml` (incl. AUTO-heavy).
2. `ingest/iem.py` (checksummed backfill) + `ingest/awc.py` (collector — capture
   the bundled decoded JSON too: it's a free 4th oracle voice *and* the
   post-cutoff eval pool). First raw METAR/TAF in `data/raw/` + populated manifest.

---

## Session 2 — 2026-07-29 · Phase 1: scope probe

**Done**

- Built `ingest/awc.py` (bulk METAR-cache fetch → immutable raw zone + checksummed
  manifest) and `probe.py` (3 parsers + AWC's decoded columns as a 4th voice).
  `make probe`. Verified parser names + failure semantics (python-metar/mivek
  raise; avwx returns nulls).
- Ran it over **5,005 live METARs**.

**Findings (the honest version)**

- **Messy-tail fraction ≈ 3.4%** on a clean, US-heavy current feed:
  - **~1.5% hard parse failures** — almost all regional/ICAO format extensions
    (Scandinavian `W///S3` sea-state, `9999NDV`, `///` cloud-type suffixes) the
    US-centric python-metar doesn't implement; avwx/mivek cover most. **0% defeat
    all three parsers.**
  - **~2.0% disagreements** which — once units are handled — are *entirely* the RMK
    `T`-group precision convention (body `13/13` vs `T01250125` = 12.5°C). A schema
    rule (prefer the T-group), not garbled input.
  - wind_dir & wind_speed: **0% genuine disagreement.**

**Learned**

- **My first number (6.3%) was inflated by my own tool.** The 3% wind_speed
  "disagreement" was 100% a unit bug — avwx reports m/s natively and I read the unit
  off the wrong object. The disagreeing values ([4,8],[2,4],[5,10]… all exact 2×
  ratios) gave it away. Lesson in miniature: *"testing against a buggy oracle"
  includes your own measurement code* — verify a surprising number before trusting it.
- **A clean current feed does NOT justify the fine-tune by itself.** Parsers handle
  ~96.6% cleanly and unanimously. The prize lives in material this sample
  under-represents and that we must deliberately source/manufacture: (a) international
  + historical data (more regional formats → more failures), (b) the free-text **RMK**
  section (untested here — where the real unstructured mess is), (c) corruption
  augmentation (natural all-parser-failures ≈ 0, so the abstention signal must be
  manufactured — exactly as both briefings predicted). This *confirms ADR-004's
  hybrid framing with numbers.*
- Infra: **GMKtec EVO-X2** home server profiled (Strix Halo, 123 GiB, `dashi`) →
  ADR-005; **no OVH VPS exists** (plan corrected).

**Next**

- Phase-2 schema: prefer the RMK `T`-group for temp/dewpoint precision.
- Phase-1 data: finalize ~50 stations (weight EU + AUTO-heavy to raise the failure
  signal); `ingest/iem.py` historical backfill; extend probe/collector to capture
  RMK + international feeds; run the collector on a schedule (`dashi` or Actions).

---

## Session 3 — 2026-07-29 · Phase 1: station set + IEM backfill

**Done**

- Finalized **51 stations** (`configs/stations.yaml`), evidence-weighted per the
  scope probe (Scandinavia + EU-AUTO for regional formats, US-AUTO small fields for
  the T-group, CIS/China for MPS units, plus terrain/coastal/arctic variety).
  Validated **51/51** against IEM.
- Wrote `ingest/iem.py` (fetch / validate / backfill / to_parquet), reusing awc.py's
  raw-zone + manifest discipline; gzip `mtime=0` for byte-idempotent raw.
- Backfilled 3 years (2023-07-30 → 2026-07-28): **6,489,661 observations**, 51
  stations, **152 MB** Parquet (`data/processed/metar/iem_metar.parquet`),
  100% distinct raw.

**Learned**

- **The row counts betrayed a sampling trap.** US stations carry ~320k obs each vs
  ~52k for EU — US ASOS is archived sub-hourly (~5-min) vs EU half-hourly, a **6:1**
  gap (all genuine distinct reports, not dupes). Uniform row-level sampling would be
  ~70% US and **drown the EU messy tail I deliberately selected for.** Station-level
  weighting is necessary but *not sufficient*: downstream eval/train sampling must be
  **stratified per station** (or downsampled to a common cadence, e.g. hourly). The
  careful station picks only pay off if the row sampler respects them.
- **Reproducibility lives at the eval-freeze layer, not the raw pull.** Last session
  I flagged the drifting "3-years-from-today" end-date as a wrinkle; corrected view:
  the raw corpus is *meant* to grow (the collector extends it), so a moving end-date
  is fine. The manifest records each pull's provenance; byte-reproducibility is
  enforced when we freeze `eval/v1` with content hashes (Phase 3), not on the raw
  layer.

**Next**

- **Phase 2 (schema):** canonical pydantic model from FMH-1 / AC 00-45H, encoding
  the rules the probe already gave us (wind → kt; prefer the RMK `T`-group for
  temp/dewpoint). Then oracle adapters over the 3 parsers + AWC decode → tier-0
  round-trip/invariants → consensus.
- Build a **stratified sampler** (per-station cap / cadence downsample) before any
  eval set, so US volume can't dominate.

---

## Session 4 — 2026-07-29 · Phase 2: canonical METAR schema

**Done**

- `schema/metar.py`: pydantic v2 `MetarObservation` + component models (`Wind`,
  `CloudLayer`, `WeatherGroup`) + controlled-vocab `StrEnum`s. v1 covers report
  metadata, wind, visibility, weather, clouds, temp/dewpoint, altimeter.
- `tests/test_schema.py`: builds a real EPGD report into the schema; asserts
  structural validation rejects bad input and forbids extra fields. 8 tests green.

**Design decisions (encoded as the schema)**

- **Canonical units** (KT, hPa, °C) + a `*_source` field for round-trip — directly
  fixes the probe's phantom unit "disagreements".
- **None == absent** — load-bearing for hallucination (value where gold=None) and
  abstention (correctly None) scoring.
- **Schema validates STRUCTURE only.** Cross-field physical invariants
  (dewpoint ≤ temp, gust > wind) and WMO-vocabulary membership go in `quality/`
  next. The big WMO weather table lives in ONE place (the CCT file), not duplicated
  as a schema enum — so `phenomena` stay validated strings, small vocab stays enums.
- **Decode only what the raw encodes:** time = day/hour/minute (DDHHMMZ), not a
  full datetime (that's external metadata from IEM/AWC).

**Deferred (TODO in code):** RVR, directional/variable visibility, structured RMK
groups, TAF schema.

**Next:** oracle adapters (`oracles/`) mapping python-metar / avwx / mivek + the
AWC decode into `MetarObservation`; then tier-0 round-trip + invariants
(`quality/`); then consensus.
