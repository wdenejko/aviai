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

---

## Session 5 — 2026-07-29 · Phase 2: oracle adapters

**Done**

- `oracles/`: one interface (`Oracle` → `OracleResult`) + shared header/unit helpers
  (`base.py`), and 3 adapters mapping python-metar / avwx / mivek into
  `MetarObservation` (core fields; weather is TODO(v2)). `ORACLES` panel.
- `tests/test_oracles.py` (12): clean decode, uniform failure, MPS→kt, COR — 20 green.
- Cross-checked all 3 on **1,020 real corpus METARs**.

**Findings**

- True per-parser failure ≈ **0.6–0.7%**, each on a *different* regional format:
  python-metar on Australian `RF` groups + clouds-after-CAVOK (YSSY), mivek on
  Russian RVR (UUWW), avwx on none. Oracle independence demonstrated — different
  lineages fail on different things, so the panel covers more than any one parser.
- Field agreement (of all-parsed): wind / clouds / visibility ≈ 100%. Genuine
  remaining differences: **temp/dew 0.3%** (RMK T-group precision — python-metar
  reads it, the others use the body) and **alt 22%** (mivek rounds inHg→hPa ~1 hPa
  low; majority vote python-metar+avwx outvotes it). Both are the panel working as
  designed, not bugs.

**Bugs the cross-check caught (all fixed) — measurement artifacts, again**

- Leading `COR` modifier (`COR EGCC ...`) broke the header regex + mivek's strip.
- mivek fractional SM (`1/8SM`) parsed as 1 SM (the fraction was dropped).
- python-metar "day out of range" crash: it assumes the *current* month for the
  DDHHMMZ datetime, so day-31 reports crashed (~2% phantom "failure"). Passing
  `month=1` fixes it — and it's exactly why the schema takes day/hour/min from the
  raw header, not the parser's constructed datetime.

**Next:** `quality/` tier-0 — round-trip (re-encode ≈ raw after canonicalization,
via Hypothesis) + invariant suite (dewpoint ≤ temp, gust > wind, vocab ∈ WMO/CCT,
station ∈ registry); failures route to the hard-case queue. Then consensus.

---

## Session 6 — 2026-07-29 · Phase 2: tier-0 quality (invariants + round-trip)

**Done**

- `quality/invariants.py`: cross-field physical checks (dewpoint ≤ temp, gust >
  wind, variable-range completeness, cloud-base ordering, plausible temp/altimeter
  ranges, CAVOK consistency). `check_invariants(obs)` → violations; non-empty routes
  to the hard-case queue.
- `quality/roundtrip.py`: `encode_metar(obs)` — minimal canonical encoder (header,
  wind, temp/dew, Q-altimeter) enabling the round-trip property.
- Tests: 6 invariant unit tests + a **Hypothesis round-trip** (`decode(encode(obs))`
  reproduces the numeric core, 150 generated examples). 27 green.
- Scanned the corpus through decode → invariants (15,300, then a 3× rescan of 46,920).

**Findings**

- After fixes, **0 invariant violations in 46,695 parsed reports** (3× rescan) — archived ASOS
  data is physically clean (QC'd). So tier-0's live signal on this archive is
  parse-failures; the invariants are the safety net for corrupted/messy inputs
  (corruption augmentation, later) and a dev-time adapter-bug detector.

**Two bugs the scan caught (both fixed) — tier-0 working in both directions**

- `is_cavok` matched `CAVOK` inside a trend group (`BECMG CAVOK`) and falsely
  flagged the whole obs as CAVOK. A real *adapter* bug, caught by the CAVOK
  invariant. Fixed: `is_cavok` now reads only the body (truncate at
  BECMG/TEMPO/NOSIG/RMK/PROB/FM).
- cloud-base-ascending fired on `BKN100 FEW054CB` — CB/TCU are legitimately
  reported out of height order. The *invariant* was too strict. Fixed: exempt
  CB/TCU. (One real bug below the checker, one mis-calibrated checker — exactly the
  two things tier-0 exists to surface.)

**Next:** tier-1 **consensus** — field-level majority vote across the 3 oracles
(+ AWC decoded JSON as a 4th voice); disagreement → hard-case queue; track panel
agreement (Krippendorff's α). Then the gold seed + mutant-injection calibration.

---

## Session 7 — 2026-07-29 · Phase 2: tier-1 consensus

**Done**

- `consensus/vote.py`: field-level majority vote across voices → consensus label +
  per-field agreement + hard-case trigger (**no clear majority**, not mere dissent).
- `consensus/alpha.py`: hand-rolled **Krippendorff's α** (nominal, missing-tolerant)
  — the panel-agreement KPI.
- Tests: 4 α (incl. a hand-computed 0.444) + 3 consensus. **34 green.**
- Demonstrated on 2,040 real reports (3-voice panel).

**Findings**

- Per-field α: wind / clouds / report-type / cavok = 1.000; vis / temp / dew
  0.997–0.999; **altimeter = 0.856** — α cleanly quantifies mivek's ~1 hPa inHg
  dissent as a KPI, isolating the one field where the panel is least unanimous.
- **0% no-majority hard cases on every field.** With 3 voices, mivek's altimeter
  dissent is always outvoted 2-1 by python-metar+avwx → resolved, not flagged. The
  design choice (hard case = *no majority*, not *any dissent*) is what keeps 22%
  dissent from flooding the queue while α still records it.
- So on this clean archive the panel essentially never genuinely conflicts. The open
  risk is no longer "do they disagree" but "is the AGREEING majority *correct*" —
  which is exactly what the gold seed + mutant audit calibrate next.

**Property-test aside:** the Hypothesis round-trip caught my own *test strategy*
generating an impossible METAR (VRB + 75 kt), whose encoding is ambiguous. Fixed the
strategy to realistic winds and made it deterministic (`derandomize`) so CI can't flake.

**Next:** the AWC decoded JSON as an independent **4th voice** (different lineage —
guards the "3 parsers share a bug" risk); then the **gold seed** (transcribe
100–200 worked examples from AC 00-45H / FMH-1) + **mutant-injection audit** to
calibrate whether the consensus is actually right, not merely agreed.

---

## Session 8 — 2026-07-29 · Phase 2: gold seed + mutant audit (trust-ladder calibration)

**Done**

- `consensus/mutants.py`: mutant injectors + `detect()` (parse | invariant | consensus)
  + audit. Judgment-free — plant a known error, confirm it's caught.
- `consensus/gold.py`: `calibrate(consensus vs authoritative reference)` +
  `gold_from_awc_rows` (NOAA's official decode as the scalable Tier-2 authority). The
  AC 00-45H hand-transcribed gold is the user's copy-task, plugging into the same
  `GoldRecord` format — **not fabricated here** (correctness comes from an authority,
  never our own judgment).
- Tests: 4 mutant + 2 gold. **40 green.**

**Mutant audit (604 clean reports)**

- dewpoint>temp, altimeter-out-of-range: **100%** caught by invariants; gust<wind 99%;
  garble **100%** by parse. **temp+4 / altimeter+6 (plausible): 0% caught.** The empirical
  proof that consensus catches *disagreement*, invariants catch *impossibility*, but a
  plausible-but-wrong value slips through everything short of an authority. That 0% is
  *why* Tier-2 exists.

**Gold calibration (consensus vs NOAA official, ~800 reports)**

- Caught a **real shared bug** consensus was blind to: altimeter 93.5% — every divergence
  identical (consensus 1012 vs official 1013 on inHg reports). Root cause: the adapters
  `round(…,1)`'d 1012.53 → 1012.5, then Python's banker's rounding took 1012.5 → 1012. All
  parsers shared it, so consensus *agreed on the wrong value*; only the authority caught it.
- **Fixed** (keep full precision, round only at compare) → altimeter **100%**. The full
  loop: gold reveals bug → fix → re-calibrate → resolved. Every field now ≥98.7% vs NOAA;
  the residual temp/dew ~1% is the known RMK T-group-vs-body precision (v2: prefer the
  T-group in consensus).

**The trust ladder, whole:** tier-0 (impossibility) + tier-1 (disagreement) + tier-2
(authority) each catch a *different* error class; none alone suffices. The mutant audit
measures the gaps; the gold seed fills the biggest one.

**Next (Phase 3 — the eval harness, the "heart"):** wire the hard-case queue as a
first-class artifact (parse-fail + invariant + no-majority + gold divergence); freeze
`eval/v1` (station+time splits, content hashes); then the scorers / stats / report.

---

## Session 9 — 2026-07-29 · Phase 3: hard-case miner + altimeter panel cleanup

**Done**

- `harness/hardcases.py`: `classify(raw) -> CaseVerdict` — one report through the whole
  ladder, one structured verdict (headline `label` = hardest signal; keeps every signal
  + the consensus `reference`). The labeling function the eval set is built from. A richer
  superset of `mutants.detect()`, kept in `harness/` to preserve layering.
- `harness/mine.py`: deterministic (md5-ordered), station-stratified scan of the corpus →
  labeled `pool.jsonl` + provenance `manifest.json`. Stratified because 6.4M rows are
  US-dominated; equal per-station quotas protect the rare regional formats.
- Buckets: `parse_fail` > `invariant` > `no_majority` (tie, **no** reference → human/gold
  queue) > `dissent` (majority resolved, **keeps** reference → the richest auto-scorable
  stratum) > `clean`. Test: 3 (reuses the mutant injectors as known-labelled inputs). **43 green.**

**The number that lied (and the discipline that caught it)**

- First mine (25,500 reports): **dissent 21.58%.** Suspiciously high. *Looked at which
  fields* → `altimeter_hpa` was **5,362 of 5,502** (96%). The vote.py docstring had already
  predicted it: "mivek ~1 hPa low on inHg." Not a messy tail — one parser's systematic quirk.
- Root cause #1: mivek only exposes **integer** hPa and **truncates** the inHg→hPa convert
  (30.14 inHg = 1020.66 → `1020`). Unrecoverable at source → mivek now **abstains** on inHg
  altimeters (`None`), stays a full voice on Q reports. Dissent 5,502 → 233.
- That *exposed* 114 new `no_majority` **ties** — all altimeter, all inHg. With mivek gone,
  python-metar vs avwx straddled the X.50 rounding boundary (A2984: `1010.501` vs `1010.499`
  → 1011 vs 1010) because the two libraries hardcode inHg→hPa constants that differ in the
  5th digit. Root cause #2: **canonicalize from the reading, not the parser's constant** —
  python-metar now converts `value("IN")` with the shared `inhg_to_hpa`, identical to avwx.
  Ties 114 → 0.
- Honest tail after both fixes: **2.09%** = parse_fail 1.17% + dissent 0.92% (temp / dewpoint
  / visibility / clouds — genuine, e.g. RMK T-group vs body). Altimeter noise: gone.

**Regression guard:** re-ran the gold seed (5,005 NOAA records) → altimeter **99.98%**
(was worried the canonicalization would drift it; it didn't). The lone altimeter divergence
is a dual-altimeter report whose own `Q1002` and `A2962` disagree at source — a data quirk,
not our bug.

**Lesson:** a headline rate (21.58% "hard") is a story you haven't finished reading. The
field-level breakdown turned it into *two* latent parser-quality bugs; mining is how you
find the bugs the unit tests can't see, because they only surface at corpus scale. Separate
**reading** (what the parser saw) from **canonicalization** (how we normalize) — spurious
disagreement lives in the gap.

**Next:** freeze `eval/v1` from the pool — stratified (over-sample the 2.09% tail per
ADR-004), disjoint station+time splits, content-hashed manifest; then scorers / stats / report.
