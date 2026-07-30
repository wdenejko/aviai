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

---

## Session 10 — 2026-07-29 · Phase 3: freeze eval/v1 (the immutable eval set)

**Done**

- `harness/freeze.py`: deterministic, station-stratified builder → `eval/v1/eval.jsonl`
  (**620 records**) + `manifest.json` (sha256 = immutability anchor). Froze twice → identical
  hash. Pure `assign_split` (station+time → disjoint bucket) unit-tested. **46 green.**
- ADR-006 (split policy) + `eval/v1/README.md`. Split chosen by the user: **station + time
  both** (strictest — two generalization axes reported separately).

**The design was discovered by measuring, not decided up front — four findings reshaped it:**

1. **US-volume dominance (again).** A global-random recent sample is swamped by high-freq US
   airports (all clean); the rare tail stations vanish. Fix: station-stratify the freeze scan,
   same as the miner. (First symptom: unseen_time found 6 parse_fails in 30k.)
2. **The station tags were wrong.** `stations.yaml` credits Scandinavia with the parse-fail
   tail; the data says **UUWW** (Moscow, 38%) and **YSSY** (Sydney, 18%) — and dissent lives at
   the US T-group fields (KAIG/KOZW/KEKM). Picked held-out stations from the *measured*
   distribution instead of the tags.
3. **"Frozen" needs a total order.** The stratified query had no outer `ORDER BY`, so DuckDB
   streamed rows in nondeterministic parallel order and the set's hash changed run-to-run.
   A frozen artifact that isn't byte-reproducible is not frozen. Added the outer md5 order.
4. **Parse failures are non-stationary in time.** YSSY: 53% (2023) → 0% (fixed exactly at
   2025-02). UUWW: 37% → **74%** across 2026-05. Post-cutoff, UUWW is the *only* real
   parse-fail source. So abstention can't be tested by station-holdout (holding out UUWW
   strands the sole source with nothing to train on). Forced design: UUWW stays in training,
   abstention tested by **time-holdout** on it (train pre-boundary, score post-boundary) — which
   is actually the cleanest abstention experiment. Hence the **asymmetric** strata:
   unseen_station = {200 clean, 60 dissent}; unseen_time = {200 clean, 100 parse_fail, 60 dissent}.

**Rigor checks:** every item ≥ 2025-02-01 (past Gemma's Jan-2025 cutoff — no memorization);
splits provably disjoint (held-out stations appear in exactly one); 620/620 ids unique;
manifest sha256 == sha256(eval.jsonl). Records are textbook tail: YSSY `RF00.0/000.0`
(Australian rainfall group), UUWW `22002MPS … R01/000062` (Russian m/s + RVR), KEKM
`T02450228` RMK (24.5 °C vs body 25).

**Lesson:** an eval set's split policy is a *hypothesis about where difficulty lives*, and the
data will falsify it. Every a-priori belief here (tags, symmetry, "any-time station holdout")
was overturned by a measurement. Freeze late, measure first, and let a short cell stand as a
finding rather than padding it.

**Next:** the scorers — field-level exact match, abstention/hallucination (the None-vs-value
distinction the schema was built for), whole-record EM — then bootstrap CIs + the run report.
The harness stays frozen from here.

---

## Session 11 — 2026-07-29 · Phase 3: scorers + stats (the None-aware core)

**Done**

- `harness/score.py`: field-level scoring on a five-way taxonomy — HIT / WRONG / ABSTAIN
  (declined a knowable value, safe) / HALLUCINATE (fabricated from nothing, dangerous) /
  TRUE_ABSTAIN. Aggregates to precision (fabrication penalty), recall = value accuracy
  (abstention penalty), F1, hallucination_rate, whole-record EM; `aggregate_by` slices by
  split × label. The point: in a safety domain the two ways of being wrong are NOT equal, so
  they are never folded into one "incorrect" bucket.
- `harness/stats.py`: `bootstrap_ci` (resample RECORDS — the independent unit, not the
  correlated fields — seeded, reproducible) + `mcnemar` (paired baseline-vs-finetuned test
  for Phase 4; continuity-corrected, p via erfc, no SciPy). Tests: 8. **54 green.**

**Demonstration** (scorer on real eval/v1, python-metar as a mock predictor):

| label | recall (value acc) | halluc | EM |
|-------|-------------------:|-------:|---:|
| clean | 100.0% | 0.0% | 1.00 |
| dissent | 89.9% | 0.0% | 0.09 |
| parse_fail | 97.0% | 0.0% | 0.97 |

Three lessons fell straight out of it:

1. **Parsers cannot hallucinate — 0% everywhere.** They HIT, WRONG, or ABSTAIN; they never
   invent a field. That is exactly why they're safe but capped, and why the whole point of
   the LLM comparison is: can it match their accuracy *without* lighting up the hallucination
   column? The null / always-abstain baseline confirms the floor — 0% recall, 0% halluc.
2. **EM ≠ field accuracy** — dissent shows 90% of fields right but 9% of *records* perfect:
   one wrong field (python-metar's RMK T-group temp, the very thing it's outvoted on) tanks
   the whole-record score. Report both or you'll fool yourself.
3. Scoring a consensus MEMBER against the consensus is mildly circular (clean → 100% by
   construction); harmless as a wiring check, and moot for the LLM, which isn't a voice in it.

**Next:** the runner — put the base model (Gemma-4-E4B) behind the same predict(raw)->fields
interface the demo used, produce the first *real* scored baseline + a reproducible run report.
That needs a compute/serving decision (ADR-003/005: eval locally — llama.cpp on the Mac, or
the GMKtec), so it's the next checkpoint.

---

## Session 12 — 2026-07-29 · Phase 3: runner + report (harness plumbing complete)

**Done**

- `harness/prompt.py`: versioned prompt (`decode-json-v1`) — the load-bearing line is
  "use null; do NOT guess", which is what makes a fabrication the model's own fault rather
  than obedience to an implied "always answer". Plus a robust output parser: balanced-brace
  JSON extraction (survives prose + ```json fences + truncation) that canonicalises into the
  reference's space (visibility→100 m buckets, temps/altimeter→int, clouds list→tuple).
- `harness/runner.py`: the `Predictor` contract `(raw) -> fields | None` — the one seam
  between how a decode is produced (parser / local LLM / API) and how it's judged. `run_eval`
  scores a predictor over eval/v1; `write_report` emits `reports/runs/<id>/` with report.md,
  run.json, scores.jsonl and a **reproducibility block** (eval sha256 + prompt id + model id +
  decode params). Phase 3's definition-of-done artifact. Tests: 5. **59 green.**

**Proved end-to-end** with the parser baseline (python-metar behind the predict contract):

| scope | value acc | halluc | EM |
|-------|----------:|-------:|---:|
| overall (620) | 97.5% [96.9, 98.1] | 0.0% | 82% |

The plumbing is complete: frozen eval → predict → score → bootstrap CI → report, all
re-derivable from the sha256 in the block. Caveat recorded in the report: the parser baseline
is mildly circular (a consensus member scored against the consensus) — a wiring check and a
reference point, not the real eval. It sets the bar the LLM must clear: high accuracy at
**zero** hallucination.

**Next (the checkpoint):** the model backend. Put Gemma-4-E4B behind `predict` and generate
the first real baseline. Needs the serving call — llama.cpp on the Mac (ADR-005's eval box,
least friction) vs the GMKtec (needs ROCm/llama.cpp stood up first). Then Phase 4: finetune,
re-run the SAME frozen harness, McNemar the pair.

---

## Session 13 — 2026-07-29 · Phase 3: model backend + FIRST REAL BASELINE

**Done**

- `harness/models.py`: `http_predictor` — the model backend behind the `predict` seam, an
  OpenAI-compatible HTTP client via httpx. CI-safe: no inference lib enters the package, so
  Linux CI stays green and the finetuned model plugs into the identical seam. `runner` CLI +
  `make serve` / `make harness` wired; pyproject serving note corrected. **59 green.**
- **Serving reality:** the ADR-003 target `gemma-4-E4B-it` was already on disk (MLX 4-bit —
  no download). It's Gemma **3n = multimodal**, so plain `mlx-lm` refused it (`126 params not
  in model`); `mlx-vlm` loads it text-only. Served via `mlx_vlm.server` (a `uv tool`, off the
  package), harness talks HTTP. ~4 s/record, 620 records in ~40 min, temp 0 (reproducible).

**First real baseline — base Gemma-4-E4B on frozen eval/v1 (620 records)**

| scope | value acc | halluc | EM |
|-------|----------:|-------:|---:|
| overall | 75.1% [73.5, 76.7] | 1.5% | 4% |

The single dominant finding, and it's mechanistic:

- **Altimeter is the Achilles heel — 31% overall, and split by source: `Q` (hPa) 90% vs `A`
  (inHg) 2%.** The model reads hectopascals but ALMOST NEVER converts inHg→hPa (390/416
  wrong — it emits `30.12` for `A3012` instead of 1019). One narrow arithmetic skill owns the
  error budget and caps EM: ~400 of 620 records are inHg, so EM can't exceed ~35% until this
  is fixed (actual EM 4%).
- **Strong at "reading":** report_type 96%, cavok 92%, temperature 88%, dewpoint 87%.
  **Weak at "extract/convert":** wind_speed 60%, altimeter 31%.
- **Hallucination low but NOT zero: 1.5% (11 fabrications — mostly invented wind_gust/wind_dir).**
  The smoke test's 0% was a 20-record artifact; at scale the model does occasionally fabricate.
- vs the parser reference (97.5% / 0% / 82%), the base LLM is far behind on structured decode —
  as ADR-004 predicted — and the gap is dominated by one learnable skill.

**Phase 4 hypothesis, now concrete and measurable:** LoRA the inHg→hPa conversion (and wind
extraction). Target: altimeter-`A` 2% → 90%+, overall value-acc → mid-80s, EM off the floor —
while hallucination stays flat (the safety constraint). Re-run the SAME frozen harness (same
eval sha256, same prompt), McNemar the paired per-record correctness. The harness is frozen;
nothing about the eval moves from here.

*(Numbers in this session were pre-correction — superseded by Session 14 after a bug scan.)*

---

## Session 14 — 2026-07-30 · Bug scan + corrected baseline + detailed report

Requested a bug scan "just in case" after the clouds crash. Audited the whole measurement path.

**Scoring path — 3 bugs found & fixed:**

1. **Clouds-shape crash.** The output parser assumed `clouds` = `[cover, base]` pairs; the model
   sometimes emits *dicts*, and the `KeyError` scored the WHOLE record invalid. Turned out **all 25**
   of the run's "invalid" records were this — not the model. Fix → JSON-valid 96% → **100%**, overall
   value-acc **75.1% → 77.8%**. A 2.7-point harness artifact, not a model gain.
2. **Under-canonicalisation.** `_canon` passed the model's raw value through for `wind_dir/speed/gust`
   (ref: int), `report_type` (upper), `automated/cavok` (bool) — a right-but-mistyped answer scored
   WRONG. Fixed to coerce every field (incl. the `bool("false")==True` trap). **Audit: re-scoring the
   saved predictions with/without the fix flipped 0 outcomes** — a no-op for this clean-JSON model,
   kept for robustness.
3. **Div-by-zero** on an empty eval — guarded (latent).

**Reference path — clean bill of health.** The gold seed only validated *numeric* fields; `clouds` /
`cavok` / `automated` / `report_type` were never checked, and the last three are shared functions
echoed across all parsers (illusory consensus). Adversarial review + empirical checks found **no
bugs**: conversions correct; `automated`/`cavok` matched an independent recompute **620/620**;
`clouds` decode hand-verified, adapters agree 99% (disagreements are same-base ordering), 0 None;
**0/620 degenerate** references; all 100 parse_fail refs are **2-way cross-checked**.

**Corrected baseline (620 records):** value-acc **77.8%** [76.8, 78.9], halluc **1.5%**, EM **4%**,
JSON-valid **100%**. Conversion splits, sharper: altimeter `Q` 95% / `A` **2%**; wind `KT` 75% /
`MPS` **1%**; visibility metric 94% / `SM` 71%. EM ceiling if altimeter fixed ≈ **28%**.

**Deliverable:** `reports/baseline_v1_analysis.md` — 12-section report (exec summary → methodology →
findings → measurement integrity → Phase-4 targets → threats to validity).

**Lesson:** the scan recovered 2.7 points that were a harness artifact, not model skill. Always audit
the measurement code before a headline number gates the next phase — the scorer is as much on trial
as the model. Two ways of being wrong (fabricate vs abstain), and two places a bug can hide (scoring
vs reference): check both.

---

## Session 15 — 2026-07-30 · Phase 4 pivot to dashi + small-model size sweep

Pivoted Phase 4 to **dashi** (GMKtec Strix Halo, AMD, 123 GiB) — more headroom, and the study is stronger as a
**size sweep** than one model. Full write-up: `reports/size_sweep_v1.md`; decision: ADR-007.

**dashi analysis:** Fedora 43, Ryzen AI Max+ 395, GPU via **Vulkan** (ROCm only inside `kyuz0/amd-strix-halo`
**toolbox** containers), models served `toolbox run … llama-server` on **:8080** (only firewall-open port; M5
reaches it over LAN). `*-Unsloth` dirs → an existing finetune workflow. Built `~/serve.sh` to swap the :8080
model; harness runs from the M5 → `dashi:8080`.

**Size sweep (5 models, Q8_0 GGUF, frozen eval/v1):**

| model | value-acc | halluc | EM | inHg→hPa |
|-------|----------:|-------:|---:|---------:|
| 270M / 0.5B-Qwen / 1B / 2B / 4B | 17 / 5 / 26 / 51 / **69**% | 67 / 1 / 20 / 10 / 8% | 0 / 0 / 0 / 0 / 4% | 0 / 0 / 0 / 0 / **1**% |

Three findings:

1. **Decode scales cleanly** (Gemma 17→26→51→69%), but **whole-record EM needs ~4B**.
2. **The unit-conversion gap is UNIVERSAL** — inHg→hPa 0–1% and m/s→kt 0–2% at *every* size, including 4B. Not a
   scaling problem; a capability none of them have. The strongest confirmation yet that conversion is the lever.
3. **Two opposite tiny-model failure modes (cross-family):** Gemma-270M is *reckless* (67% hallucination), Qwen-0.5B
   is *conservative* (abstains ~60% of fields → 5% acc, 1% halluc, 99% JSON). Same size, opposite safety profiles.

**The serving stack is a variable:** the SAME E4B scores **69% on dashi (llama.cpp/Q8)** vs **77.8% on the M5
(MLX/4-bit)** — ~9 points, *not* quantization (Q8 > 4-bit). Likely llama.cpp's Gemma-3n (PLE/MatFormer) being
less faithful than MLX. So the finetune anchor is the **same-stack 69%**, never the M5 number — which is exactly
why redoing the baseline on dashi was right.

**Lesson:** the inference stack is part of the measurement, worth ~9 points on a hard arch — a before/after must
never cross stacks. **Next:** plan the LoRA finetune (Unsloth on dashi), targeting the universal conversion gap.

---

## Session 16 — 2026-07-30 · Phase 4 training stack: Unsloth bf16 LoRA on Strix Halo (ROCm)

Stood up a real GPU finetuning stack on dashi's AMD iGPU — a frontier combo (Unsloth is
CUDA-first; gfx1151 is new silicon). Full runbook + traps: ADR-008. Built layer by layer,
checkpointing each:

1. **PyTorch-ROCm sees the GPU.** `torch 2.11+rocm7.2` → `cuda.is_available()==True`, device
   "AMD Radeon 8060S", real matmul — **natively, no HSA override** (ROCm 7.2 has genuine
   gfx1151 support). The make-or-break layer, and it cleared first try.
2. **Unsloth bf16 LoRA loads on the GPU.** `unsloth[amd]`; no QLoRA/bitsandbytes quantisation
   needed at ≤4B on 123 GiB (though bnb must still be *importable* — Unsloth loads Linear4bit
   during patching).
3. **SFT data** (`finetune/build_sft.py`): 3000 conversion-heavy pairs from the disjoint train
   split — targets carry the CORRECT conversions (`18005MPS -> wind_kt 10`, A-reports -> hPa),
   inHg 1195 / m/s 333. (Caught + fixed a coverage bug: the row cap starved MPS to 0 until I
   interleaved stations by md5.)
4. **Training loop works** (`finetune/train_lora.py`): gemma-3-1b smoke, loss 3.75 → 3.57,
   adapter saved. Full 1-epoch 1B run launched.

**Traps worth remembering (each cost a cycle):** system Python 3.14 has no ML wheels (pin
3.12); uv rejects the bnb preview wheel's non-PEP440 version "1.33.7.preview" (use pip);
bitsandbytes is imported even for bf16; Unsloth's SFTTrainer won't auto-format `{"messages"}`
(apply the chat template into a `text` field yourself).

**Cost:** ~14 s/step for a 1B LoRA — early ROCm, no CK/flash-attn on gfx1151, so no fast
iteration, but it trains. **Next:** finish the 1B run, merge -> GGUF, serve on dashi:8080,
re-run the frozen harness, McNemar the finetuned-1b vs base-1b (26%). Then test whether Unsloth
can load gemma-3n for the E4B anchor.

---

## Session 17 — 2026-07-30 · Re-anchor on *real* Gemma 4 + E2B finetune (the big correction)

The user asked why I'd finetuned "gemma 3." I hadn't noticed: **Gemma 4 shipped 2026-04-02, after my
knowledge cutoff**, so every prior session silently ran **Gemma 3n** while *labelling* it gemma-4 (the
baseline header literally read "gemma-4-E4B-it (Gemma 3n)"). Verified real Gemma 4 exists via the HF API
(`unsloth/gemma-4-E2B-it`), archived all gemma-3 work (repo `reports/gemma-3/`, dashi `~/models/gemma-3/`,
`~/ft/gemma-3/` — nothing deleted), and redid E2B on the real model. Full record: ADR-009; write-up:
`reports/gemma-4/e2b_finetune_analysis.{md,html}`.

**The E2B result (real Gemma 4, same-stack, 620 records):**

| metric | base | finetuned | Δ | McNemar |
|--------|-----:|----------:|---:|---------|
| value accuracy | 69.5% | **92.9%** | +23.5 | 1437 fix / 3 regress, p≈10⁻³¹² |
| hallucination | 31.0% | **0.8%** | −30.1 | — |
| whole-record EM | 2.4% | **38.2%** | +35.8 | 223 fix / 1 regress, p≈10⁻⁴⁹ |

Uniform across difficulty (clean +24.2, dissent +23.4, parse_fail +20.6). LoRA r=16, 1 epoch, 3000 pairs,
loss 1.73→0.15, ~34 min. Unsloth 2026.7.6 loads the Gemma-4 PLE arch natively (resolves ADR-008's open Q).

**Learned**

- **A post-cutoff release is a load-bearing blind spot, and *labels lie*.** Session 1's log even flagged
  "verify the linchpin fact" — but the check decayed: the name `gemma-4-E4B-it` on disk was gemma-3n, and I
  trusted the name for weeks. The famed "77.8% M5 vs 69% dashi stack gap" was **gemma-3n mislabelled on both
  sides** — not a stack finding at all. Re-verify identity against reality (HF API / config `model_type`),
  not the filename, before a number gates anything.
- **Gemma 4 is a *reasoning* model.** Served naively it thinks until the token cap and returns empty
  `content`; needs `--reasoning-budget 0` to answer directly (which matches the finetune's bare-JSON target).
  A generational arch change the gemma-3n path didn't have.
- **Template drift is a serving stack.** The finetuned model first scored 73.5% JSON-valid — it emitted
  `<|turn>model` as text on 164 records. Cause: llama.cpp renders Gemma-4's tool-calling chat template with
  its own engine (minja), diverging from HF Transformers' render at *training*. A base model shrugs it off;
  a hard-finetuned model (loss 0.06) overfits to the exact training prompt and shatters. Feeding the training
  wrapper raw via `/completion` → 100% valid, 92.9%. **"Same template, different engine" is a different
  stack — serve == train, byte for byte.** (→ `harness/models.completion_predictor`, `runner --completion`.)
- **The eval is a *fidelity* test, not a *superiority* test — and I'd half-forgotten it.** Every reference is
  parser-consensus + NOAA gold, so the LLM's ceiling *is* the parser (~99.9%). I teased "parse_fail is where
  the LLM earns its keep"; the data corrected me — `parse_fail` here means "≥1 parser choked, consensus held,"
  its reference still parser-derived. The finetune reaches 92.9% *fidelity to the parser*, never beats it.
  Testing LLM > parser needs new data: non-conforming input the decoders get wrong, vs. **human** gold.

**Next**

- **E4B** for the size-scaling curve (baseline → LoRA → same completion-path eval → McNemar), same recipe.
- **Later (user-requested):** (a) a messy-tail / human-gold eval that could actually test LLM > parser;
  (b) research scaling the corpus past 51 stations toward *all* stations (data volume, stratified sampling,
  storage, and what a global station set does to the messy-tail signal).

---

## Session 18 — 2026-07-30 · E4B finetune + the size-scaling curve (+ aviation-SME demo)

Ran the full E4B pipeline (real Gemma-4 E4B) with the identical E2B recipe — baseline → LoRA (r=16, 1 epoch,
3000 pairs) → GGUF → serve base+adapter → eval via `--completion` → McNemar — to turn one before/after into a
**size-scaling curve**. Also built an aviation-SME demo (`reports/gemma-4/finetuning_explainer.html`, published
as an Artifact).

**The curve (620 records, all served via `/completion`, same stack):**

| model | base value | FT value | base EM | FT EM | base halluc | FT halluc |
|-------|-----------:|---------:|--------:|------:|------------:|----------:|
| E2B | 69.5% | 92.9% | 2.4% | 38.2% | 31.0% | 0.8% |
| E4B | 81.4% | **98.3%** | 5.6% | **83.7%** | 15.6% | 0.4% |

McNemar (value): E2B 1437 fix / 3 regress, p≈10⁻³¹²; E4B 1041 fix / 5 regress, p≈10⁻²²⁴. Reference (single
python-metar vs the consensus): 97.5% / 82% EM. E4B-FT **edges a single parser** and sits just under the
3-parser+gold ceiling (~99.9%).

**Learned**

- **The finetune installs the *habit* at any size; the *precision* scales with capacity.** Both sizes learn to
  attempt every conversion (base did ~0%). But whole-record EM — which needs the arithmetic *exactly* right —
  jumps 38%→84% from 2B to 4B. The 2B converts approximately (`A2979` → "about 1014" not 1009); the 4B converts
  exactly. So the finetune's *lift* is large at both sizes, but its *ceiling* scales with the model. The single
  cleanest takeaway of the whole study.
- **A scaling curve must be served-consistently, and Gemma-4 makes that non-trivial.** E4B is a stronger reasoner:
  on the chat path (even with `--reasoning-budget 0`) it writes its reasoning into `content` and overruns the
  256-token cap before emitting JSON — 4/5 invalid. The raw `/completion` wrapper fixes it (direct JSON, 100%
  valid). Adopted `/completion` uniformly for all gemma-4 base+finetuned; **verified E2B base is path-invariant**
  (chat 69.5% ≈ completion 70.0%), so base-to-base comparison across sizes is valid.
- **The honest ceiling holds at 4B.** 98.3% is *fidelity to the parser consensus*, not superiority — every eval
  reference is parser/gold-derived, so the LLM can approach but not beat it. The LLM>parser question still needs
  the messy-tail / human-gold dataset.
- **Demo craft:** the SME explainer went through several rounds — no horizontal scroll (wrap table cells, reflow
  code), a chart grouped *by metric* (Base vs FT pairs, consistent colours, direction + delta), quiet inline
  asides instead of repeated callout boxes, and — for an SME audience — drop the "what is a METAR" hand-holding
  and show 3 full parsed-data decodes of the trickiest conversions (inHg, m/s, compound-fraction SM) with the
  base model's real wrong answers.

**Next** — (user-queued) the messy-tail / human-gold eval and the corpus scale-up past 51 stations. Optional
wider curve: the dense 12B or the 26B-A4B MoE rungs (bigger/slower) if the scaling trend is worth extending.
