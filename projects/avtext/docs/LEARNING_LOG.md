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

---

## Session 19 — 2026-07-31 · v2: 500-station corpus, 10x eval, rank-64 ablation

Rebuilt the study at scale: **490 geo-balanced stations** (US 74%→35% by volume, ~4% after per-station
stratification), a fresh frozen **eval/v2 = 6,200 records** (10x v1; 78 held-out stations spanning every
region; full 1,000 parse_fail), an 8,000-example v2 SFT, and the user's rank-64 test on E4B with a rank-16
ablation — all on the frozen v2 set. Corpus/eval/harness committed *before* results (Phase 5, cb2b6f6).

**Results (E4B, v2, 6,200 records):**

| model | value | halluc | EM |
|-------|------:|-------:|---:|
| base | 88.8% | 21.4% | 24.8% |
| rank-16 FT | 99.4% | 7.2% | 90.3% |
| rank-64 FT | 99.6% | 7.0% | 92.1% |

**Learned**

- **Rank has hit diminishing returns.** rank-64 vs rank-16: **+0.2% value, +1.8% EM, −0.2% halluc** — McNemar
  significant (p≈10⁻⁹) but tiny, and with real churn (193 fixes / 92 regressions on value). For **4x the
  adapter** (147M vs 37M params, 281 vs 71 MB) and the same training time, rank-64 buys ~1.8 EM points.
  **rank-16 is the deployment choice** — capacity is no longer the bottleneck for this task.
- **v1's low base was partly a US-sampling artifact.** Geo-balancing cut the US-specific inHg conversions
  (the base's main weakness), so v2 base = 88.8% vs v1's 81.4%. "Harder" is not one-dimensional: v2 is
  *heavier on the messy tail but lighter on conversions*. Always ask which axis "harder" moved.
- **Hallucination is now the residual, and it lives in the messy tail.** Even finetuned, v2 halluc = **7.0%**
  (vs v1's 0.4%). The 500-station diversity surfaced regional/exotic formats where the model still fabricates
  and the finetune can't fully suppress it. This is the honest cost of a representative eval — and the
  clearest pointer to the next work (abstention on unfamiliar formats).
- **Infra:** `--parallel > 1` corrupts E4B output on this llama.cpp/toolbox build (`<unused49>` garbage at
  parallel 4 and 8; only parallel 1 is clean) → the 6,200-record evals ran **sequentially, ~5.5h each**
  (~3.2s/record for the 8B). Geo-balanced selection via IEM per-country network geojson
  (`ingest/select_stations.py`); freezer + SFT builder parameterized with v1 defaults preserved (`--v2`);
  harness gained `--eval` / `--concurrency`.

**Next** — the residual **7% tail-hallucination** is the open problem (teach abstention on exotic formats,
maybe corruption augmentation). Product extension to TAF/SIGMET/NOTAM is scoped in
`docs/research/aviation-data-sources.md` — TAF is the cheap next win (IEM pipeline reuse + free 4-voice
answer key).

## Session 20 — 2026-08-01 · Hallucination anatomy + the whole TAF stack (all non-GPU work)

Two threads. (1) Dissected the v2 "7% hallucination." (2) Built the **entire TAF measurement
apparatus** end-to-end — corpus → oracles → consensus → gold → frozen eval → harness → SFT — mirroring
METAR's Phases 1–3 in one session. Standing decision from S19 now in force: **rank-16 only.**

**Hallucination anatomy** (`docs/research/hallucination-anatomy-v2.md`). Reclassified all 625 flagged
fabrications in the r16 v2 run (conservative test: only credit a "recovery" when the value is provably
in the raw). Result: **≥70% are mislabelled parser-defeat WINS** — 616/625 are on the `parse_fail` split
where gold is empty *because the parsers failed*, and the model read glued tokens (`15/14Q1016`) /
doubled ids (`MGHT METAR MGPB`) correctly. **Real fabrication is ≤2.1%, not 7%**, dominated by one crisp,
learnable mode: **slash-masked fields** (`///18G24KT`, `////`) the model fills in instead of nulling.
The scorer literally can't tell the model's best behaviour from its worst on `parse_fail` — the strongest
case yet for the human-gold messy-tail eval.

**TAF extension (US-only deep history is the trap).** IEM `taf.py` is **US/NWS-only** (EGLL/EDDF/UUEE/RJAA
all return 0) — the research doc's "global to 1996" was wrong for TAF. Chose US-only IEM for the stack;
then, for the geo-balanced non-US corpus the METAR lesson demands, built a one-time global collector from
the AWC bulk cache (`tafs.cache.xml.gz` — **XML, not CSV**) + a ~30-day backfill via the AWC Data API.
Collected **61,443 non-US TAFs** (2,824 stations, 30-day span, all ICAO regions); committed to git because
it is **not reproducible** (the API window slides).

**Learned**

- **Deep global TAF history does not exist for free.** IEM = US-only; AWC = global but current + **hard
  ~30-day ceiling** (verified: 30d returns data, 31d empty — *not* 90); Ogimet is the only deep global
  archive and is `robots: Disallow: /`. So global TAF = forward-collection + a 30-day catch-up. Always
  verify a "global/deep" claim endpoint-by-endpoint — the same trap as S17's gemma-3/4 conflation.
- **The voices disagree by CONVENTION, so align by sequence not timestamp.** avwx *resolves* change-group
  timing (`BECMG 0107/0109` → `0109/0117`), mivek *states* it (`1:7→1:9`). Aligning periods positionally
  (both list INITIAL→groups in raw order; 99% agree on count) and taking timing from the stated voice
  (mivek) sidesteps it. 2-voice consensus = **98% clean, 0% structural dissent**.
- **A "gold" voice can be worse than the consensus.** AWC's bundled decode is **SM-native**, so on the
  metres corpus it injects visibility round-trip noise (as a co-voter it *lowered* clean 99%→97%). Used it
  as the gold ANCHOR instead: consensus matches AWC **100%** on change_type/prob/wind/sky_clear/VV over
  1,375 records; the lower vis_m (69%) / cavok (90%) are AWC's own limits, where the consensus is the more
  accurate side. Measuring the "authority" before trusting it mattered.
- **The nested taxonomy is free.** score_taf aligns periods by sequence and reuses `score_field`: an
  OMITTED change group → every field ABSTAIN (safe), an INVENTED one → HALLUCINATE (dangerous). Whole-
  forecast EM is strict on purpose — structure is the point of TAF. A reference→SFT-target→parse→score
  round-trip scores **perfectly**, proving prompt/scorer/SFT-builder are mutually consistent.
- **eval/taf/v1 frozen:** 5,294 records (5,000 clean / 215 dissent / 79 parse_fail; sha256 f103708d),
  deterministic md5 15% station holdout (262/1,903), 5-day unseen_time window (corpus is only 30d deep).
  Clean-heavy by design. Harness validated on the real eval via a parser-consensus predictor (100%/0%/100%).
- **SFT:** `train_taf.jsonl`, 8,000 clean/dissent targets (== the exact reference), disjoint train split,
  640 MPS-wind examples, full 1–5+ period-complexity spread. Gitignored (reproducible from the committed corpus).

**Next (dashi only)** — serve base Gemma-4-E4B → TAF baseline (`runner_taf --completion`); train **rank-16**
LoRA on `train_taf.jsonl`; serve finetuned → harness; compare. Caveat: `--parallel 1` + ~1024-token nested
output makes the 5,294-record eval slow (~overnight) — consider `--limit` for a first read. Then the queued
METAR work: the masked-field abstention augmentation (from the hallucination anatomy) + the human-gold eval.

## Session 21 — 2026-08-03/04 · TAF finetune + combined METAR+TAF multi-task (on dashi)

Ran Phase 6's training half unattended on dashi: the single-task TAF finetune, then a combined
METAR+TAF adapter. **rank-16 throughout** (the standing decision).

**TAF single-task (E4B r16, eval/taf/v1, 1,500 records):**

| model | value | halluc | EM | period-match |
|-------|------:|-------:|---:|-------------:|
| base  | 83.6% | 11.9% | 7.2% | 87.4% |
| **r16 FT** | **93.1%** | **1.5%** | **89.1%** | 95.7% |

**Combined multi-task (ONE r16 adapter on 16k mixed examples, eval on both):**

| product | single-task r16 | combined r16 |
|---------|-----------------|--------------|
| TAF (same 1,500) | 93.1% / 89.1% EM | 93.2% / **90.1% EM** |
| METAR | 99.4% / 90.3% EM* | 99.7% / **93.1% EM** |

*single-task METAR on the full 6,200; combined on 1,500 — not strictly same-N.

**Learned**

- **A finetune's value on TAF is almost entirely STRUCTURE.** The base model reads individual TAF
  fields fine (83.6% value) but almost never gets a whole nested forecast right (7.2% EM) — it drops
  or garbles change groups. The finetune takes whole-forecast EM 7%→89% (+82 pts) and cuts
  hallucination 12%→1.5%. The nested task makes the base far worse than METAR (24.8% EM) and the
  finetune lift correspondingly larger.
- **Multi-task combining is FREE here — arguably slightly synergistic.** One rank-16 adapter matches
  both single-task adapters (TAF EM 89.1→90.1 on the airtight same-1,500 comparison; METAR no
  regression). Flat-JSON METAR and nested-JSON TAF don't interfere — the model routes on the prompt,
  and the shared aviation-decoding signal may help. Strong evidence that one adapter for all four
  products is viable.
- **This iGPU is slow for this workload; size the estimate empirically.** seq-2048 training ran
  ~10.3h (TAF 8k) / 17.5h (combined 16k); `--parallel 1` long-output eval ~7h per 1,500 TAFs. I
  repeatedly under-quoted ETAs.
- **Packing didn't pack (caution).** `SFTConfig(packing=True)` was accepted (dry-run ran clean) but
  the step count was unchanged (1,333 = full dataset) → no speedup. A 5-step "does it run" dry-run is
  NOT a test that packing reduced sequences — verify the step/epoch ratio dropped, not just that it ran.
- **Infra:** train + serve on dashi (`~/ft` unsloth env + serve.sh toolbox); harness runs there via a
  `uv` venv in `~/avtext`; `scripts/dashi/{phase_taf,phase_combined,chain_combined,train_lora}.py/.sh`
  orchestrate unattended (dashi-side chainer waits for one phase then launches the next — robust to
  clock skew / session loss). 73 tests.

**Next** — masked-field abstention augmentation (from S20's hallucination anatomy); the human-gold
messy-tail eval; and, on this evidence, extend the one-adapter approach toward SIGMET/NOTAM (all-four).

---

## Session 22 — 2026-08-05/08 · NOTAM harness + the all-products adapter (+ a serving bug)

**Done**

- Built the NOTAM runner (`harness/{runner_notam,models_notam}`) handling BOTH tasks (extraction via
  row-aligned scoring, classification via accuracy/macro-F1), the all-products SFT
  (`finetune/build_sft_all` → 29,248 mixed examples), and a benchmark **web app**
  (`report/{collect,dashboard}` → self-contained dashboard, published as an artifact).
- Ran the full base / single-task / all-products matrix on dashi for all four evals.

**Result — one adapter covers all three product families** (matches or beats every specialist):

| task | base | single-task | all-products |
|---|--:|--:|--:|
| METAR (EM) | 24.8% | 91.1% | **93.3%** |
| TAF (EM) | 7.2% | 89.1% | 89.9% |
| NOTAM extraction (EM) | 0.8% | 76.0% | 76.1% |
| NOTAM classification (acc) | 78.2% | 94.8% | **95.0%** |

**Learned**

- **Multi-task combining stays free at THREE product families / four tasks.** One rank-16 LoRA on
  29k mixed examples matches-or-beats each single-task specialist — no interference between flat-JSON
  METAR, nested-JSON TAF, category-keyed NOTAM extraction, and single-label NOTAM classification. The
  model routes on the prompt; the "one deployable adapter for everything" thesis holds.
- **NOTAM extraction's hard part is free-text, not structure.** Most categories hit 85–95% EM; the
  overall (76%) is dragged by `area` (huge, free-text `area_summary` → 51% value) and a genuinely
  hard tail (`airway`/`standard`/`procedure`). ~13.5% of outputs are malformed JSON on that free-text
  — "invalid-but-trying" beats base's "valid-but-useless" 0.8% EM.
- **⚠️ Serving degradation bug on the ROCm llama-server (the session's real lesson).** Long,
  heavy-generation eval runs deterministically collapse into `<unused49>` reserved-token spam after
  ~1000 records (first NOTAM ext run: 0% invalid in the first 200, rising to 100% by ~record 1000).
  **cache_prompt=false and flash-attn=off made ZERO difference** (bit-for-bit identical collapse) —
  only a **process restart** clears it. Fix: chunked eval (100 records/chunk + server restart between
  chunks; `eval_notam_chunked.sh` + `report/merge_notam`). Short-output tasks (classification,
  METAR/TAF) never trip it — the full 6,200-record METAR run confirms short gens are safe.
- **`--limit N` is invalid on an id-sorted eval** where the id starts with the category — the first
  1,500 NOTAM records were 73% `area` and 0% of 7 categories. Chunked runs use the FULL set.
- **Packing WORKED this time** (contra S21): `max_steps` 2,438 vs 4,875 unpacked = ~2× speedup on the
  29k mixed set, train_loss 0.078. S21's no-op packing was likely dataset/version-specific — always
  check the step/epoch ratio, but don't assume packing is dead.

**Next** — masked-field abstention augmentation (S20); human-gold messy-tail eval; Q4-quant survival
of the all-products adapter; publish the adapter (user's action). Root-cause the llama-server
`<unused49>` leak (newer build?) so long evals don't need chunking.

## Session 23 — 2026-09-07 · Reconstructing the lost dashi training env (gfx1151 from scratch)

**Done** — the finetune recipe was **gone** (dashi's `~/ft` tree wiped; the old unsloth
`train_lora.py` with it), so retraining was blocked until the whole env was rebuilt. Rebuilt it
end-to-end and **validated every link of the train→serve loop**:
- `~/fttorch` venv (uv, py3.12) + **TheRock gfx1151 torch** `2.12.0a0+rocm7.13` from
  `rocm.nightlies.amd.com/v2/gfx1151/` — the only torch with native Strix-Halo kernels.
- New `scripts/dashi/train_lora_peft.py` — standard `transformers`+`peft`+`trl` SFT (no unsloth),
  run inside toolbox `llama-rocm-7.2.4_2`. Smoke trained clean: 34.9M trainable (0.44%), loss fell,
  backprop ran on the 8060S iGPU, `SAVED_ADAPTER`.
- `convert_lora_to_gguf.py` turned that adapter into a valid **69.8M f16 GGUF** (516 tensors).
- Full recovery recipe written to `docs/DASHI_TRAINING_ENV.md` so this never costs a session again.

**Learned**
- **gfx1151 has no stock kernels.** Every pytorch.org ROCm wheel dies at the first matmul with
  `hipErrorNoBinaryForGpu` / "invalid device function". AMD's **TheRock per-arch nightly** is the
  only fix — and it's *native* gfx1151, so **no `HSA_OVERRIDE_GFX_VERSION`** is needed.
- **Run inside the toolbox, never the bare host.** The host lacks `libatomic.so.1` and doesn't wire
  `/dev/kfd`+`/dev/dri` into a random process; the kyuz0 `llama-rocm-7.2.4_2` container supplies both.
- **Two Gemma-4 multimodal traps.** (1) LoRA targets must be a **text-tower regex** — the vision/audio
  projections are `Gemma4ClippableLinear` (not `nn.Linear`) and PEFT refuses them. (2) Pass
  `processing_class=tok` so trl doesn't auto-load the PIL-requiring `Gemma4Processor`.
- **The feared conversion blocker was a non-issue.** Adapter keys carry the multimodal nesting
  (`...model.language_model.layers.N...`); `convert_lora_to_gguf` knows the `gemma4` arch and maps
  them to `blk.N...` itself. No key-stripping.
- **Host py3.14 is too new for torch** — the venv must be 3.12.

**Next** — the reason the env was rebuilt: **grow the data + retrain**. scp the local SFT sets to
dashi (its copies are gone), grow the METAR/TAF corpora reproducibly (IEM), retrain on the bigger
set + re-eval all families under `--grammar` for comparability. Then the **benchmark harness** (an
independent held-out set + a unified cross-family report card) and a new **SIGMET** family.
NOTAM licensed structural rebuild stays parked (no FAA key; no keyless source with Q-lines).

### S23 continued — data growth, a real leak, and the gfx1151 throughput wall

**Done** — with the env rebuilt, executed the "more data + retrain" plan and hit two findings.
- **Grew the per-family SFT sets** (METAR 8k→20k, TAF 8k→16k; `build_sft --v2 --max N` now respects
  an explicit cap). Grown all-products set = 49,214 leak-clean examples.
- **Leak audit found a real train/eval overlap.** Auditing the grown set against every frozen eval
  surfaced **34 NOTAM extraction training rows that were exact duplicates of eval/notam/v1** (the
  corpus `split` label missed them — dedup-before-split); the METAR builder also leaked **11 rows**
  into the *legacy* eval/v1 (v1 vs v2 holdouts differ). METAR/TAF/NOTAM-classification were exact-
  clean. Both builders now carry an exact-match eval blocklist, and a reusable gate
  (`finetune/leakcheck`) asserts zero record-level overlap before any run. The old NOTAM-ext EM
  (76.1) was mildly inflated by its 1.5% overlap; the leak-free adapter will be the honest number.
- Rebuilt the base Q8 GGUF (the `~/models/...` copy was wiped) — needed patching the fork's
  `convert_hf_to_gguf` to read gemma4's nested `text_config.global_head_dim`. Staged the full
  grown-adapter eval chain (`chain_lg.sh`) on dashi.

**Learned — the gfx1151 throughput wall (the session's hard lesson).** The reconstructed standard-
PEFT trainer is correct but **~10–30x slower than the lost unsloth path**; Phase 7 was fast only
because unsloth had fused flash-attn/varlen kernels this stack lacks.
- **`--packing` is catastrophic here**: no flash-attn → SDPA math does O(flattened_len²) attention
  → 310s/step. Non-packed + dynamic padding is the fix (keeps short rows short).
- **Bigger batch does NOT help** — the iGPU is compute/bandwidth-bound: batch 6 ≈ 40s/step, batch 32
  ≈ 136s/step (worse). Throughput ≈ 0.30 ex/s regardless. So a full 49k epoch ≈ **48 h** — a real
  ~2-GPU-day cost, not the "retrain now" the unsloth era implied.
- Practical overnight run = a **capped partial epoch** (`--max-steps 1200` ≈ 14k examples, ~13 h,
  grad-checkpointing ON for memory safety). Proves the reconstructed train→convert→serve→eval loop
  end-to-end on grown leak-clean data + gives preliminary numbers; a full-scale retrain is a
  deliberate GPU-days decision.

**Next** — when the capped run lands: convert→eval all four families (`chain_lg.sh`, NOTAM ext with
grammar), fold grown-vs-Phase7 into the dashboard, decide whether the full 49k run is worth ~2 days.
Then SIGMET (new family) and the independent held-out benchmark set.

## Session 24 — 2026-09-08 · Squeezing gfx1151: the step was padding-, fusion- and launch-bound

**Done** — asked "fastest possible LoRA training on this Strix Halo?"; answered with local
measurement first, web research second (three parallel passes, ~50 dated sources; report in
`docs/STRIX_HALO_TRAINING_SPEED.md`).
- Local ground truth: SDPA backend × head_dim × mask matrix; torch.profiler decomposition of a real
  step; sustained-clock/power under load; token-length + padding analysis; torch.compile A/B; Liger
  A/B; a no-root C compiler for Triton (`ziglang` pip wheel + a `zigcc` wrapper).
- Trainer levers shipped: `--group-by-length` (a `LengthGroupedSampler` subclass — trl 1.x dropped
  the flag), `--compile`, `--no-grad-checkpointing`, `--grad-accum`.
- Sweep 1 on real steps + a numerical-parity check; results committed into the report.

**Learned**
- **We were FLOP-starved by our own batches, not by the chip.** 49 % of linear compute was padding
  (random batches mix ~460-tok METAR with ~1100+-tok TAF). Length grouping → 1 % padding, ~2x.
  Profiler: ~35 % unfused elementwise + copies, ~12 % tiny rank-16 LoRA GEMMs (0.44 % of params),
  ~15k launches/microbatch, CPU ≈ GPU time. Raw bf16 matmul 27.9 TFLOP/s — the GEMMs were fine.
- **Composed on real steps: grouping 12.0 → +compile 11.8 → +no-checkpointing 7.0 s/step** (vs 40
  random). compile's static-shape 1.42x mostly vanishes under dynamic shapes (next: bucketed padding
  for static graphs); no-checkpointing is 1.7x on top and OOM-safe once batches are grouped.
- **Compiled loss matches eager to ~1 %, zero NaNs** — the Gemma-on-RDNA NaN hazard (unsloth
  disabled compiled fwd for Gemma-3) does not reproduce on Gemma-4 here. Always check; never assume.
- **The GPU is power-capped, not thermal:** 2175 of 2900 MHz at 88 W = the EVO-X2's default
  *Balanced* P-mode (85 W PL1). *Performance* (120 W PL1) sustains ~2787 MHz on this box model —
  a button press, +28 % clock, owner's action; watch >88 °C (reboots reported at 90).
- **head_dim 512 (Gemma-4's 7 global layers) has no fused attention on gfx11 by design**; and some
  2026 wheels return *silently wrong* values for it instead of MATH fallback — re-validate after any
  torch upgrade. Any padding mask also disables FLASH (no causal+bias kernel) — structural.
- **Liger-Kernel is 8x slower here** (fused-CE 6044 vs 762 ms): compute-for-memory is the wrong
  trade on a compute-bound, memory-rich box, and Triton codegen trails Tensile on RDNA 3.5.
- **Folklore corrected:** unsloth now officially supports gfx1151, but its own claim is ≤ ~1.4x vs
  TRL. Our "10–30x faster unsloth" memory compared against a mis-configured run (packing with no
  varlen attention). The honest gap vs a tuned standard stack is ~1.3–1.5x — mostly closed now.
- Measurement lessons: `LengthGroupedSampler`'s first megabatch is a *random* 300 sorted
  longest→shortest, not the 300 longest; tqdm's first-step s/it includes warm-up (64 s → 12 s);
  `toolbox run` swallows the rest of a piped script unless `</dev/null`; a function-local
  `import torch._dynamo` shadows module-level `torch` (UnboundLocalError).

**Next** — per-length cost curve → integrated epoch time (running); sweep 2 (`HIP_FORCE_DEV_KERNARG`,
`expandable_segments`, batch 12/24 *with* grouping); bucketed-padding compile; then the real grown-set
retrain overnight on config C (+ Performance mode). Frontier: flash-attn varlen packing, FlexAttention.

### S24 continued — the "page fault" was a memory blow-up in disguise; sweep 2; adaptive checkpointing

**Done**
- Root-caused the deterministic GPU fault at the ~1,130-token batch. It was ours, twice over: the
  probe scripts never called `model.train()` (`from_pretrained` returns eval mode; HF applies gradient
  checkpointing only when `self.training`), so every probe was silently a no-checkpointing run — and
  on an APU an over-allocation is not an OOM but a swap storm (GTT 119/124 GiB, host free ~1 GiB, GPU
  1 % busy, the process swapped to a 41 MB RSS) that ends in amdgpu `Page not present`. Same rows with
  `.train()`: 30 GiB allocated, three clean steps. The Trainer calls `.train()`; never affected.
- Guards: `--mem-fraction` (default 0.85 → a fast, retryable OOM instead of a hung box), a GTT-drain
  gate between GPU processes (a just-exited process still holds 80–95 GiB for tens of seconds;
  `gttwait` in `run_resilient.sh` and the sweeps), a sysfs GTT watchdog thread for probes.
- Sweep 2 (`steady2.py`, quantile-matched windows): eager no-ckpt **OOMs at step 0**; env knobs
  **1.13x** (6.2 s/step); batch 12 **nothing** (992 vs 983 ms/example). Re-reading sweep 1's per-step
  times showed config C had only "survived" the 2048 bucket by thrashing (110–200 s steps).
- Built **length-adaptive gradient checkpointing** (`--ckpt-above N`, flipped per microbatch inside
  `training_step`) + `--log-mem`; sweep 3 config H: 6.4 s/step = 533 ms/example on the short half (D: 6.2), the 2048-token head checkpointed at 23.7 GiB, and the compiled no-ckpt peak only **51 GiB at 1,059 tokens** (eager: >107 GiB).
- Epoch arithmetic from one megabatch minus the compile step: A ~25 h, D ~17 h, **H measured in production: 347 s per
  300-example megabatch = 15.8 h** at Balanced (~13 h with Performance P-mode), threshold 1,800 (97 % of
  rows no-ckpt) — 3.0x end to end; random-batch baseline was ~48 h.
- Launched the grown-set (49,214) production retrain on config H through `run_resilient.sh`
  (200-step checkpoints, eval chain `chain_lg.sh` chained after `SAVED_ADAPTER`).

**Learned**
- **There is no clean OOM on an APU.** The "GPU" pool is host RAM; the failure mode of too much
  memory is a swap storm and then a GPU page fault that looks like a driver/kernel bug. Cap the pool,
  gate launches on GTT drain, and read `mem_info_gtt_used` before believing any "hardware" fault.
- **Every standalone probe must call `model.train()`.** Three measurement artifacts in a row each
  looked like a hardware problem first: eval-mode checkpointing (the fault), lingering GTT after exit
  (the "fresh process OOMs"), and a progress-bar regex matching the dataset-map bar (negative s/step).
  The old per-length curve (23.7 h) was, by accident, the eager no-ckpt curve.
- **Compiled memory ≠ eager memory.** An eager no-ckpt forward needs >107 GiB at 1,130 tokens; the
  compiled graph fits 2,048 tokens (with `expandable_segments`) — inductor fusion removes the
  materialized intermediates. Memory thresholds must be measured under the production graph.
- **Bigger batch is not a lever on a saturated GPU** (E), and compile's fusion is what makes
  no-checkpointing fit at all (G vs D). The recompute trade is only worth paying on the long tail —
  hence per-length adaptive checkpointing, which nothing off the shelf offers.
- Batch-size comparisons need quantile-matched windows (megabatch = 50×batch, sorted longest→shortest).

**Next** — owner: Performance P-mode (+~20 %). Engineering frontier, in order of expected value:
bucketed static-shape compile (1.42x measured on a static step), flash-attn varlen packing,
FlexAttention. Study: SIGMET family; an independent held-out benchmark set; the post-retrain dashboard.

