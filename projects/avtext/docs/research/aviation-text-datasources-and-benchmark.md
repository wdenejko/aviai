# Aviation-Text LLM — Data Sources & a Benchmark You Can Trust *Without* Aviation Expertise

*Analysis · July 2026 · prepared for Wojtek · scope: aviation-text use cases — NOTAM, PIREP, SIGMET/AIRMET, METAR/TAF decode/extract/briefing, jargon expansion, grounded ops assistant, training/education (multilingual localization dropped from scope)*

---

## 0. The core worry, answered first

> *"The dataset and harness must be created without my aviation knowledge — I can't tell if the values are good or bad."*

**This is a solved problem, and it has a name: the *test-oracle problem*.** You don't need to become an aviation expert — you need to make correctness a property of **authoritative artifacts and machine checks**, never of your own judgment. The domain expertise already exists, encoded as data and published by regulators (FAA/ICAO/WMO/NOAA). Your job is to *wire it up*, not re-derive it.

Concretely, three things make this work in your favour:

1. **For most of the 9 use cases, the decoded "answer" ships *with* the data.** NOAA's Aviation Weather Center API returns raw METAR/TAF/PIREP/SIGMET *and* a decoded JSON structure for each. Iowa Environmental Mesonet ships raw + parsed columns. So the ground truth is bundled — you're not labeling anything.

2. **Where it doesn't ship, correctness is a table lookup.** A NOTAM carries its own **Q-code**; decode it with the official FAA Q-code table and the NOTAM has labeled *itself*. An abbreviation's meaning is one row in the FAA Contractions order. No judgment required.

3. **For the few generative outputs (e.g. a plain-language briefing), you check *faithfulness*, not taste.** Extract every fact from the model's prose and diff it against the authoritative decode — machine-checkable. Only fluency needs a human, and that's a *one-time paid reviewer on a small sample*, never you.

The rest of this doc is the machinery: a **trust ladder** that converts "I can't tell if it's good" into "I have a statistically-bounded, auditable correctness guarantee," the **authoritative artifacts** that define correctness, the **public datasets** (with licenses), a **per-use-case map**, and a **harness architecture** that fits your Airflow/ClickHouse stack.

**Framed in your language (data engineering):** this is schema validation + referential integrity + cross-source reconciliation + round-trip serialization tests + property-based testing — applied to aviation text. You already own every one of these patterns.

---

## 1. The Trust Ladder — how a non-expert gets a defensible ground truth

Layer cheap machine oracles over 100% of the corpus, then calibrate them with a tiny expert-validated seed. No single layer proves truth; each *rules out* a class of error. You (the non-expert) stay off the critical path at every rung.

| Tier | Oracle | Corpus coverage | Cost | What it proves | What it can't |
|---|---|---|---|---|---|
| **0** | **Round-trip + invariants + schema** | 100% | ~free | internal consistency, structural validity, physical plausibility | not *truth* (a plausible-but-wrong value passes) |
| **1** | **Cross-parser / cross-source consensus** | 100% | cheap | truth **iff sources are independent** | correlated bugs; "agreement with a buggy oracle" |
| **2** | **Borrowed-expertise gold seed** (official worked examples) | small | free | truth on the seed items | small, can be stale, misses long tail |
| **3** | **One paid expert pass** on a stratified sample | tiny | \$ | truth **+ a statistical bound on the rest** | nothing about strata you didn't sample |

**Tier 0 — Round-trip & invariants (zero domain knowledge).**
- *Round-trip:* `re-encode(decode(raw)) == raw` after canonicalization. The input *is* the oracle — no gold needed. This is a serialization property test / golden-master test. Any record that fails to round-trip goes to a "hard cases" queue (that's where the model earns its value).
- *Invariants (property-based testing):* encode spec rules as assertions — `dewpoint ≤ temperature`, `gust ≥ sustained wind`, `0 ≤ wind_dir ≤ 360 or VRB`, `CAVOK ⇒ no conflicting cloud/vis`, TAF `valid_end > valid_start`, present-weather token ∈ WMO code set, station id resolves in a registry. Implement as **Great Expectations** expectations or **dbt tests** (0 violating rows = pass). Runs on the whole corpus, deterministically.

**Tier 1 — Cross-parser consensus (differential testing).** Run ≥3 *independent* decoders (different codebases/languages to avoid shared bugs) plus NOAA's own decoded API/IWXXM as a reference. Agreement → consensus label; disagreement → flagged hard case. A weak-supervision label model (Snorkel/Dawid–Skene) denoises the panel and estimates each parser's reliability. Track panel agreement with Krippendorff's α as a data-quality KPI. **Pitfall to respect:** parsers that share a spec-misreading agree on a wrong answer — maximize independence, and never let "N agree" mean "true" without Tier 2/3.

**Tier 2 — Borrowed-expertise gold seed.** Transcribe raw→decoded worked examples straight from **FAA AC 00-45H**, **FMH-1**, the **NWS METAR decode key**, and **BoM reference cards**. You copy an authority's answer; you never make a call. Stratify by the taxonomy the domain already publishes (METAR field types × edge features; NOTAM Q-code families), size by the rule of three (below).

**Tier 3 — Minimal expert-in-the-loop, with residual-risk math.** One paid CFI/dispatcher/meteorologist reviews a *stratified random sample* (oversampling the Tier 1 disagreement queue). Use the **rule of three**: *0 errors found in n items ⇒ ~95% confidence the true error rate is < 3/n.* So **0 defects in 300 ⇒ error rate < 1%**, per stratum — a number you can put in a data-quality SLA. Budget scales with (#strata × per-stratum n), not corpus size. You make zero aviation judgments; you get a bounded guarantee.

**Silent-failure audit (are we just agreeing with a buggy oracle?):** (a) score the machine oracle against the Tier-2 seed; (b) mine items where many independent *models* agree *against* your gold — that often indicts the key, not the model (empirically shown on the Pre-Flight benchmark); (c) inject mutants with known answers (swap `KT`↔`MPS`, add a gust, corrupt a Q-code) and confirm the oracle reacts — an oracle that stays silent on a planted error is silently failing.

---

## 2. Authoritative artifacts — where "correct" is defined by an official file

These are the backbone. Split into **Tier A (machine-readable data — correctness *is* the file)** and **Tier B (expert-authored prose with worked examples — transcribe once)**.

### Tier A — machine-readable authoritative data

| Artifact | What it defines | Format | License | URL |
|---|---|---|---|---|
| **WMO Codes Registry** — table **4678** (aerodrome present weather), cloud/colour tables under `49-2` | METAR/TAF token vocabulary | RDF/Turtle/JSON-LD/CSV (content-negotiated) | *Content license unstated — flag*; use CCT instead for reuse | codes.wmo.int/306/4678 |
| **wmo-im/CCT** — WMO Common Code Tables | Same tables, versioned flat files | **CSV** (+XML) | **MIT** ✅ | github.com/wmo-im/CCT |
| **wmo-im/iwxxm** — official XML data model | *Structural* validity of decoded METAR/TAF/SIGMET/AIRMET + raw↔structured example files | XSD + Schematron + examples | Open (WMO) | github.com/wmo-im/iwxxm |
| **FAA JO 7930.2 Appendix B — NOTAM (Q) Codes** | 2-letter SUBJECT + 2-letter CONDITION decode → **self-labels every ICAO NOTAM** | HTML (scrapeable) | **Public domain** ✅ | faa.gov/…/notam_html/appendix_b.html |
| **FAA JO 7340.2P — Contractions** | Abbreviation → meaning dictionary | HTML/PDF | **Public domain** ✅ | faa.gov/…/cnt_html/ |
| **FAA Pilot/Controller Glossary** | Term → definition | HTML/PDF | **Public domain** ✅ | faa.gov/…/pcg_html/ |
| **FAA NASR (28-day)** | Referential integrity: airports/runways/navaids/fixes exist | CSV / fixed-width / AIXM 5.1 / shapefile | **Public domain** ✅ | faa.gov/…/NASR_Subscription/ |
| **OurAirports** / **mwgg/Airports** | Global ICAO/IATA identifier + runway registry | CSV / JSON | **CC0** / open ✅ | ourairports.com/data/ |

> **ICAO Doc 8400 (abbreviations), Doc 8126/PANS-AIM (NOTAM), Annex 3, WMO-306** are the *de jure* source of truth but **paywalled**. For a machine-readable, redistributable pipeline, the **FAA public-domain equivalents above cover the same ground** — use them as your anchors and cite the ICAO docs only for adjudicating edge cases.

### Tier B — expert-authored worked examples (borrowed-expertise gold seed)

| Source | Gives you | Format | License |
|---|---|---|---|
| **FAA AC 00-45H — Aviation Weather Services** | Paired raw→decoded examples for METAR/TAF/PIREP/SIGMET/AIRMET + code tables | PDF | Public domain ✅ |
| **FMH-1 (2019)** — Federal Meteorological Handbook No. 1 | US METAR coding spec + present-weather/sky/remarks tables + examples | PDF | Public domain ✅ |
| **NWS METAR decode key** | One-page field-by-field annotated METAR | PDF | Public domain ✅ |
| **BoM / NAV CANADA / UK Met Office decode cards** | International-format worked examples | PDF | National-gov copyright — *flag reuse* |

---

## 3. Public datasets — what to actually pull

**Licensing headline:** US federal works are public domain (17 U.S.C. §105) → FAA + NOAA/NWS/AWC raw content is **freely usable for training and redistribution**. The crown jewels are the sources that ship **decoded structure alongside the raw** — your ground truth comes for free.

### METAR / TAF

| Source | Ships decoded GT? | Coverage | License | Notes |
|---|---|---|---|---|
| **AWC Data API** (`/api/data/metar`,`/taf`) | **✅ yes** (JSON/CSV/XML/IWXXM) | Worldwide, **live 15 days** | Public domain ✅ | Run a collector to accumulate history. ≤100 req/min. |
| **Iowa Env. Mesonet (IEM) ASOS** | **✅ yes** (raw + parsed columns) | Global, ~1995→present, huge | *"Any lawful purpose incl. commercial"* ✅ | **Best open bulk historical.** No key. |
| **NCEI ISD** | ✅ decoded elements | 1901→present, 35k stations, AWS Open Data | Public domain, **but WMO Res.40** limits some foreign obs | Long-horizon; mind redistribution of foreign stations. |
| Ogimet | ❌ raw only | METAR from ~2005 | Informal/unofficial — *flag ToS* | Raw mirror; scrape gently. |

### NOTAM

| Source | Ships GT? | Coverage | License | Notes |
|---|---|---|---|---|
| **FAA FNS / SWIM (Digital NOTAM)** | Partial (**AIXM structured** fields + raw) | All active US, real-time | Public domain content ✅ (free SWIM registration) | Digitized NOTAMs carry structured location/schedule/geometry. |
| **FAA NOTAM Search** | Raw + **Q-code** (self-label) | US + ICAO, current | Public domain ✅ | Unofficial JSON endpoint (scrape); no bulk export. |
| **Pik 2023 (Zenodo)** | ✅ georeferenced metadata + geometry | **264,365 NOTAMs** + GPS + traffic | **CC-BY-4.0** ✅ | Strong open, redistributable structured set. |
| **Knots** (github Estrellajer/Knots) | ✅ **expert gold** (per-Q-code fields, α=0.96) | ~10.6k in repo (paper says 12,347) | **Ambiguous** — Apache badge, *no LICENSE file* | Best expert NOTAM gold — **treat eval-only until authors confirm license**; China-skewed, multilingual free-text. |

### PIREP

| Source | Ships GT? | Coverage | License |
|---|---|---|---|
| **AWC PIREP API** | **✅ decoded JSON** (loc, FL, aircraft, turb, icing, sky) | US + N. Atlantic, live 15 days | Public domain ✅ |
| **IEM parsed/geocoded PIREP archive** | **✅ parsed + geocoded** | US, multi-year | *Any lawful purpose incl. commercial* ✅ |

### SIGMET / AIRMET / G-AIRMET

| Source | Ships GT? | Coverage | License | Notes |
|---|---|---|---|---|
| **AWC airsigmet API** | **✅ decoded** (hazard, severity, FL band, movement, polygon, IWXXM) | Worldwide SIGMET; Alaska AIRMET, live 15 days | Public domain ✅ | — |
| **AWC gairmet API** | **✅ inherently structured** | CONUS | Public domain ✅ | **CONUS textual AIRMET was discontinued Jan 2025 → G-AIRMET (structured) only.** |

### Restricted — do **not** build public/training sets from these

EUROCONTROL NM B2B (PKI-gated), ICAO API Data Service (paid), ICAO/WMO paywalled docs, commercial APIs (CheckWX, AVWX Pro, Cirium), and the ACL-2025 NOTAM-translation corpus (SWIM-restricted, IDs-only, eval-only).

---

## 4. Per-use-case map — data × ground truth × how a non-expert validates it

The key column is **Validatability**: how much of the quality judgment can be made by machine, with no aviation expertise from you.

- ✅ **Machine-checkable** — deterministic authoritative anchor; you can fully trust the number.
- 🟡 **Mostly machine-checkable** — decoded GT + invariants cover the fields; free-text tail needs a small seed.
- ⚠️ **Generative** — no deterministic gold; validate *factual faithfulness* by machine, *fluency* by a one-time paid reviewer.

| # | Use case | Primary data | Ground-truth anchor (correctness defined by…) | Validatability | Residual gap → how to close without your expertise |
|---|---|---|---|:--:|---|
| 1 | **NOTAM decode/classify** | FAA FNS/SWIM, NOTAM Search, Pik (CC-BY), Knots (eval-only) | The NOTAM's **own Q-code** via FAA Appendix B (self-label) + contractions dict + NASR referential integrity | ✅ (classify) / ⚠️ (prose summary) | Summary prose → faithfulness check vs decoded fields; fluency → paid sample |
| 2 | **Aviation-text → structured** | **AWC API decoded JSON** (all 4 types), IEM parsed | The decoded JSON that **ships with the data** + IWXXM XSD structural validity + cross-parser consensus | ✅ **best case** | Almost none — GT is bundled. Audit oracle vs Tier-2 seed |
| 3 | **Jargon / Q-code / abbrev expansion** | FAA 7340.2P, PCG, Q-code table | The **official dictionary itself** (exact-match lookup) | ✅ **fully** | None — pure table lookup, zero judgment |
| 4 | **PIREP parsing** | **AWC PIREP API** (decoded), **IEM parsed/geocoded** | Two independent decoded sources cross-checked + FL/hazard vocab invariants | 🟡 | Free-text remarks → invariants + small seed |
| 5 | **SIGMET/AIRMET decode** | AWC airsigmet/gairmet (decoded + geometry + IWXXM) | AWC decoded + IWXXM schema + polygon/FL invariants | 🟡→✅ | G-AIRMET already structured; note 2025 textual-AIRMET retirement |
| 6 | **METAR/TAF messy-tail decode + briefing** | AWC API, IEM (raw+parsed), NCEI ISD | **Triple-parser consensus** (metaf ∧ avwx ∧ mivek) + WMO 4678/cloud tables + AC 00-45H seed + round-trip + invariants | ✅ (decode) / ⚠️ (briefing) | Briefing → extract facts, diff vs parser (machine); fluency → paid sample |
| 7 | **Grounded ops assistant** | Your MCP hub (Confluence/Databricks/GitHub) + Pre-Flight (MIT), ERAU (eval-only) | **Citation-grounding**: answer must be supported by the retrieved doc (NLI/faithfulness check) + MCQ accuracy on Pre-Flight/ERAU | ⚠️ (faithfulness machine-checkable) | Open-ended helpfulness → paid rubric review on a sample |
| 9 | **Training/education exercises** | Any raw report + authoritative decoder | **Self-labeling**: question from raw, **answer key = authoritative decode** (machine-derived) → grading is exact-match | ✅ (gen + grading) | Pedagogical quality ⚠️ → light educator review |

**The pattern to notice:** most in-scope tasks are ✅/mostly-✅ *because the decoded structure ships with the data, or the answer is a published table*. The 2 generative ones (briefing #6, assistant #7) are still **largely** expertise-free because you can machine-check the *facts* via faithfulness diffing; only *fluency* needs a human, once, on a sample.

---

## 5. The benchmark harness — one pipeline, Airflow-shaped

This doubles as your **before/after model eval**: the same gold + same harness scores the base E4B, a frontier baseline, and the fine-tuned E4B (identical prompts, temperature 0, held-out post-cutoff split).

```
 ┌─ INGEST ──────────────────────────────────────────────────────────────┐
 │ Pull raw METAR/TAF/PIREP/SIGMET/NOTAM (AWC API + IEM + FNS/SWIM).       │
 │ Continuously snapshot AWC (15-day window) to build history + a          │
 │ POST-Jan-2025 held-out split (contamination-resistant test set).        │
 └───────────────────────────────────────────────────────────────────────┘
        │
 ┌─ TIER 0: round-trip + invariants (Great Expectations / dbt) ──────────┐
 │ re-encode(decode(raw)) == raw ; physical + vocab + refint checks.       │
 │ Non-bijective / invalid  →  HARD-CASE QUEUE (do not drop — this is       │
 │ where the fine-tuned model must prove value).                           │
 └───────────────────────────────────────────────────────────────────────┘
        │
 ┌─ TIER 1: fan-out to ≥3 independent decoders + AWC/IWXXM reference ─────┐
 │ Normalize to one schema → weak-supervision label model → consensus      │
 │ label + per-parser reliability + DISAGREEMENT QUEUE.                     │
 └───────────────────────────────────────────────────────────────────────┘
        │
 ┌─ TIER 2: borrowed-expertise gold seed (AC 00-45H / FMH-1 / NWS key) ───┐
 │ Stratified by published taxonomy; calibrate the oracle against it.       │
 └───────────────────────────────────────────────────────────────────────┘
        │
 ┌─ SILENT-FAILURE AUDIT ────────────────────────────────────────────────┐
 │ oracle-vs-seed agreement; model-consensus-vs-key mining; mutant inject. │
 └───────────────────────────────────────────────────────────────────────┘
        │
 ┌─ TIER 3: one paid expert pass on a stratified + disagreement sample ───┐
 │ Rule of three → residual error bound per stratum (0/300 ⇒ <1%).         │
 └───────────────────────────────────────────────────────────────────────┘
        │
 ┌─ MODEL EVAL (before/after) ───────────────────────────────────────────┐
 │ Field micro/macro-F1, whole-record exact match, JSON/IWXXM validity,    │
 │ round-trip consistency, faithfulness (prose facts vs decode),           │
 │ hallucination/abstention rate. Bootstrap CIs + McNemar paired test.     │
 │ External held-out: Pre-Flight (MIT), ERAU (eval-only), Knots (eval-only).│
 └───────────────────────────────────────────────────────────────────────┘
```

**Tooling that maps to your stack:** Great Expectations / dbt for Tier 0 gates; the decoders (`metaf`, `avwx-engine`, `metar-taf-parser`, plus `PyNotam` for NOTAM — note **GPL-2.0**, keep it as an offline oracle) as Tier-1 labeling functions; Snorkel or a plain Dawid–Skene model for consensus; JSON Schema / IWXXM XSD for structural validity; Airflow to orchestrate; ClickHouse to store per-field results and power the eval dashboards.

---

## 6. Licensing cheat-sheet (train + redistribute vs eval-only)

| Can train **and** redistribute | Eval-only / restricted — do **not** train or redistribute |
|---|---|
| AWC API (METAR/TAF/PIREP/SIGMET/AIRMET, decoded) — **public domain** | **Knots** NOTAM gold — license ambiguous (no LICENSE file) → *eval-only until authors confirm* |
| IEM (METAR + PIREP, raw+parsed) — *any lawful purpose incl. commercial* | **ERAU** weather question bank — **CC BY-NC-ND** (eval key OK; no commercial, no derivatives) |
| FAA (NOTAM raw, Q-code table, contractions, PCG, NASR, AC 00-45H) — public domain | ACL-2025 NOTAM translation — SWIM-restricted, IDs-only |
| NCEI ISD — public domain *(WMO Res.40 caveat on foreign obs)* | EUROCONTROL NM B2B, ICAO API Data Service, ICAO/WMO paywalled docs |
| Pik/Zenodo NOTAM — **CC-BY-4.0** | Commercial APIs (CheckWX, AVWX Pro, Cirium) |
| wmo-im/CCT (MIT), OurAirports (CC0), most parsers (MIT) | **PyNotam (GPL-2.0)** — usable as an offline oracle; copyleft if you *distribute* it |
| Pre-Flight (MIT) | codes.wmo.int table *content* — license unstated → prefer CCT (MIT) |

---

## 7. Honest limits & risks

1. **Correlated-oracle blind spot.** If all your parsers share a spec-misreading, consensus launders the error. Mitigation: genuinely independent codebases + the Tier-2 seed + mutant injection. This is *the* residual risk in an expertise-free pipeline — name it and bound it, don't pretend it's gone.
2. **Free-text NOTAM/RMK semantics.** The genuinely free-text tail (field E prose, plain-language remarks) is where no table helps — this is exactly the fine-tune's *target*, and exactly where you most need the Tier-3 sample.
3. **Knots caveats:** repo (~10.6k) vs paper (12,347) count gap; missing LICENSE file; source-skewed sample. Great for eval, license-gate before training. Open an issue to confirm.
4. **Contamination:** raw METARs are in the base model's pretraining (Jan-2025 cutoff). Draw the headline test set from **post-cutoff** reports so the "before" number isn't inflated.
5. **AWC 15-day window:** history requires you to run a collector now, or lean on IEM/NCEI for backfill.

---

## 8. Recommended first build

1. **Start with METAR/TAF extraction (#2 + #6 decode).** Best data (AWC decoded + IEM), strongest anchors (triple-parser + WMO tables + AC 00-45H), fully ✅ validatable. This *proves the harness* end-to-end on the easiest-to-trust task.
2. **Bolt on the Trust Ladder** (Tier 0→3) and lock the residual-risk report format. Now you have an auditable "we can trust these numbers" story before any model training.
3. **Add NOTAM (#1) via Q-code self-labeling** + contractions (#3) — highest value, and the Q-code makes it self-supervising.
4. **Then PIREP (#4) and SIGMET/AIRMET (#5)** — both ship decoded GT.
5. **Only after the harness is trusted, run before/after** on base E4B → frontier → fine-tuned E4B.

---

## Sources

**Datasets & APIs**
- AWC Data API — https://aviationweather.gov/data/api/ · IEM METAR — https://mesonet.agron.iastate.edu/info/datasets/metar.html · IEM PIREP — https://mesonet.agron.iastate.edu/request/gis/pireps.php · NCEI ISD (AWS) — https://registry.opendata.aws/noaa-isd/
- FAA FNS/SWIM client — https://github.com/faa-swim/fns-client · FAA NOTAM Search — https://notams.aim.faa.gov/notamSearch/ · FAA NASR — https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/
- Zenodo NOTAM (Pik 2023, CC-BY) — https://zenodo.org/records/11420433 · Knots — https://github.com/Estrellajer/Knots · https://arxiv.org/abs/2511.12630

**Authoritative anchors**
- WMO Codes Registry 4678 — https://codes.wmo.int/306/4678 · wmo-im/CCT (MIT) — https://github.com/wmo-im/CCT · wmo-im/iwxxm — https://github.com/wmo-im/iwxxm
- FAA NOTAM Q-codes (Appendix B) — https://www.faa.gov/air_traffic/publications/atpubs/notam_html/appendix_b.html · FAA Contractions 7340.2P — https://www.faa.gov/air_traffic/publications/atpubs/cnt_html/ · FAA PCG — https://www.faa.gov/air_traffic/publications/atpubs/pcg_html/
- FAA NASR — https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/ · OurAirports — https://ourairports.com/data/
- FAA AC 00-45H — https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_00-45H.pdf · FMH-1 (2019) — https://www.icams-portal.gov/resources/ofcm/fmh/FMH1/fmh1_2019.pdf · NWS METAR decode key — https://www.weather.gov/media/wrh/mesowest/metar_decode_key.pdf

**Parsers (oracles)**
- avwx-engine (MIT) — https://github.com/avwx-rest/avwx-engine · metaf (MIT) — https://github.com/nnaumenko/metaf · metar-taf-parser — https://github.com/aeharding/metar-taf-parser · PyNotam (GPL-2.0) — https://github.com/slavak/PyNotam

**Methodology & eval assets**
- Metamorphic testing / oracle problem — https://en.wikipedia.org/wiki/Metamorphic_testing · Hypothesis encode/decode invariant — https://hypothesis.works/articles/encode-decode-invariant/ · Snorkel (weak supervision) — https://arxiv.org/abs/1711.10160
- Great Expectations — https://docs.greatexpectations.io/ · dbt tests — https://docs.getdbt.com/docs/build/data-tests · Rule of three — https://www.statology.org/a-concise-guide-to-the-statistical-rule-of-three/ · LQAS — https://en.wikipedia.org/wiki/Lot_quality_assurance_sampling
- Pre-Flight (MIT) — https://huggingface.co/datasets/AirsideLabs/pre-flight-06 · https://arxiv.org/html/2607.01829v1 · ERAU weather question bank (CC BY-NC-ND) — https://commons.erau.edu/ga-wx-display-interpretation/17/
