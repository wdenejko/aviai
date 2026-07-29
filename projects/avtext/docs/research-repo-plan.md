# Research Repo Plan — Aviation-Text LLM Lab

*July 2026 · for Wojtek · companion to `metar-taf-finetune-analysis.md` and `aviation-text-datasources-and-benchmark.md` (put both in `docs/research/`)*

---

## 0. Framing: this is a learning project

The point is not to ship a model — it's to learn three crafts end to end, with a real domain as the vehicle:

| Skill | Where you learn it | Your starting point |
|---|---|---|
| **A. Building ML datasets with automated acquisition** | Phase 1 | Strong (your day job) — kept deliberately lean |
| **B. Building a benchmark/eval harness you can trust** | Phase 2–3 | New — the deepest learning here |
| **C. Fine-tuning a small LLM (LoRA/QLoRA) properly** | Phase 4–5 | New |

Two design consequences:

1. **Thin vertical slice first.** METAR/TAF only, ~50 stations, one task family — taken all the way from download → harness → baseline → finetune → before/after report. Only then widen to NOTAM/PIREP/SIGMET. A working end-to-end loop teaches more than a complete dataset for a loop that doesn't exist yet.
2. **Hand-roll the learning targets, buy the rest.** Write the scorers, stats, and consensus labeler yourself (that's skill B). Don't hand-roll parsers, downloads, or training loops (use avwx/mivek, curl, Unsloth/TRL).

Standing constraints from the prior analyses: correctness comes from the **trust ladder** (round-trip + invariants → cross-parser consensus → official gold seed → one paid expert pass), never from your own aviation judgment; no weather-*prediction* training targets; test set drawn **post-Jan-2025** (E4B's data cutoff) with **station+time** splits.

---

## 1. Repo blueprint

Name it something neutral to the whole aviation-text scope (`avtext-lab`, `skydecode`) — not `metar-*`, since NOTAM/PIREP arrive later.

```
avtext-lab/
├── README.md                  # research question, method, current results table
├── LICENSE                    # Apache-2.0 (code)
├── DATA_LICENSES.md           # per-source terms; what may NOT be redistributed
├── pyproject.toml             # uv-managed; pinned deps
├── Makefile                   # make backfill / collect / harness / report
├── .github/workflows/
│   ├── ci.yml                 # pytest + lint + 10-item smoke-score on fixed outputs (no GPU)
│   └── collect.yml            # cron collector (see §2.2)
├── configs/                   # stations.yaml, eval_v1.yaml, train_run_*.yaml
├── src/avtext/
│   ├── ingest/                # iem.py, awc.py, notam.py, references.py, zenodo.py
│   ├── schema/                # pydantic canonical records (MetarRecord, TafRecord, …)
│   ├── oracles/               # adapters: python-metar, metar-taf-parser, avwx-engine
│   ├── quality/               # tier0: round-trip, invariants (Hypothesis + pytest)
│   ├── consensus/             # tier1: majority vote → later Dawid–Skene
│   ├── tasks/                 # eval-set builders (decode→JSON, flight-cat, jargon, …)
│   ├── harness/               # runner, scorers, stats, report
│   └── finetune/              # SFT data prep, train configs, GGUF export
├── data/                      # GITIGNORED except manifest
│   ├── manifest.json          # committed: URL, sha256, fetch time per artifact
│   ├── raw/ reference/ processed/ gold/
│   └── third_party/           # Knots, ERAU — EVAL-ONLY quarantine, never redistributed
├── eval/v1/                   # frozen eval sets (JSONL + content hash) — committed
├── reports/runs/<run_id>/     # results JSON + markdown report — committed
├── docs/
│   ├── research/              # the two analysis docs
│   ├── adr/                   # one ADR per real decision (you know this drill)
│   └── LEARNING_LOG.md        # what you learned per session — the actual deliverable
└── tests/
```

Conventions that make it a *research* repo:

- **Reproducibility contract:** every run report records model hash, adapter hash, prompt-template id, decode params, eval-set hash. Anyone (future-you) can re-run any number.
- **Data never in git** (except tiny gold seeds, manifests, frozen eval JSONL). Bulk lives in a **Hugging Face dataset repo** (free, versioned, made for this) or your GMKtec home server (`dashi`) + rclone. `make backfill` must rebuild `data/` from nothing.
- **Experiments are issues, decisions are ADRs**, learnings go to `LEARNING_LOG.md`. Tag repo states at milestones (`v0-slice`, `v1-benchmark`, `v2-finetune`).

---

## 2. Next step 1 — automated data acquisition

Two mechanisms, both in-repo. **Start the collector on day 1** — it accumulates the decoded ground truth that the 15-day AWC window would otherwise lose while you build everything else.

### 2.1 One-off backfills — `make backfill` (idempotent, checksummed)

| # | What | Source / endpoint | Script | Slice scope |
|---|---|---|---|---|
| 1 | **METAR history, raw strings** | IEM `cgi-bin/request/asos.py` (station, date range, `data=metar`); discover stations via IEM network GeoJSON (e.g. `PL__ASOS`, US networks) | `ingest/iem.py` | ~50 stations × 3 yr: mix US/EU (incl. EPGD), coastal/mountain/AUTO-heavy for messy tails |
| 2 | **TAF + SIGMET/AIRMET raw history** | IEM AFOS text archive (by product PIL); AWC archive tool for spot checks | `ingest/iem.py` | Same stations |
| 3 | **PIREP parsed history** (for later) | IEM `request/gis/pireps.py` (CSV, geocoded) | `ingest/iem.py` | Defer until slice done |
| 4 | **NOTAM bulk** (for later) | Zenodo Pik 2023 (`11420433`, CC-BY, 264k georeferenced NOTAMs) | `ingest/zenodo.py` | Defer |
| 5 | **Reference pack** (trust anchors) | FAA Q-code Appendix B (HTML→CSV), FAA Contractions 7340.2 (HTML→CSV), PCG, NASR 28-day zip, OurAirports CSVs, `wmo-im/CCT` clone (MIT), `wmo-im/iwxxm` schemas | `ingest/references.py` | Day 1 — cheap, tiny |
| 6 | **Gold-seed source PDFs** | AC 00-45H, FMH-1 (2019), NWS METAR decode key | `ingest/references.py` | Day 1 (transcription happens in Phase 2) |
| 7 | **External eval sets** | HF `AirsideLabs/pre-flight-06` (MIT); clone Knots + ERAU into `third_party/` eval-only | `ingest/references.py` | Day 1; open the Knots license issue now |

Pattern per script: fetch → verify/record sha256 in `manifest.json` → write immutable `data/raw/{source}/{product}/dt=…/….gz` → normalize to Parquet in `data/processed/` (DuckDB — keep the research repo self-contained; ClickHouse adds nothing at this scale and you already know it).

### 2.2 Scheduled collector — `.github/workflows/collect.yml` (cron, every 6 h)

The AWC caches carry **raw + officially decoded** current data but only ~15 days of history. A tiny cron job turns that into an ever-growing labeled corpus — and your post-cutoff test pool:

```yaml
on:
  schedule: [{cron: "17 */6 * * *"}]   # offset minute; be a polite client
jobs:
  collect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |            # custom User-Agent per AWC ToU; gzip snapshots kept verbatim
          python -m avtext.ingest.awc snapshot \
            --products metar,taf,airsigmet,gairmet,pirep --out snapshots/
      - run: |            # push to a HF dataset repo — not into git (bloat)
          huggingface-cli upload $HF_DATASET snapshots/ raw/awc/$(date -u +%F-%H) \
            --token ${{ secrets.HF_TOKEN }}
```

Products: `metars.cache.csv.gz`, `tafs.cache.xml.gz`, `airsigmets.cache.xml.gz`, `gairmets.cache.xml.gz`, plus `/api/data/pirep`. Add the FAA NOTAM Search snapshot later (unofficial endpoint — polite rates; FNS/SWIM subscription is the proper stretch goal). A weekly consolidation job dedupes on `(station, obs_time, sha1(raw))` → Parquet. Fallback if GH Actions annoys you: same script as a systemd timer on your GMKtec home server (`dashi`).

**You'll learn (A):** designing an acquisition layer for ML — immutable raw zone, manifests/checksums, dedupe keys, license bookkeeping — and why eval data gets collected *before* it's needed.

---

## 3. Next step 2 — the test harness

Build order matters: **schema → oracles → checks → gold → eval sets → runner → scorers → stats → report.** Each stage is testable without the next; the harness is *finished and frozen before any training starts*.

1. **Canonical schema** (`schema/`) — pydantic v2 models for a decoded METAR/TAF. Write it by reading FMH-1/AC 00-45H tables, not by copying a parser's output model. *This step is where you actually learn the format.*
2. **Oracle adapters** (`oracles/`) — three independent, pip-installable parsers behind one interface: `python-metar` (`metar`), mivek's `metar-taf-parser`, `avwx-engine`. Each maps into your canonical schema; adapter bugs are found by the next two layers.
3. **Tier-0 checks** (`quality/`) — Hypothesis round-trip property tests (`re-encode(decode(raw)) ≈ raw` after canonicalization) + the invariant suite (dewpoint ≤ temp, gust > wind, vocab ∈ WMO/CCT tables, station ∈ registry, TAF period sanity). Failures route to the **hard-case queue** — never dropped; that's the finetune's target material.
4. **Tier-1 consensus** (`consensus/`) — v1: field-level majority vote across the 3 oracles + the AWC decoded JSON as a 4th voice; disagreement → hard-case queue. v2 (later): Dawid–Skene, and compare — a perfect small learning experiment. Track panel agreement (Krippendorff's α) as a KPI.
5. **Gold seed** (`data/gold/`) — transcribe 100–200 worked examples from AC 00-45H / FMH-1 / NWS decode key (copy-only, zero judgment), stratified by field type × edge feature. Calibrate the consensus oracle against it; run the mutant-injection audit (plant known errors, confirm the oracle catches them).
6. **Eval-set builders** (`tasks/`) → frozen `eval/v1/*.jsonl` with content hashes. Slice tasks: raw→JSON decode (clean), raw→JSON (hard-case/messy), flight-category, plus held-out external sets (Pre-Flight, ERAU eval-only). Split discipline **enforced in code**: test stations never in train, test time-window strictly post-cutoff.
7. **Runner** (`harness/`) — thin: model-in (HF checkpoint via transformers on GPU, **or GGUF via llama.cpp on your MacBook** — E4B Q4 is ~4–5 GB and runs fine on Apple Silicon), frontier API for the reference baseline. Temperature 0, fixed prompt templates, JSON-schema-constrained decoding as a togglable flag (llama.cpp GBNF / outlines — a built-in constrained-decoding lesson).
8. **Scorers + stats** (`harness/`) — hand-rolled on purpose: per-field exact match, micro/macro-F1, whole-record EM, JSON validity, hallucination rate (fabricated value where gold = null), abstention correctness. Stats: bootstrap CIs + McNemar paired test (scipy/statsmodels). Report generator → `reports/runs/<id>/report.md` with the reproducibility block.

Definition of done: `make harness MODEL=google/gemma-4-E4B-it` produces a scored, CI'd report from a clean clone, twice, identically. *Then* look at `inspect-ai`/`lm-evaluation-harness` and compare their design choices with yours — instructive in both directions.

**You'll learn (B):** eval design as engineering — oracle problem, leakage, contamination, metric decomposition, statistical honesty. This phase is the heart of the project.

---

## 4. Next step 3 — the finetune

### Environment (pick per run, cheap either way)

| Option | Fit | Notes |
|---|---|---|
| **Colab T4 (free) + Unsloth** | QLoRA E4B needs ~10 GB → fits 16 GB T4 | Start here; Unsloth has a Gemma-4 notebook — the canonical tutorial path |
| **Rented 4090/A100 (RunPod/Vast, ~$0.3–0.6/h)** | Comfortable LoRA bf16 (~17 GB) | For the real runs; a few $ per run |
| MLX on your MacBook (`mlx-lm`) | Only if Apple Silicon ≥32 GB **and** E-series/PLE arch is supported — verify first | Fun local path, off the beaten track |

### The runs — a deliberate curriculum

- **Run A — mechanics (one evening, ~300 examples).** QLoRA r=16, deliberately overfit. Goal is not a good model: it's learning the chat template + response-masking pitfalls, loss curves, checkpoints — then scoring it with your harness and *seeing memorization vs generalization with your own eyes* (train-station items ace it, held-out stations don't).
- **Run B — the real one (1–5k stratified examples).** Consensus-labeled + corruption-augmented + hard cases + explicit abstention examples ("garbled input → flag, don't guess"). r=16, α=32, lr 2e-4, 1–3 epochs, all linear modules, early-stop on held-out *rare-token* metrics, not loss. Data prep is a repo module (`finetune/prep.py`) that reuses the harness's split logic — leakage prevention as code.
- **Run C — one ablation (pick exactly one).** Data-size curve (500 vs 2k vs 5k) *or* corruption-augmentation on/off. Chart it. This is where fine-tuning stops being a recipe and becomes something you understand.

### Close the loop

Merge adapter → export **GGUF Q8 and Q4** → run on your MacBook → re-run the *same frozen harness* on fp16 vs Q8 vs Q4. You get the before/after headline (base vs FT vs frontier, CIs + McNemar) **plus** a measured quantization-cost curve — deployment reality included. Track runs in W&B (free tier) or TensorBoard; every run gets a results JSON committed under `reports/`.

**You'll learn (C):** LoRA/QLoRA hyperparameters as levers you've pulled, data curation → model behavior causality, quantization trade-offs, and honest before/after methodology.

---

## 5. Guardrails (the five ways this project quietly fails)

1. **Widening before the slice is done.** No NOTAM/PIREP code until the METAR/TAF loop produces a report. The 8-use-case map is the roadmap, not the sprint.
2. **Training before the harness is frozen.** Otherwise you'll (unconsciously) tune the benchmark to flatter the model.
3. **Trusting consensus without the seed calibration + mutant audit.** "Three parsers agree" can be three parsers sharing a bug.
4. **Data in git / license leaks.** Knots and ERAU never leave `third_party/`; `DATA_LICENSES.md` is part of code review.
5. **Prediction creep.** The moment a training example asks the model to know something not derivable from the input text, delete it.

---

## 6. Kickoff checklist — first two weekends

**Weekend 1 — repo + data heartbeat**
- [ ] Create repo, `uv init`, layout from §1, Apache-2.0, `DATA_LICENSES.md` stub
- [ ] Drop the two analysis docs into `docs/research/`; ADR-001: "vertical slice = METAR/TAF"
- [ ] `ingest/references.py`: reference pack + PDFs + Pre-Flight; manifest with checksums
- [ ] **Collector live** (`collect.yml` → HF dataset repo) — it runs while you sleep
- [ ] Open the Knots license issue on GitHub
- [ ] Pick ~50 stations (`configs/stations.yaml`), run the IEM backfill, Parquet in DuckDB

**Weekend 2 — first truth**
- [ ] Pydantic schema from FMH-1/AC 00-45H (start with wind/vis/wx/clouds/temp/QNH)
- [ ] First oracle adapter + Hypothesis round-trip test suite
- [ ] First invariant checks; hard-case queue exists and is non-empty (it will be)
- [ ] `LEARNING_LOG.md` entries for both weekends

From there: consensus labeler → gold seed → `eval/v1` freeze → baselines (base E4B on your Mac via GGUF, one frontier API) → Run A. Realistic cadence: **6–8 weekends to the full before/after report** on the slice.

---

## Key links (working set — full source lists in the two research docs)

- IEM ASOS/AFOS/PIREP: https://mesonet.agron.iastate.edu/request/download.phtml · AWC API + caches: https://aviationweather.gov/data/api/
- References: FAA Q-codes https://www.faa.gov/air_traffic/publications/atpubs/notam_html/appendix_b.html · Contractions https://www.faa.gov/air_traffic/publications/atpubs/cnt_html/ · NASR https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/ · WMO CCT https://github.com/wmo-im/CCT · OurAirports https://ourairports.com/data/
- Gold seeds: AC 00-45H https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_00-45H.pdf · FMH-1 https://www.icams-portal.gov/resources/ofcm/fmh/FMH1/fmh1_2019.pdf
- Oracles: https://github.com/python-metar/python-metar · https://github.com/mivek/python-metar-taf-parser · https://github.com/avwx-rest/avwx-engine
- Finetune: https://unsloth.ai/docs/models/gemma-4/train · model https://huggingface.co/google/gemma-4-E4B
- Eval sets: https://huggingface.co/datasets/AirsideLabs/pre-flight-06 · https://github.com/Estrellajer/Knots (eval-only) · https://commons.erau.edu/ga-wx-display-interpretation/17/ (eval-only)
