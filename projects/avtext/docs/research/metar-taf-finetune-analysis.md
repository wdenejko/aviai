# Fine-tuning Gemma 4 E4B for METAR/TAF — Viability, Gaps, Accuracy & Benchmark Design

*Deep-research briefing · July 2026 · prepared for Wojtek*

---

## 0. TL;DR verdict

**The project is viable and genuinely worth doing — but only if you scope it as a *hybrid interpretation/robustness* system, not as an end-to-end "LLM decodes METAR" replacement, and not as a weather *predictor*.**

Three findings drive everything below:

1. **Raw METAR/TAF decoding is a solved, deterministic problem.** Mature open-source parsers (`python-metar`, `metaf`, `avwx-engine`, `metar-taf-parser`) hit ~99–100% on well-formed reports. An LLM will be *slower, costlier, and less reliable* than a parser on clean input. So "fine-tune E4B to decode METAR" — taken literally — is re-solving a solved problem with a worse tool.

2. **The LLM's real value is in the places a parser fails:** malformed/garbled automated observations, typos, non-standard national idioms, the free-text US `RMK` section, fluent plain-language briefing, cross-report Q&A, and risk framing. That's where you have defensible headroom — and there is genuine **white space**: no published benchmark, dataset, or fine-tuned small model for this task exists (as of July 2026).

3. **"Predicting" weather from a report text is not something fine-tuning can deliver.** A TAF is *already* a forecast produced by NWP + a human forecaster. Your model can *decode* it perfectly, but training it to *generate* forecasts (or infer unstated facts) teaches confident hallucination — a well-documented fine-tuning failure mode. Keep prediction out of the training targets and pull real forecasts via tools instead.

**Net:** scoped right (parser + fine-tuned E4B for the messy tail and natural language, benchmarked for hallucination as well as accuracy), this is a strong, low-cost, novel project. Scoped as "replace the parser" or "predict the weather," it will disappoint.

> **Model correction:** `google/gemma-4-E4B` is **not** an MoE. It's a **dense** model of ~**8B total weights** that runs at ~**4.5B "effective" parameters** via **Per-Layer Embeddings (PLE)** — Google's on-device efficiency technique (the successor to Gemma 3n's E-series). The sparse-MoE variant in the Gemma 4 family is the separate `26B-A4B` (26B total / ~3.8B active). Practically, E4B being dense is *good news*: no expert router to destabilize during fine-tuning, and it fits on one consumer GPU.

---

## 1. The model: Gemma 4 E4B

| Property | Value | Implication for you |
|---|---|---|
| Architecture | **Dense** transformer + hybrid local/global attention; **Per-Layer Embeddings (PLE)**. Not sparse MoE. | Simpler, stable fine-tuning; no router freezing. |
| Parameters | ~**8B total** (with embeddings), ~**4.5B effective** | Slightly more capable than a literal 4B dense; still small — expect a *capacity ceiling* on reasoning-heavy tasks. |
| Context | **128K tokens** | Ample: fits many reports + reference tables + few-shot exemplars at once. |
| Modalities | **Text, image, audio** | Latent bonus — could later ingest weather charts/imagery; unnecessary for text METAR/TAF now. |
| Training cutoff | **January 2025** | Matters for benchmark contamination (see §6.3): pre-cutoff reports may be memorized. |
| License | **Apache 2.0** | Clean for on-prem/commercial use. |
| Vocab | 262K | Fine. |

**Fine-tuning footprint** (Unsloth/HF, corroborated):

- **QLoRA (4-bit): ~10 GB VRAM** → fits a single 16 GB (or 24 GB comfortably) consumer GPU.
- **LoRA (bf16): ~17 GB VRAM.**
- Cost of a run on a few thousand short examples for 1–3 epochs: **single-GPU, a few GPU-hours, order of tens of dollars.** You can afford many iterations.

---

## 2. The central reframe — where an LLM adds value

METAR/TAF is governed by **ICAO Annex 3** (21st ed., Amendment 82, applicable 27 Nov 2025, with the new **PANS-MET Doc 10157** now carrying much of the coding detail), **WMO No. 306** code forms **FM-15 (METAR/SPECI)** and **FM-51 (TAF)**, and in the US **FMH-1 (2019)**. The *body* of both formats is a near-regular, context-free-parseable grammar. The **US `RMK` section**, automated-station corruption, and national deviations are the messy, semi-free-text part.

So the domain splits cleanly:

```
  DETERMINISTIC  ───────────────────────────────►  JUDGEMENT / MESSY
  (parser wins)                                     (LLM has headroom)

  wind, visibility, RVR, clouds,        malformed/garbled AUTO obs,
  temp/dewpoint, altimeter,             typos, truncation, non-standard
  present-weather w'w' codes,           entries, free-text RMK, national
  flight-category (VFR/MVFR/IFR/LIFR),  idioms, plain-language briefing,
  TAF change-group timing               cross-report reasoning, risk framing
```

**Rule of thumb:** if a 200-line parser already does it exactly, the LLM should *not* be your primary tool for it. Fine-tune the LLM for the right column, and let the parser own the left column *and* act as your free label generator / verifier.

**Recommended architecture (hybrid):**

1. **Deterministic parser** = primary decoder for well-formed input + gold-label oracle + output verifier.
2. **Fine-tuned E4B** = robustness to messy input + fluent natural-language interpretation.
3. **Constrained/structured decoding** (JSON-schema/GBNF) wraps the model's structured output — but decouple any free-text normalization step from the rigid grammar (rigid schemas *hurt* when a step needs reasoning first).
4. **Tools/RAG** for reference facts (station elevation, runway data, rare-remark tables, regulatory minima) and for pulling *actual* forecasts. **Never** train the model to invent these.

---

## 3. Task-by-task viability & accuracy expectations

> **These are informed estimates, not measured results** — there is no published METAR-decode benchmark, so treat them as hypotheses your benchmark (§6) exists to confirm. "Parser" = deterministic baseline; "Base E4B" = few-shot, no fine-tune; "FT E4B" = your target after fine-tuning.

| Task | Nature | Parser | Base E4B (est.) | FT E4B target (est.) | Is the LLM worth it? |
|---|---|---|---|---|---|
| **Field extraction — clean body → JSON** | Deterministic | ~99–100% | 85–95% field-F1 | 97–99%+ | ❌ Marginal — parser already wins |
| **Field extraction — messy/garbled/RMK** | Mixed | ~60–80% | 55–75% | **85–95% field-F1** | ✅ **Core value** |
| **Raw → plain-language briefing** | Mostly deterministic, fluent | n/a (templated) | Fluent but some fabrication | Fluent + **<2–5% hallucinated facts** (target) | ✅ Value = phrasing + graceful RMK |
| **Flight category (VFR/MVFR/IFR/LIFR)** | Fully deterministic | ~100% | 80–92% | high-90s | ❌ 20 lines of code does this; use as consistency check |
| **TAF → timeline flattening** | Deterministic rules, tricky precedence | ~90–98% | 60–80% | 85–95% | ⚠️ Parser+code better; LLM ok as convenience |
| **Plain-language → raw encoding** | Under-constrained target grammar | n/a | High hallucination risk | Needs validator; treat cautiously | ⚠️ Only with grammar-constrained decode + validator |
| **Anomaly / error detection** | Mixed (rules + judgement) | Rules catch range errors | Weak | Moderate — **real headroom on "looks-wrong"** | ✅ Promising, safety-relevant |
| **Cross-report / route reasoning** | Judgement + regulatory rules | n/a | Weak (small model) | Weakest area — needs frontier-class reasoning | ⚠️ Highest value *and* highest risk; small model struggles |
| **Weather *prediction* / nowcasting** | Predictive — needs physical data | n/a | — | **Not achievable from text; do not attempt** | ❌ Out of scope — impossible |

**Reality check on the small-model ceiling:** the best available proxy — the **Pre-Flight benchmark** (July 2026, aviation *operational* MCQs) — shows ~8B models scoring **~58%** vs frontier **~80–83%** and human experts **~95%**. So expect E4B to be strong on narrow *format/extraction* tasks (fine-tuning's sweet spot) and weak on multi-step *reasoning* tasks. This is exactly the pattern in the *LoRA Land* study: fine-tuned small models beat GPT-4 on NER (0.99 vs 0.75) and structured extraction, but lose badly on open-ended generation.

---

## 4. Gaps & risks (ranked)

1. **The "already solved" gap (strategic, biggest).** Before writing any training code, answer one empirical question: *what fraction of your real feed does a good parser fail on?* If it's small, the LLM is over-engineering; if your feeds are messy (typos, truncation, mixed conventions, heavy RMK), that failure rate *is* your justification. Measure it first.

2. **Prediction is impossible from text (scope creep).** "Predicting etc." in the original goal is the trap. Decoding an existing TAF = fine. Generating a forecast = needs NWP/physical inputs the model doesn't have. Training on prediction targets *manufactures confident hallucination* (Gekhman et al., EMNLP 2024 — the load-bearing paper here: fine-tuning teaches a model to *use* knowledge, not *acquire* it, and fitting "unknown" examples linearly increases hallucination).

3. **Safety-critical hallucination + regulatory posture.** Aviation weather is safety-critical. FAA (AI Safety Assurance Roadmap, 2024) and EASA (AI Concept Paper Issue 2/3) treat AI output as *advisory only*; the pilot retains legal responsibility (14 CFR 91.103). Industry ships these features with explicit "verify against official sources" disclaimers (ForeFlight AI Connector, 2026). **Consequence: you must benchmark hallucination and abstention, not just accuracy** — a model that fabricates a ceiling value is worse than one that flags "unparseable."

4. **No existing benchmark/dataset → you must build it (this is your explicit ask).** Good news: raw data is abundant and mostly public-domain (NOAA). Bad news: contamination — raw METARs are in the base model's pretraining, inflating the "before" number and hiding real fine-tuning lift. Defense: build a test set from reports issued *after* the Jan-2025 cutoff (§6.3).

5. **Label-source circularity.** If you train only on one parser's output, the model *launders that parser's bugs into ground truth*. Mitigation: dual independent parsers, treat disagreements as hard cases, human-audit a sample.

6. **Distribution mismatch.** Synthetic/clean generated reports are "too tidy" vs real feeds. Always mix in and *test on* real reports; measure the clean-vs-real gap explicitly.

7. **Data licensing nuance.** NOAA/aviationweather.gov + IEM archive → effectively public domain (attribute, be gentle on rate limits). But the **ERAU validated weather question bank** (a great ready-made eval asset, 65 human-validated items) is **CC BY-NC-ND** → usable for *evaluation only*, not training or redistribution.

8. **Multimodal capacity is mostly unused.** E4B carries vision+audio encoders you won't need for text. Fine — just know you're using a fraction of the model. (Weather-chart ingestion is a plausible *later* extension that would actually exploit it.)

---

## 5. Benchmark design (the core deliverable)

This is the "measure before vs after" system you asked for. Design it once, well — a sloppy benchmark will make a real improvement look like noise, or noise look like improvement.

### 5.1 Task suite & metric per task

Build the benchmark as **separately-scored task modules**, not one blended score:

| Module | Metric(s) | Gold source |
|---|---|---|
| Field extraction (clean) | Field-level **micro & macro-F1**; whole-record exact match | Dual-parser agreement |
| Field extraction (messy/RMK) | Field-F1 (macro surfaces rare-field regressions) | Human-adjudicated |
| Flight category | Accuracy + confusion matrix | Deterministic rules |
| TAF timeline | Per-time-slice state accuracy | Parser + human spot-check |
| Plain-language briefing | **Faithfulness** = every extractable fact matches parser; fluency via calibrated LLM-judge | Parser cross-check |
| Robustness | Field-F1 on corrupted inputs vs clean | Parser on clean version |
| **Hallucination / abstention** | % fabricated values where gold = null; correct-abstention rate | Parser null-map |

Report **macro-F1 prominently** — overall accuracy hides the case where fine-tuning improves common fields while quietly breaking a rare token (RVR, wind-shear `WS`, `VV///`, `PROB40 TEMPO`).

### 5.2 Data sourcing & gold labels

- **Bulk / recent:** `aviationweather.gov` Data API + bulk cache files (`metars.cache.csv.gz`, `tafs.cache.xml.gz`).
- **History / volume:** **Iowa Environmental Mesonet (IEM)** ASOS archive — worldwide, deep history, raw strings, free.
- **Gold labels:** run **two independent parsers** (e.g. `python-metar` + `metar-taf-parser`/`avwx-engine`); where they *agree*, auto-accept; where they *disagree* or fail → that's your **hard-case pool**, human-adjudicate a sample. This gives near-free labels at scale plus a curated messy set.
- **Stratify / oversample the long tail:** `CAVOK`, `NSC/NCD/SKC/CLR`, `VV///`, RVR with tendency (`R28/0600V1200U`), `WS`, recent weather `RE`, `TEMPO/BECMG/PROB30/40`, `VRB`, missing-group `/////`, negative `M` temps, `Q` vs `A` altimeter, free-text `RMK`. Uniform sampling under-represents these — and they're exactly where value lives.
- **Corruption augmentation** (for the robustness module and training): inject typos, missing spaces, truncation, wrong-order groups, mixed conventions — paired with the *correct* label from the clean version.

### 5.3 Splits & contamination (do NOT random-split)

- **Group by station (ICAO):** no test station appears in training → tests cross-airport/convention generalization.
- **Split by time:** train on an earlier period, test on a strictly *later* one → matches deployment, prevents look-ahead leakage. (Reports from one station hours apart are near-duplicates; random splitting leaks badly.)
- **Contamination-resistant test set:** draw the headline test set from reports issued **after Jan 2025** (the base model's cutoff) so "before" numbers aren't inflated by memorization.
- **Hold out entire phenomenon types** to measure tail performance, not just the common-case average.

### 5.4 Baselines to run (this is what "before/after" means)

Run *all* of these through the **identical harness**:

1. **Deterministic parser** — the ceiling on clean input (and reality check: if you can't beat it on messy input, stop).
2. **Base E4B, few-shot** — the "before."
3. **Frontier model, few-shot** (e.g. a GPT-5/Claude-class model) — the "is a small specialized model even competitive?" bar.
4. **Fine-tuned E4B** — the "after."

### 5.5 Fair before/after protocol (the crux)

Hold **everything** constant except the weights: same held-out test set, same prompts, same few-shot exemplars, same decoding params (temperature — use **0** for safety-critical determinism — top-p, max tokens), same constrained-decoding settings, same post-processing.

**Statistics — no bare point estimates:**
- **Bootstrap confidence intervals** on every metric.
- **McNemar's paired test** for exact-match deltas (before/after evaluated on the *same* items).
- **Sizing:** to distinguish a few-percentage-point difference you need **several hundred to ~1–2k items per stratum**. With ~100 items, CIs are too wide to trust. Size *per field/stratum*, not just overall.

### 5.6 Benchmark pitfalls that produce misleading results

1. Random split → station/time leakage → inflated scores.
2. Testing against parser labels the parser itself got wrong → measuring agreement-with-a-buggy-oracle.
3. Different prompt/decoding for base vs FT → unfair delta.
4. Single aggregate exact-match hiding field-level regressions.
5. No CIs / no paired test → over-reading noise.
6. Ignoring contamination → inflated "before."
7. Testing only on clean synthetic data → misses the real messy distribution.
8. Uncalibrated LLM-as-judge standing in for a parser oracle. If you use a judge for prose, report **Cohen's κ** (not raw agreement — it overstates reliability by 33–41 points) and audit position bias with AB/BA swaps.

---

## 6. Recommended fine-tuning recipe (E4B)

Because E4B is dense (no MoE router), this is the straightforward case:

- **Method:** **QLoRA** (4-bit, ~10 GB) to iterate cheaply; move to **bf16 LoRA** (~17 GB) for the final run if val metrics justify it.
- **Rank r = 16** (start), raise to 32 only if rare-field metrics plateau below the parser ceiling. **α = 2r** (32→64). **LR 2e-4.** **1–3 epochs.** Target **all linear modules** (`q,k,v,o,gate,up,down_proj`). Dropout 0–0.1, weight decay 0.01–0.1, warmup 5–10%.
- **Data:** ~**1–5k stratified examples** (coverage over volume — every token type represented dozens of times). Parser-generated `(raw ↔ JSON ↔ plain-language)` triples + corruption-augmented + edge-case-oversampled. For the plain-language layer (no deterministic gold), **distill from a frontier model then filter**: round-trip check + parser cross-check every extractable field, drop disagreements.
- **Format:** `gemma-4` chat template; `train_on_responses_only` masking.
- **Never** include prediction/unstated-fact targets. Add explicit **abstention** examples ("input unparseable → flag, don't guess").
- **Overfitting signal:** train loss < ~0.2 is a warning; but drive early-stopping off **held-out station/time + rare-token metrics**, not loss — some memorization is fine for a near-deterministic task.

---

## 7. Suggested phased plan

1. **Scope probe (½ day):** run 2 parsers over a real sample of *your* feeds; measure the failure/messy fraction. This decides whether the LLM is justified and sizes the prize.
2. **Build the benchmark first (before any training):** data pull → dual-parser gold + hard-case pool → station/time splits → post-cutoff test set → metric harness → run parser + base-E4B + frontier baselines. Now you have your "before."
3. **Generate training data:** parser-labeled + corruption-augmented + distilled-and-filtered NL, stratified by token.
4. **Fine-tune (QLoRA), iterate** against the harness. Watch macro-F1 and the rare-token tail.
5. **Report before/after** with CIs + McNemar, per-module, including hallucination/abstention.
6. **Decide deployment shape:** parser-primary + FT-E4B-for-tail/NL, JSON via constrained decoding, tools for reference facts.

---

## 8. Honest bottom line

- **Viability: good**, with the scoping above. There's real white space (no public benchmark/dataset/fine-tune exists) and the economics are trivial (tens of dollars, one GPU).
- **Accuracy: high where it doesn't matter (clean decode — parser already does it), and the interesting question is whether FT-E4B can reach ~85–95% field-F1 on the *messy tail* and keep hallucination low** — that's the number worth chasing, and exactly what your benchmark will tell you.
- **Biggest mistakes to avoid:** (1) framing it as "LLM replaces the parser," (2) training it to "predict" weather, (3) building a benchmark that random-splits and ignores contamination. Avoid those three and this is a strong project.

---

## Sources

**Model**
- Gemma 4 E4B model card — https://huggingface.co/google/gemma-4-E4B · https://ai.google.dev/gemma/docs/core/model_card_4
- Gemma 4 fine-tuning (Unsloth) — https://unsloth.ai/docs/models/gemma-4/train
- Gemma 4 overview — https://ai.google.dev/gemma/docs/core

**Domain / standards / data / parsers**
- ICAO Annex 3 & PANS-MET 2025 reform — https://community.wmo.int/media/news/aviation-news-2025-09-05-icao-publishes-new-editions-of-annex-3-and-pans-met
- WMO No. 306 (FM-15/FM-51) — https://community.wmo.int/about-manual-codes-volume-i1
- US FMH-1 (2019) — https://www.icams-portal.gov/resources/ofcm/fmh/FMH1/fmh1_2019.pdf
- Flight categories (VFR/MVFR/IFR/LIFR) — https://aviationweather.gov/gfa/help/
- aviationweather.gov Data API — https://aviationweather.gov/data/api/ · IEM archive — https://mesonet.agron.iastate.edu/request/download.phtml
- Parsers — https://github.com/python-metar/python-metar · https://github.com/avwx-rest/avwx-engine · https://github.com/aeharding/metar-taf-parser · https://metaf2xml.sourceforge.io/

**State of the art / baselines**
- Pre-Flight benchmark — https://arxiv.org/html/2607.01829v1
- Knots (NOTAM parsing, closest analog) — https://arxiv.org/html/2511.12630 · https://github.com/Estrellajer/Knots
- AviationGPT — https://arxiv.org/abs/2311.17686 · AviationLLM — https://arxiv.org/html/2506.14336v1
- ERAU validated weather question bank (CC BY-NC-ND) — https://commons.erau.edu/ga-wx-display-interpretation/17/
- ForeFlight AI Connector — https://foreflight.com/blog/the-foreflight-ai-connector-your-flight-data-now-available-in-chatgpt
- FAA AI Safety Assurance Roadmap — https://www.faa.gov/aircraft/air_cert/step/roadmap_for_AI_safety_assurance · EASA AI Concept Paper — https://www.easa.europa.eu/en/newsroom-and-events/news/easa-publishes-artificial-intelligence-concept-paper-issue-2-guidance

**Fine-tuning & eval methodology**
- Gekhman et al., *Fine-Tuning on New Knowledge → Hallucinations* — https://arxiv.org/abs/2405.05904
- Ovadia et al., *Fine-Tuning or Retrieval?* — https://arxiv.org/abs/2312.05934
- *LoRA Learns Less and Forgets Less* — https://arxiv.org/pdf/2405.09673 · *LoRA vs Full FT: Illusion of Equivalence* — https://arxiv.org/abs/2410.21228
- *LoRA Land* — https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4
- *Let Me Speak Freely?* (constrained decoding) — https://arxiv.org/abs/2408.02442
- Synthetic data (eugeneyan) — https://eugeneyan.com/writing/synthetic/
- Contamination-resistant benchmarks — https://arxiv.org/html/2605.19999v1
- Bootstrap CIs for LLM eval — https://engineering.indeedblog.com/blog/2026/07/bootstrap-confidence-intervals-for-llm-evaluation/
- LLM-as-judge reliability — https://arxiv.org/html/2606.19544v1
- Unsloth LoRA hyperparameters — https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
