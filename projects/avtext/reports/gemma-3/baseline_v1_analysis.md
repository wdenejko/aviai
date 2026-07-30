# Baseline analysis — base Gemma-4-E4B on eval/v1

**Model:** `gemma-4-E4B-it` (Gemma 3n, MLX 4-bit) · served text-only via `mlx_vlm.server`
**Eval:** `eval/v1` — 620 records, `sha256:60870798…` · **Prompt:** `decode-json-v1` · **Decode:** temp 0, max_tokens 256
**Run:** `reports/runs/20260729T220516Z-gemma-4-e4b-it-mlx-4bit/` · **Analysis:** 2026-07-30
**Harness:** frozen (ADR-006); scoring per `harness/score.py`. Numbers are from the corrected run — see §8.

---

## 1. Executive summary

The base model is **good at *reading* a METAR and bad at *doing arithmetic on it***. It extracts
temperature, dewpoint, station metadata, and cloud/visibility state at 86–100% accuracy, but it
**cannot convert units** — and because two-thirds of the eval requires a unit conversion somewhere,
that single weakness drags whole-record exactness to near zero.

- **Overall value accuracy 77.8%** [76.8, 78.9] (hits / values that existed), **whole-record EM 4%**,
  **hallucination 1.5%**, **JSON-valid 100%**.
- **The dominant failure is unit conversion**, uniform across every field that needs one: altimeter
  inHg→hPa **2%**, wind m/s→kt **1%**, visibility SM→m **71%** — versus 94–95% on the *same fields*
  when no conversion is required.
- **Hallucination is low but non-zero** — the model mostly respects "use null, don't guess," but
  occasionally fabricates (chiefly wind gust/direction, and on masked `///` wind groups).
- Against the parser reference (97.5% / 0% / 82%), the base LLM is far behind on structured decode —
  as ADR-004 predicted — but the gap is **one concrete, learnable skill**, making it an ideal
  Phase-4 finetuning target.

**One-line takeaway:** the model reads correctly and converts almost nothing; fix conversion and
whole-record accuracy should move from ~4% to ~30%+.

---

## 2. What was measured, and how

Each of the 620 frozen eval records is one raw METAR. The model is prompted (`decode-json-v1`) to
emit a JSON decode in canonical units (kt, m, °C, hPa) with **explicit permission to abstain**
("use null; do NOT guess" — the line that makes a fabrication the model's own fault, not obedience).
Its output is parsed and compared field-by-field against the record's `reference` — the 3-parser
consensus, gold-calibrated against NOAA's official decode at ~99.98% on altimeter (Phase 2). Scoring
uses the five-way taxonomy that keeps the two ways of being wrong distinct:

| reference | prediction | outcome | meaning |
|---|---|---|---|
| value | = | **HIT** | correct |
| value | ≠ | **WRONG** | wrong value |
| value | null | **ABSTAIN** | declined a knowable value (safe miss) |
| null | value | **HALLUCINATE** | fabricated from nothing (dangerous) |
| null | null | **TRUE_ABSTAIN** | correctly silent |

Aggregated: **value accuracy** = recall = hits / values-that-existed; **hallucination rate** =
fabrications / fields-truly-absent; **EM** = fraction of records with every value right and nothing
fabricated. 95% intervals are record-level bootstrap (small cells → wide bands, reported honestly).

---

## 3. Headline results

| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 80.4% [78.3, 82.4] | 1.0% | 6% |
| unseen_station | dissent | 60 | 70.0% [66.7, 73.2] | 0.0% | 0% |
| unseen_time | clean | 200 | 79.4% [77.6, 81.2] | 2.9% | 7% |
| unseen_time | dissent | 60 | 79.7% [76.0, 83.0] | 0.0% | 2% |
| unseen_time | parse_fail | 100 | 73.0% [71.5, 74.5] | 1.3% | 0% |
| **overall** | | **620** | **77.8% [76.8, 78.9]** | **1.5%** | **4%** |

The two generalization axes land within a couple of points of each other — **no memorization
advantage** from having seen a station before, which is what we want and a small vote of confidence
in the leak-free split.

---

## 4. Finding 1 — the model cannot convert units (the dominant failure)

This is the whole story. Split each conversion-bearing field by whether a conversion is required:

| field | native (read as-is) | needs conversion | drop |
|---|---|---|---|
| **altimeter** | `Q` (hPa): **95%** (n=204) | `A` (inHg→hPa): **2%** (n=416) | −93 pts |
| **wind speed** | `KT`: **75%** (n=506) | `MPS` (m/s→kt): **1%** (n=107) | −74 pts |
| **visibility** | metric: **94%** (n=108) | `SM` (mi→m): **71%** (n=415) | −23 pts |

The pattern is unmistakable: **when the value is already in the target unit, the model reads it;
when arithmetic is required, it fails** — most totally for the two conversions far from 1:1
(inHg×33.86, m/s×1.94), least for statute-miles (a rounder, more likely-memorised mapping).

### Concrete examples (verbatim, temp 0)

**inHg altimeter — reads the digits, doesn't convert:**
```
RAW  : KSEA 280840Z AUTO 35005KT 10SM CLR 13/11 A2979 RMK T01300110 MADISHF
MODEL: altimeter_hpa=1013   (A2979 = 29.79 inHg = 1009 hPa)  -> WRONG
       temperature_c=13 ✓   dewpoint_c=11 ✓   visibility_m=16100 ✓
```
The model emits **1013** — the ISA standard-pressure default — rather than converting. When it does
attempt a conversion it can be wildly off (an `A2993` case produced `1029`, not the correct `1014`).

**m/s wind — outputs the raw number, unconverted:**
```
RAW  : UUWW 090100Z 16005MPS 9999 SCT022 17/15 Q1005 R24/000062 NOSIG
MODEL: wind_speed=5   (16005MPS = 5 m/s = 10 kt)  -> WRONG
       wind_dir=160 ✓   altimeter_hpa=1005 ✓ (Q-report, no conversion)   temp/dew ✓
```
`5` is the m/s figure copied straight through. Note the same record's `Q1005` altimeter is correct —
because it needed no conversion.

### Why this caps whole-record EM

EM requires *every* value correct. **147 of 620 records fail on altimeter alone** — the model gets
everything else right and only misses the inHg conversion. So:

- current EM = **4%** (26/620 perfect)
- **EM ceiling if altimeter conversion were fixed ≈ 28%** (173/620)
- fixing wind + visibility conversion too would lift it further

That is the quantified size of the prize for Phase 4, from one skill.

---

## 5. Finding 2 — strong at reading, weak at extraction

Per-field value accuracy across all 620 records:

| field | value acc | note |
|---|--:|---|
| report_type | 100% | near-vacuous — IEM data is all METAR (see §11) |
| cavok | 96% | strong |
| temperature_c | 91% | strong |
| dewpoint_c | 90% | strong |
| automated | 86% | misreads AUTO ~1 in 7 |
| wind_dir | 78% | mostly good |
| visibility_m | 75% | dragged by SM conversion |
| wind_gust | 72% | (small n=53; mostly correctly-absent elsewhere) |
| clouds | 69% | layer/type extraction is lossy |
| wind_speed | 62% | dragged by m/s + extraction errors |
| **altimeter_hpa** | **33%** | **the inHg cliff** |

Two clusters. The model is reliable at **"read a labelled scalar"** (temperature, dewpoint, report
type, CAVOK) and unreliable at **"extract-and-transform"** (wind speed, altimeter) and at
**structured extraction** (clouds — parsing `FEW100 SCT110` into typed layers is only 69%). A raw
extraction slip appears even without conversion: on `KSEA … 35005KT` the model returned
`wind_dir=280, wind_speed=35` for what is `350° / 5 kt` — a tokenisation error, not arithmetic.

---

## 6. Finding 3 — hallucination is low but real

**1.5% overall (11 fabrications).** By field: wind_gust (4), wind_dir (3), visibility (2),
wind_speed (1), dewpoint (1). The model is *mostly* well-behaved — it abstains rather than invents
on genuinely-absent fields — but it is **not** at zero (the 20-record smoke test's 0% was an
artifact).

The most instructive case is a **masked wind group**:
```
RAW  : KJFK 172115Z AUTO ///18G24KT 10SM CLR 22/15 A3004 RMK T02200150 MADISHF
MODEL: wind_dir=180   (raw is `///` = direction unavailable; reference is null)  -> HALLUCINATE
       wind_speed=24  (should be 18; it grabbed the gust)   altimeter=1013 (should 1017)
```
`///` means the automated station could not determine wind direction — the correct answer is
"unknown" (null). The model manufactured `180`, almost certainly by reading the `18` (speed digits)
as a heading. This is the exact failure a safety-domain eval must catch: **inventing a plausible
value where the honest answer is "I don't know."** Rare here, but real, and holding it flat is a hard
constraint on Phase 4 (a finetune that lifts accuracy by learning to always-answer is a regression).

---

## 7. Finding 4 — generalization, hard cases, format robustness

- **Generalization is flat across axes.** `unseen_station` clean 80.4% ≈ `unseen_time` clean 79.4%.
  The model gains nothing from station familiarity — expected for a base model on post-cutoff data,
  and a sanity check on the leak-free split.
- **Dissent cells** (70–80%) are near the clean cells here — the model's errors are dominated by
  conversion, which is orthogonal to the parser-disagreement that defines "dissent."
- **parse_fail (73%)** is the interesting one. These are the UUWW m/s reports that break a parser;
  the LLM *reads them fine* (unfazed by the Russian format) but **fails the m/s conversion** — so its
  errors here are the same Finding-1 story, not a collapse. The LLM's tolerance of messy formats is a
  genuine (if modest) edge over the brittle parsers.
- **JSON validity is 100%** (0 unusable) after the §8 fix.

---

## 8. Measurement integrity (bugs found and fixed)

"Your measurement code is part of the test" is the project's refrain, so this section stays in the
report. Preparing it triggered a two-part audit of the harness itself.

**Scoring path — 3 issues found and fixed (`a001da6`, `6f60386`-era):**
1. **Clouds-shape crash.** The output canonicaliser assumed `clouds` arrives as `[cover, base]`
   pairs; the model sometimes emits *dicts*, and the resulting `KeyError` scored the **entire record**
   as invalid — even when temperature, altimeter, etc. were right. This alone accounted for **all 25**
   of the run's original "invalid" records; fixing it took JSON-valid 96% → **100%** and lifted
   overall value accuracy **75.1% → 77.8%**. The conversion findings (§4) were unaffected — they
   concern *wrong values*, not invalids.
2. **Under-canonicalisation.** `_canon` coerced only some fields, passing the model's raw value
   through for `wind_dir/speed/gust` (reference: int), `report_type` (upper), and `automated/cavok`
   (bool). A right-but-differently-typed answer would score WRONG. Now every field is coerced into the
   reference's type. **Audit: re-scoring the run's saved predictions with vs without the fix flipped
   0 outcomes** — this model emits clean JSON, so the fix is a no-op here (kept for robustness).
3. **Div-by-zero** on an empty eval — guarded (latent; never triggered).

**Reference path — clean.** The gold seed (Phase 2) validated only the *numeric* fields; `clouds`,
`cavok`, `automated`, `report_type` were never gold-checked, and the latter three are computed by
*shared* functions echoed across all three parsers (so their "consensus" is illusory). Adversarial
review + empirical checks found **no bugs**: conversions correct; `automated`/`cavok` matched an
independent recompute from the raw **620/620**; `clouds` decode hand-verified, adapters agree 99%
(disagreements are same-base ordering, resolved by majority), **0 references are None**; **0/620
degenerate** references; and **all 100 parse_fail references are 2-way cross-checked** (no record
leans on a single parser). The ground truth is trustworthy.

---

## 9. Comparison to the parser reference

| | value acc | halluc | EM |
|---|--:|--:|--:|
| python-metar (parser baseline)¹ | 97.5% | 0% | 82% |
| **base Gemma-4-E4B** | **77.8%** | **1.5%** | **4%** |

¹ Mildly circular (a consensus member scored against the consensus) — a reference point, not a fair
rival. But the direction is unambiguous: **on clean structured decode the parser dominates**, and
even a strong finetune is unlikely to beat 200 lines of deterministic parser. The strategic
implication (consistent with ADR-004) is that the LLM's durable advantage is **not** structured
decode — it is the free-text/generative tail (NOTAM/SIGMET briefing) where no parser competes. The
METAR finetune is the *method* being learned, not the destination.

---

## 10. Implications for Phase 4 (the finetuning study)

The baseline hands Phase 4 an unusually clean hypothesis:

> **LoRA the unit conversions (inHg→hPa, m/s→kt, SM→m) and the raw-extraction slips, and measure the
> lift on the same frozen harness.**

Concrete, falsifiable targets:

| metric | baseline | target | how |
|---|--:|--:|---|
| altimeter `A` accuracy | 2% | ≥ 90% | teach the ×33.86 mapping |
| wind `MPS` accuracy | 1% | ≥ 90% | teach the ×1.94 mapping |
| overall value acc | 77.8% | ~85%+ | above + extraction |
| whole-record EM | 4% | ≥ 28% | flows from altimeter alone |
| **hallucination** | **1.5%** | **≤ 1.5%** | **must not rise — the safety floor** |

Evaluation is already fixed: re-run the identical harness (same eval `sha256`, same prompt), then a
**McNemar paired test** on per-record correctness (base vs finetuned on the same 620 records) to
confirm any gain is real, not resampling noise. Because the training set is the disjoint `train`
split (station+time held out from eval/v1), a lift is generalization, not memorisation.

---

## 11. Threats to validity

- **Reference circularity.** The `reference` is parser-consensus, not an independent human decode.
  Mitigated by the Phase-2 gold calibration (99.98% vs NOAA on altimeter); on `dissent` records the
  "correct" answer is the majority of parsers, a soft target.
- **`report_type` is near-vacuous.** IEM data carries no METAR/SPECI keyword, so every reference is
  `METAR` and the field's 100% tells us nothing about capability. Ignore it as a signal.
- **Cloud scoring is order-sensitive.** Layers at the same base (`FEW030 BKN030`) have an ambiguous
  order; the reference commits to one and dings the model for the other valid reading. Rare, but a
  slight downward bias on the 69% clouds number.
- **CAVOK → visibility is null.** A model that emits `10000` for a CAVOK report is scored as
  *hallucinating*. Defensible (CAVOK reports no visibility number), but it penalises a reasonable
  interpretation; only affects the handful of CAVOK records.
- **Single prompt.** All results are for `decode-json-v1`. The conversion failure *could* be partly
  promptable away — though the prompt already says "convert m/s→kt, inHg→hPa" explicitly, so this
  reads as a capability gap, not phrasing. A small prompt-ablation is worth doing before over-reading
  absolute numbers.
- **4-bit quantisation.** The served model is MLX 4-bit; some accuracy is lost vs bf16. The finetune
  must be evaluated at the *same* quant for a fair paired comparison.
- **Small cells.** Dissent cells are 60 records → CIs span ~7 points. Treat per-cell numbers as
  directional; the overall and the per-field conversion splits (larger n) are solid.

---

## 12. Appendix

**Artifacts** (`reports/runs/20260729T220516Z-gemma-4-e4b-it-mlx-4bit/`): `report.md` (auto),
`run.json` (repro block + overall metrics), `scores.jsonl` (per-record outcomes **and predictions**).

**Reproduce:**
```bash
make serve                 # mlx_vlm.server on the local Gemma
make harness MODEL=<path>  # scores it against frozen eval/v1 -> reports/runs/<id>/
```
Reproducibility anchor: eval `sha256:608707987dfc8f13…`, prompt `decode-json-v1`, decode `temp=0`.
