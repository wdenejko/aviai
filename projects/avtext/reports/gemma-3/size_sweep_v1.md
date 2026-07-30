# Small-model size sweep — aviation decode on eval/v1

**Server:** dashi (GMKtec Strix Halo, AMD Ryzen AI Max+ 395) · **Serving:** llama.cpp (Vulkan/ROCm toolbox), Q8_0 GGUF
**Eval:** `eval/v1` — 620 records, `sha256:60870798…` · **Prompt:** `decode-json-v1` · **Decode:** temp 0
**Date:** 2026-07-30 · Same frozen harness as the M5 baseline (ADR-006)

Five small models, one family ladder (Gemma) plus a cross-family point (Qwen), scored on the identical
frozen eval. The question: **how does aviation-decode capability scale with size, and where does it break?**

---

## 1. Results

| model | params | value-acc | halluc | EM | JSON-valid | altimeter (all) | **inHg→hPa** | **m/s→kt** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 270M | 0.27B | 17% | **67%** | 0% | 69% | 0% | 0% | 0% |
| Qwen2.5 0.5B | 0.5B | 5% | 1% | 0% | 99% | 0% | 0% | 0% |
| Gemma 3 1B | 1B | 26% | 20% | 0% | 90% | 3% | 0% | 0% |
| Gemma-3n E2B | ~2B | 51% | 10% | 0% | 92% | 29% | 0% | 2% |
| Gemma-3n E4B | ~4B | **69%** | 8% | **4%** | 99% | 33% | **1%** | 1% |

*value-acc = hits / values that existed; halluc = fabrications / truly-absent fields; EM = whole-record; the last
two columns isolate the two hard unit conversions.*

---

## 2. Decode scales cleanly with size

Within the Gemma family, value accuracy climbs monotonically: **17% (270M) → 26% (1B) → 51% (2B) → 69% (4B)**.
More parameters, more of the report read correctly. Whole-record EM, though, only appears at 4B (and only 4%) —
a *perfect* decode is out of reach below ~4B.

---

## 3. The unit-conversion cliff is *universal*

The single most important row-pair. **No model at any size can convert units:** inHg→hPa is **0–1%** and
m/s→kt is **0–2%** across the entire ladder, including the 4B. The models that read hPa-native altimeters
increasingly well (altimeter-all rises to 33% at 4B) still emit the raw inHg number on A-reports.

This is not a scaling problem — it is a **capability none of these models possess**. It confirms, across five
models, that **unit conversion is the finetuning lever**, and it is missing at every rung. A small model that
learned it could leap disproportionately.

---

## 4. Two opposite failure modes at the tiny end

The most striking cross-family finding. At ~0.3–0.5B, the two families fail in **opposite** ways:

- **Gemma 3 270M — reckless.** 67% hallucination: it fabricates a value for almost every absent field, and
  only manages valid JSON 69% of the time. Confident and wrong.
- **Qwen2.5 0.5B — conservative.** It *abstains* on ~60% of fields (outcome mix: 4068 abstain vs 322 hit),
  producing 99% valid JSON and just 1% hallucination — but 5% value accuracy. Safe and empty.

Same size class, opposite safety profiles. "Use a small model" is not one decision — the *family* sets whether
the failure is fabrication or silence. For a safety domain, Qwen's silence is far preferable to Gemma-270M's
confident fabrication.

Hallucination otherwise **shrinks with Gemma size**: 67% → 20% → 10% → 8%. Bigger Gemma, safer Gemma.

---

## 5. The serving stack matters — and it validates the dashi redo

The **E4B scores 69% here (dashi · llama.cpp · Q8_0) vs 77.8% on the M5 (MLX · 4-bit)** — the *same model*,
~9 points apart. Q8 is *higher* fidelity than 4-bit, so this is **not** quantization. The likely cause is the
**stack**: llama.cpp's Gemma-3n implementation (a tricky Per-Layer-Embeddings / MatFormer architecture) being
less faithful than MLX's, and/or a chat-template difference.

The consequence is the point: **the finetune must be measured against a same-stack baseline.** Phase 4 trains
and evaluates on dashi/llama.cpp, so **69% is the number to beat — not 77.8%.** Re-running the baseline here was
the right call; the serving stack is worth ~9 points and cannot be mixed across the before/after.

---

## 6. Implications for Phase 4

- **Target is unchanged and confirmed:** LoRA the unit conversions (inHg→hPa, m/s→kt). The sweep shows the gap
  is universal, so the lever is real at every size.
- **A richer question than before:** *does finetuning help more at small or large sizes?* Can a finetuned 1–2B
  reach base-4B decode? A small model that gains conversion + loses the reckless hallucination would be a strong
  on-device story — the whole point of the small-model focus.
- **Anchor:** base Gemma-3n-E4B on dashi = **69% value-acc / 8% halluc / 4% EM**. Success = beat it on value-acc
  and EM *without* raising hallucination, verified by McNemar on the same 620 records.

---

## 7. Caveats

- **Cross-family confound.** Only Qwen-0.5B is non-Gemma; its low value-acc reflects a *conservative* strategy
  (abstention), not raw incompetence. Read the Gemma rungs for the clean size curve.
- **llama.cpp 3n fidelity.** The 9-point E4B gap suggests llama.cpp's Gemma-3n may be lossy; worth a check
  (chat-template ablation) but it does not affect the internal dashi before/after.
- **One prompt, one quant.** All results are `decode-json-v1` at Q8_0; absolute numbers would shift under a
  prompt or quant change — comparisons within this sweep are apples-to-apples, across stacks are not.
- **Reproduce:** `~/serve.sh <gguf> <alias>` on dashi, then `make harness MODEL=<alias> BASE=http://192.168.0.131:8080`.
