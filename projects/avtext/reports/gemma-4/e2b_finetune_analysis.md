# E2B finetune analysis — teaching Gemma-4 the conversion

**Base:** `gemma-4-E2B-it` Q8_0 · **Adapter:** bf16 LoRA r=16, 1 epoch, 3000 examples · served on
dashi (llama.cpp) · **Eval:** `eval/v1` — 620 records, `sha256:60870798…` · **Prompt:** `decode-json-v1`
· temp 0. Rendered: `e2b_finetune_analysis.html`.

This is the study's **first result on real Gemma-4** (see ADR-009 — earlier work conflated Gemma-4 with
Gemma-3n; that work is archived under `reports/gemma-3/`).

---

## 1. Executive summary

One epoch of LoRA on 3,000 conversion-heavy examples takes base Gemma-4 E2B from a strong *reader* that
can't do arithmetic into a decoder that converts and abstains correctly.

| metric | base | finetuned | Δ |
|---|--:|--:|--:|
| value accuracy (recall) | 69.5% | **92.9%** | **+23.5** |
| hallucination rate | 31.0% | **0.8%** | **−30.1** |
| whole-record EM | 2.4% | **38.2%** | **+35.8** |
| JSON-valid | 98.9% | 100% | +1.1 |

**McNemar (paired, per-field value correctness):** b=3 regressions, c=1437 fixes, p ≈ 4.6×10⁻³¹².
**McNemar (whole-record EM):** b=1, c=223, p ≈ 2.4×10⁻⁴⁹. As unambiguous as a paired test gets.

The improvement is **uniform across difficulty** (clean +24.2, dissent +23.4, parse_fail +20.6), so it's a
real skill, not overfit to easy records.

## 2. What it fixed

Base errors concentrated in two learnable places:

- **Unit conversion** — the base converted almost nothing (inHg→hPa, m·s⁻¹→kt, SM→m). The finetune now
  *attempts every conversion*; often exactly right (`A2994`→1014, `16005MPS`→10kt, `10SM`→16100m).
  Precision is capped by model size — a 2B does the multiply approximately, so `A2979` can land on 1014
  (should be 1009). Faithful *approximate* converter → value acc climbs to 93%, EM stops at 38%.
- **Fabricated `wind_gust`** — 167 of the base's 220 hallucinations (76%) were invented gusts. The finetune
  emits `null` correctly, which is most of the −30 pt hallucination drop.

## 3. The serving bug that hid the win

The first finetuned eval scored only **73.5% JSON-valid** — the model emitted `<|turn>model` as literal
text on 164/620 records. Root cause: Gemma-4's chat template is a large tool-calling template, and
llama.cpp's jinja engine (**minja**) renders it subtly differently than HF Transformers did at training.
A base model tolerates the drift; a hard-finetuned model (loss 0.06) overfits to the exact training prompt
and breaks. Fed the exact training format via raw `/completion`, JSON-valid → **100%**, value acc → 92.9%.

Fix: harness gained a `--completion` path (`models.completion_predictor`) that reproduces the training
wrapper byte-for-byte. **Lesson: a before/after must never cross serving stacks — and "same template,
different engine" is a different stack.**

## 4. The ceiling — fidelity, not superiority

Every reference in `eval/v1` comes from the parser consensus + NOAA gold. By construction the LLM's ceiling
**is** the parser (~99.9%); no subset here can show the LLM *beating* it. `parse_fail` in this eval means
"≥1 parser choked but the consensus held," not "unparseable" — its reference is still parser-derived. So:

- **As a production decoder:** not viable — the parser wins on accuracy, speed, cost, and zero hallucination.
- **As a learning study:** near-ideal — perfect ground truth, a crisp skill, a textbook before/after.
- **The frontier where an LLM could win** (non-conforming input the parser gets wrong, vs. human gold) is
  *not in this dataset* — the next investment worth making.

## 5. Recipe (reproducibility)

- Base `unsloth/gemma-4-E2B-it` (5.15B, Dense+PLE); Unsloth 2026.7.6 native `Fast Gemma4 patching`.
- LoRA r=16, α=32, dropout 0, targets q/k/v/o/gate/up/down_proj → 25.3M params (0.49%), language tower only.
- 3000 pairs from the disjoint train split; lr 2e-4, batch 8×2, 1 epoch (188 steps); loss 1.73→0.15;
  ~34 min on dashi (Radeon 8060S / gfx1151 / ROCm 7.2).
- LoRA→GGUF (`convert_lora_to_gguf.py`, 50.7 MB) → base Q8 + `--lora` on llama.cpp → eval via `--completion`.

Runs: base `reports/gemma-4/runs/20260730T111552Z-gemma-4-e2b-it-q8/`, finetuned
`…/20260730T130553Z-gemma-4-e2b-it-q8-lora/`. Compare: `python -m avtext.harness.compare <base> <finetuned>`.
