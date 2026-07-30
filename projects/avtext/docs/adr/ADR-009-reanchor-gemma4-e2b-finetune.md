# ADR-009: Re-anchor the study on *real* Gemma 4; E2B finetune result + serving contract

- **Status:** Accepted
- **Date:** 2026-07-30
- **Supersedes the effective target of:** ADR-007 (dashi roster was gemma-3/3n), and corrects the
  execution drift noted below. ADR-003's *stated* target (`google/gemma-4-E4B-it`) was right all along.

## Context

ADR-003 named the target `google/gemma-4-E4B-it` on day one. But **Gemma 4 was released 2026-04-02 —
after this assistant's Jan-2026 knowledge cutoff** — so across earlier sessions the execution silently
drifted to **Gemma 3n** (E2B/E4B): the dashi size sweep and the first finetune ran `gemma-3n-*` while the
baseline analysis header literally read *"gemma-4-E4B-it (Gemma 3n)"*. The consequence was not cosmetic:
the "77.8% M5 vs 69% dashi stack gap" (ADR-007/size_sweep) compared a **mislabelled gemma-3n on both
sides**, so it was never a clean result. The user caught it when a finetune launched on `unsloth/gemma-3n-E4B-it`.

## Decision

1. **Re-anchor the entire study on real Gemma 4** (`unsloth/gemma-4-E2B-it` / `-E4B-it`), starting E2B,
   all training on dashi. Verified the models exist via the HF API before assuming (the lesson from the
   cutoff miss: a post-cutoff release is invisible to training knowledge — check reality).
2. **Archive, don't delete, all prior gemma-3 work.** Repo: `reports/gemma-3/` (the whole v1 batch, incl.
   the identity-ambiguous M5 run and the Qwen comparison). dashi: `~/models/gemma-3/`, `~/ft/gemma-3/`.
   New runs → `reports/gemma-4/runs/` (runner gained `--out-dir`).
3. **Serving contract for Gemma 4** (two arch facts that bit us):
   - Gemma 4 is a **reasoning model** — served with `--reasoning-budget 0` so it emits the answer directly
     (else it thinks until the token cap and `content` is empty). This matches the finetune's direct-JSON target.
   - A **finetuned** Gemma-4 model is served via the raw `/completion` path with the exact training wrapper
     (`--completion`), **not** `/v1/chat/completions` + `--jinja`. See below.

## The E2B result (the decision's evidence)

Base 69.5% → finetuned **92.9%** value accuracy; hallucination **31.0% → 0.8%**; whole-record EM
**2.4% → 38.2%**. McNemar p ≈ 10⁻³¹² (1437 fixes / 3 regressions). Uniform across clean/dissent/parse_fail.
Full write-up: `reports/gemma-4/e2b_finetune_analysis.{md,html}`.

## Lessons worth encoding

- **Template drift is a serving stack.** llama.cpp renders Gemma-4's tool-calling chat template with its own
  engine (minja); it diverges subtly from HF Transformers' render at training. A hard-finetuned model (loss
  0.06) overfits to the exact training prompt and breaks on the drift — 73.5% JSON-valid via chat, 100% via
  raw `/completion` with the training wrapper. **A before/after must not cross serving stacks; "same template,
  different engine" counts as different.** (→ `harness/models.completion_predictor`.)
- **The eval is a fidelity test, not a superiority test.** All references are parser-consensus + NOAA gold, so
  the LLM's ceiling *is* the parser (~99.9%). `parse_fail` here = "≥1 parser choked, consensus held," not
  "unparseable." No subset can show LLM > parser. Testing the LLM-beats-parser thesis needs new data:
  non-conforming input the automated decoders get wrong, validated against **independent human gold**.
- **Unsloth loads the Gemma-4 PLE arch natively** (2026.7.6, *Fast Gemma4 patching*) via `FastLanguageModel` —
  resolving ADR-008's open question. LoRA→GGUF via upstream `convert_lora_to_gguf.py` works despite the local
  checkouts lacking an explicit Gemma4 converter class (the base config resolves the tensor mapping).

## Consequences

- **+** A valid, same-stack before/after on the study's actual model; a clean size-scaling curve to come (E4B next).
- **+** The serving contract + `--completion`/`--out-dir` harness changes make the redo reproducible.
- **−** The gemma-3 size sweep (ADR-007) is now a *reference*, not part of the main result — different model family.
- **Open:** the messy-tail / human-gold eval (to test LLM > parser) and scaling the corpus beyond 51 stations.
