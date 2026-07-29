# ADR-003: Target model = Gemma 4 E4B; Jan-2025 cutoff drives the test split; fine-tune in cloud, evaluate locally

- **Status:** Accepted
- **Date:** 2026-07-29

## Context

The whole before/after methodology hinges on the exact target model and its data
cutoff. Two things had to be pinned before eval design:

1. **Model identity & cutoff** — verified against Google's official Gemma 4 model
   card and the Hugging Face card on 2026-07-29:
   - `google/gemma-4-E4B-it` — **4.5B effective params** (8B total w/ Per-Layer
     Embeddings), **128K context**, text+image+audio, **Apache-2.0**.
   - **Training data cutoff: January 2025.**
   This confirms the plan's "test set drawn post-Jan-2025" is correct, not an
   assumption.
2. **Local hardware** — Apple **M5 / 24 GB**. Below the ~32 GB practical floor
   for local MLX fine-tuning of this model, but ample for local *inference* of a
   Q4 GGUF (~4–5 GB).

## Decision

- **Target model:** `google/gemma-4-E4B-it` for both base baseline and the
  fine-tune. (Multimodality is irrelevant here — the task is text-only.)
- **Time split:** every eval item's observation time is **strictly after
  January 2025**. Enforced in `src/avtext/tasks/`. This is the primary defense
  against the base model having memorized specific observations.
- **Compute split:** **fine-tune in the cloud** (Colab T4 free / rented 4090),
  **evaluate locally** on the M5 via `llama.cpp` (`unsloth/gemma-4-E4B-it-GGUF`
  exists; `llama-cli` is installed). Local MLX fine-tuning is out of scope given
  24 GB.

## Consequences

- **+** The reproducibility block can cite an exact checkpoint + cutoff.
- **+** Cheap, reproducible local eval; cloud used only for the few training runs.
- **−** Time-split shrinks the eval pool to post-Jan-2025 data — which is exactly
  why the scheduled collector starts on day 1 (it accumulates post-cutoff data
  while the harness is built).
- **Caveat to revisit in Phase 3:** a time split stops the model *memorizing*
  specific METARs, but METAR/TAF are highly templated — the base model can still
  "decode" post-cutoff strings from pre-cutoff *format* knowledge. So the honest
  measure of fine-tuning benefit is weighted toward **hard/messy cases, rare
  tokens, and abstention**, not clean decodes. The eval report must reflect that.

## Alternatives considered

- *Assume the plan's cutoff without checking.* Rejected — the split's validity
  depends on it; the model post-dates this assistant's own knowledge cutoff, so
  verification was mandatory.
- *Local MLX fine-tuning.* Deferred — needs ≥32 GB; revisit on different hardware.
