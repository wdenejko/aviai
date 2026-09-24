# Gate 2 — fixed-M loss eval: base vs Gate 1 vs Gate 2

**Date:** 2026-09-24 · raw numbers: `20260924-gate2-behavior-shift.json`

This is the loss-based evaluation that follows the Gate-2 training run. It is **not** the ADR-001
Gate-2 acceptance battery, which needs free-form generation (see "What this does not show").

## Training

| | |
|---|---|
| steps | 4,423 (one epoch, block 2048, rank 4) |
| runtime | 31,295 s (8.7 h), 7.08 s/step throughout |
| mean train loss | 0.6205 (assistant-only; 49.6% of tokens trainable) |
| grad norm | median 0.283; clipped at 1.0 on 94 steps (2.1%) |
| outliers | 6 steps with loss > 3; 5 have tiny grad norms, consistent with near-empty assistant blocks (not verified) |
| thermal governor | 0 pauses |

Smoothed loss by tenth of the epoch: **0.706, 0.638, 0.595**, then flat at 0.587–0.657. In one
epoch every block is scored before the model trains on it, so this curve is effectively held-out
loss — and almost all of the improvement landed in the first ~25%. The remaining ~6.5 hours of data
bought little, which points at the rank-4 adapter's capacity as the binding constraint.

## What was measured

Three states in one model load, on identical blocks, through the same packed-MMQ path: **base**
(freshly injected LoRA, B=0), **Gate 1** (masked pilot adapter, 390 steps) and **Gate 2** (4,423
steps). Loss is reported over every token (`full`) and over assistant tokens only (`assistant` —
what the model actually generates).

**Held out means never trained on by either adapter** — excluded if present in the pilot mixture
*or* the Gate-2 mixture, by a hash of the `messages` content. The Gate-2 pools were re-streamed from
the same sources as the pilot's, so without the pilot exclusion Gate 1 would have been scored partly
in-sample. Recall of the exclusion was checked pool by pool: every row the pilot trained on is found
(targetA 295/295, targetC 20/20, tulu3 627/627, gretel 569/569, opencoder 499/499, jupyter 50/50,
swe 21/21).

| bucket | blocks | assistant tokens | what it is |
|---|---|---|---|
| targetA_heldout | 16 | 43.6% | Target A rows neither adapter trained on |
| targetC_heldout | 14 | 11.6% | Target C trajectories neither adapter trained on |
| general_heldout | 12 | 66.4% | tulu3 replay rows neither adapter trained on — **the forgetting control** |
| sql / code / dsnb / swe_heldout | 16/16/16/14 | 34–73% | never-trained breadth rows |
| targetA / targetC_trained | 16 / 15 | 45% / 11% | rows Gate 2 trained on (memorisation reference) |

`general_heldout` is new: Gate 1 could only report a *trained* tulu3 reference, whose delta mixes
memorisation into any forgetting signal.

## Results — assistant-only loss

| bucket | base | Gate 1 | Gate 2 | Gate 1 → Gate 2 |
|---|---|---|---|---|
| targetA_heldout | 0.7009 | 0.1592 | **0.0771** | −51.6% |
| targetC_heldout | 0.7823 | 0.2776 | **0.1385** | −50.1% |
| general_heldout | 2.0483 | 0.9681 | 0.9360 | −3.3% |
| sql_heldout | 0.6194 | 0.4190 | 0.3731 | −11.0% |
| code_heldout | 0.3239 | 0.1958 | 0.1654 | −15.5% |
| dsnb_heldout | 0.8200 | 0.5419 | 0.4602 | −15.1% |
| swe_heldout | 1.2878 | 1.0940 | 0.9402 | −14.1% |
| targetA_trained | 0.6613 | 0.1338 | 0.0689 | −48.5% |
| targetC_trained | 0.8712 | 0.3162 | 0.1465 | −53.7% |

Full-sequence loss moves the same way everywhere **except targetA**, where it rose from Gate 1 to
Gate 2 (held-out 0.499 → 0.574, trained 0.490 → 0.569) while assistant loss halved (finding 5).

## Findings

**1. Gate 2 roughly halves target loss relative to Gate 1 — but only targetA is a clean volume
comparison.** Both adapters trained on the same Target-A trap families, with 295 vs ~3,392 rows, so
targetA's −52% is the effect of more data. Target C's −50% is confounded: all 13 held-out Target-C
rows come from the **five families added for Gate 2**, which Gate 1 never saw. For Gate 1 that bucket
measures transfer to unseen task types; for Gate 2 it measures new datasets of seen types.

**2. The general control shows no forgetting, and the global shift saturated early.** Against base,
never-trained general data drops 54% — the same across-the-board format/style shift Gate 1 found, not
a capability gain. But Gate 1 already absorbed it: from Gate 1 to Gate 2 general loss moves 3%, within
noise for 12 blocks. That is why Gate 1 is the right baseline here — it cancels the global shift and
leaves the target-specific change.

**3. Breadth buckets improve a further 11–16%, all in the same direction — not individually
established.** This run recorded bucket means only, so there is no variance to judge a 15% difference
against. The eval tool now records per-block losses, so the next run can pair them.

**4. No memorisation gap.** Held-out and trained sit together for both targets (targetA 0.077 vs
0.069; targetC 0.139 vs 0.147). The generalisation is **within family**: Target C held-out rows are
new datasets for seen prompts, and Target A rows reuse paraphrased questions.

**5. targetA full-sequence loss rose while assistant loss halved.** Under assistant-only loss the
prompt tokens get no gradient, so longer training can drift how the model predicts them. It shows up
in targetA because its prompts (question plus schema) are a large share of each block. The model
never generates prompts, so this is the expected trade-off of masking rather than a regression — but
it is a reason not to compare Gate-2 `full` numbers with Gate 1's originally reported ones.

## What this does not show

- **Capability.** Lower loss is not a benchmark gain. ADR-001's Gate-2 acceptance needs DS-1000,
  BIRD-dev, IFEval, MMLU-Pro, GPQA, LiveCodeBench and HumanEval+, all generative, and this stack runs
  at 1.62 s/token. **Gate 2 remains formally open** until the adapter can be served under llama.cpp
  (GGUF export).
- **Transfer to new task types.** Every held-out bucket is within-family.
- **Error bars.** Means only; see finding 3.
- **Template fitting vs skill.** Target A is highly templated; a 0.077 loss may partly be the model
  learning the templates. Only generation on fresh problems can separate the two.

## Next

1. **GGUF export** of the adapter — unblocks the actual Gate-2 acceptance battery.
2. **Rank for Gate 3.** The plateau at ~25% of the epoch suggests rank 4 is the constraint; rank 16
   needs the AITER GMM tuning campaign the recipe lacks.
3. **Validation habit in Target C.** Only 14 of 41 kept `mlc_energy_load` trajectories run a temporal
   holdout (see the Gate-2 mixture report); require or filter on it for Gate 3.
