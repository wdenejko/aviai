# Patches to the upstream training recipe

These apply to `woct0rdho/transformers5-qwen3.5-recipe`, cloned on the box at
`~/src/transformers5-qwen3.5-recipe`. They are kept here so the box is reproducible; upstream is not
vendored into this repo.

## `recipe-fast_lora-mmq-generic-fallback.patch`
The `torch-ggml-ops` MMQ bundle is generated for the TRAINING geometry only (dense M in
{2048, 8192, 32768}; grouped-pair r in {16384, 65536, 262144}). Anything else raises
`unsupported exact deployment key`. That makes ordinary generation impossible: prefill presents
M = prompt length and each decode step M = 1.

Regenerating the bundle cannot fix this in general — prefill M is unbounded, so you would need a
kernel per possible prompt length. Instead this patch lets a *dense* projection fall back to the
generic compiled-dequant base forward (`base(x)`, the same path the GDN projections already use)
when a shape has no compiled kernel, caching the decision so it costs one exception per shape.

Training is unaffected: M=2048 is compiled, so it still takes the MMQ path.

NOTE: this only covers the dense path. The MoE expert path (`fast_moe_lora.py`, `grouped_mmq_pair`)
has no generic counterpart in the recipe, and swapping to a stock transformers experts
implementation would silently drop the expert LoRA. For faithful generation use the fixed-2048-token
window decoder (`dsbench.sftgen.gen_fixed_window`) instead, which keeps every op on a compiled shape.

## `recipe-train-assistant-only-loss.patch`
The recipe's `fixed_length_lm_collator` built labels from `input_ids`, so loss covered the whole
packed block. Our records carry `loss_mask_roles: ["assistant"]`, and only 53.7% of tokens are
assistant-authored — so 46.3% of the gradient was spent predicting prompts/tool output/system text.

This patch makes the collator use dataset-provided `labels` when present (falling back to the old
behaviour when absent, so it is safe for the recipe's own datasets). Build the labelled dataset with
`dsbench.sftgen.build_masked_dataset`, which needs `remove_unused_columns=False` — already set.

Measured effect (see reports/gate-evals/20260922-gate1-masking-ab.md): assistant-only loss improves
on both targets (targetC by 20%), general capability unchanged.

## `recipe-train-overridable-paths.patch`
`dataset_dir` and `output_dir` were hardcoded to `data_tokenized_qwen3.5` and `out_qwen36_35b`, so
training a second gate would have overwritten the first gate's tokenised dataset AND its adapter —
including the checkpoints needed to compare the two. Discovered on the way into Gate 2, with the
Gate-1 adapter (`final` plus checkpoints 100/200/300/390) still sitting in that directory.

Adds `QWEN35_DATASET_DIR` and `QWEN35_OUTPUT_DIR`, matching the existing `QWEN35_*` override style.
Defaults are unchanged, so the recipe's own invocation still works.

    QWEN35_DATASET_DIR=~/data_tokenized_gate2 QWEN35_OUTPUT_DIR=~/out_qwen36_35b_gate2 \
        python train_qwen3_5_35b.py
