# Gate-2 adapter → llama.cpp: GGUF export, verified and served

**Date:** 2026-09-24 · raw numbers: `20260924-gguf-export.json` · exporter:
`src/dsbench/sftgen/export_lora_gguf.py`

**Result:** the Gate-2 adapter runs in the production llama.cpp fork as a runtime LoRA, at
**~66 tok/s** generation versus the training stack's 1.62 s/token — roughly 100× faster. That makes
ADR-001's generative Gate-2 acceptance battery runnable.

## Why runtime LoRA, not merge-and-requantize

A runtime LoRA (`llama-server --lora`) serves exactly the function that was trained: the quantized
base plus the delta. Merging the delta into the weights and re-quantizing to the ~3-bit APEX I-Mini
recipe would add rounding noise comparable in size to a rank-4 delta. The cost is speed: the adapter
takes generation from ~80 to ~66 tok/s.

## What the export has to get right

The adapter is 660 bf16 tensors, 235M parameters, rank 4, alpha 4 (scale 1.0). Most tensors map
straight across: PEFT's `A [r, in]` / `B [out, r]` is the orientation llama.cpp validates, including
the 3-D expert tensors. llama.cpp's `convert_lora_to_gguf.py` cannot be used, because the expert LoRA
uses the recipe's own layout. Two transforms are non-trivial, and both would fail **silently**: the
adapter would load, run, and simply be wrong.

1. **Value-head order in the 30 GDN linear-attention layers.** llama.cpp's converter reorders value
   heads from grouped to tiled order (`conversion/qwen.py::_reorder_v_heads`). The training stack
   undoes that on load (`TiledToGroupedRows`, `TiledToGroupedInputs`), so the adapter was trained in
   grouped order and must be permuted back: the V block of `in_proj_qkv.lora_B` (rows 4096+), all of
   `in_proj_z.lora_B`, and the **columns** of `out_proj.lora_A`. The last one needed care: the
   recipe keeps `out_proj` packed and permutes its *input* inside `GgufLinear.forward`, but its fused
   LoRA path (`_fused_lora_add(result, x, …)`) feeds the factors the un-permuted input — so `lora_A`
   is grouped too.
2. **Expert gate/up split.** The recipe applies one A to the fused projection and splits the output
   with `chunk(2, dim=-1)` — gate first. llama.cpp stores `ffn_gate_exps` and `ffn_up_exps`
   separately, so B is split and A is shared.

Also checked and ruled out: the recipe's `fast_moe_ranking` and `expert_prior` do not reorder the
expert axis. llama.cpp's loader does not validate that axis (`ne[2]`), so the exporter asserts 256
experts itself.

## Verification, three ways

**Unit tests** (`tests/test_export_lora_gguf.py`) re-implement both stacks' own operations — the
llama.cpp converter's reorder and the recipe's load-time inverse — and check the exporter against
them: the round trip is the identity, the Q/K rows never move, and the column permutation keeps
`out_proj` exact on tiled input.

**Ablation.** Deliberately broken exports must lose to the correct one on identical held-out text
(`llama-perplexity`, ctx 2048, same build; lower is better):

| text | base | **correct** | no V-perm | gate/up swapped |
|---|---|---|---|---|
| Target A | 1.340 | **1.231** | 1.224 | 1.286 |
| Target C | 2.065 | **1.915** | 1.935 | 1.935 |
| general | 4.301 | **3.554** | 3.579 | 3.599 |

The gate/up swap is clearly worse (~6 standard errors on Target A). The V permutation was not
separable here: the V-permuted tensors are only ~14% of the adapter's weight change (the routed
experts are 71%), and the identical remainder dilutes the difference below the noise.

**Isolated V-permutation test** — GDN-only adapters, nothing else to dilute them:

| text | base | GDN-only **correct** | GDN-only no V-perm |
|---|---|---|---|
| Target A | 1.340 | **1.327** (−1.0%) | 1.337 (−0.2%) |
| Target C | 2.065 | **1.970** (−4.6%) | 2.039 (−1.3%) |
| general | 4.301 | **3.700** (−14.0%) | 4.122 (−4.2%) |

The permuted version wins on every text, by ~3–5 standard errors on the two diverse ones, with 3.3×
the improvement on general text. Code reading, unit tests and measurement agree.

## Served by production llama-server

The production build (`build-v2`, commit `646b10b96`) loads the adapter and switches it per request.
Same server, greedy decoding, thinking off, held-out Target A questions:

| | adapter off | adapter on |
|---|---|---|
| generation | ~80 tok/s | **~66 tok/s** |
| prompt processing | ~300 tok/s | ~230–270 tok/s |

The answers move toward the trained conventions. On the timezone-direction trap the base model writes
`AT TIME ZONE edge_city` — a city name used as a timezone, which PostgreSQL rejects — while the
adapter maps each city to the hours to *add* (New York +4, Chicago +5, Denver +6, LA +7), matching
the reference. On the weekend question it switches to the trained `DOW IN (6, 0)` form; the base's
`ISODOW IN (6, 7)` was also correct, so that one is style rather than a fix.

## A data flaw this exposed

With the adapter, the model ends SQL answers with **"Answer: 431"** when the reference is 444: it
states a count it cannot know without running the query. Target A training rows end with the
execution-verified number, so the model learned the format without the execution. In an agentic
setting it would run the SQL; in plain chat it fabricates. For Gate 3, either drop the number from
Target A targets or make the trajectory show the query being executed before the number appears.

## Operational notes

- `build-v2`'s `llama-perplexity` segfaults on startup: the binary dates from Sep 3, its `libllama`
  from Sep 6. The ablation used `build-v2-85cc-bak`, whose tool and library were built together.
  Rebuilding inside `build-v2` would have changed production: the source tree is now ahead of the
  server's commit.
- `llama-perplexity` tokenizes without parsing special tokens, so the held-out text was rendered as
  plain role-labelled transcripts. That is fine for comparing variants, but its absolute numbers are
  not comparable with the training stack's chat-format loss.

## Next

Run the ADR-001 acceptance battery against this server: base vs adapter, per-request switching on
one server load.
