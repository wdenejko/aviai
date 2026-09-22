# Gate-1 behavior-shift eval — fixed-M=2048 forward loss

**Date:** 2026-09-22 · **Adapter:** `out_qwen36_35b/final` (rank-4 LoRA, 398 steps, 1 epoch of the
Gate-1 pilot) · **Base:** Qwen3.6-35B-A3B APEX-I-Mini GGUF

## Why this method
Free-form generation is impossible on this stack: the `torch-ggml-ops` MMQ bundle is compiled only
for the training geometry (M=2048), so prefill/decode at other M fails. We therefore measure the
fine-tune with **fixed-M=2048 forward loss**, the one shape the bundle supports.

To avoid a path/numerics confound (unwrapped base = generic dequant path; wrapped = packed MMQ), the
model is built **once** with the full recipe injection. Freshly injected LoRA has **B=0**, so it is
numerically the base model *while already running the MMQ path*. We measure, then load the trained
weights into that same model and measure again — the only change is the adapter.

## Results

| bucket | base | adapter | delta | reduction |
|---|---|---|---|---|
| targetA_heldout | 1.283 | 0.164 | -1.119 | **-87.2%** |
| targetA_trained | 1.367 | 0.161 | -1.206 | -88.2% |
| targetC_heldout | 3.534 | 0.339 | -3.196 | **-90.4%** |
| targetC_trained | 3.470 | 0.423 | -3.047 | -87.8% |
| tulu3_trained | 2.906 | 1.198 | -1.708 | -58.8% |

**Generalization gap** (held-out drop vs trained drop): targetA **1.0 pp**, targetC **2.6 pp**.

## Reading
- Held-out rows (never trained: 3545 targetA / 38 targetC available) improve **as much as** trained
  rows. The tiny gap means the adapter learned the target *patterns*, it did not merely memorize.
- targetC's base loss was high (3.53 -> ppl ~34) because long tool-call trajectories are an
  unfamiliar format; much of its -90% is learning trajectory **format/schema**, which is part of the
  target but should not be read as "better data-science reasoning".
- targetA is **templated synthetic** data (families x dialects x tables), so it is low-entropy and
  large drops are expected. This is within-distribution generalization, not broad capability gain.

## Limitations (explicit)
1. **No clean general-capability control.** `tulu3_trained` rows WERE in training, so its -58.8%
   includes memorization. The breadth pools were not persisted locally, so there is no never-trained
   general bucket. This eval therefore says **nothing** about capability retention / forgetting.
2. **Not a downstream task measurement.** Loss is not task success; the dsbench agentic benchmark
   needs generation, which is blocked until the MMQ bundle is regenerated for inference shapes.

---

# Addendum: forgetting control (never-trained general data)

Limitation 1 above is now closed. A fresh `tulu3` slice was re-acquired (700k tok / 1756 rows); the
pilot's 627 rows were removed by content hash — the overlap was **exactly 627**, confirming the HF
stream is deterministic and the split is exact. The remaining 1129 rows are general instruction data
the adapter has **never seen**.

| bucket | base | adapter | delta (abs) | reduction |
|---|---|---|---|---|
| targetA_heldout | 1.295 | 0.158 | -1.137 | -87.8% |
| **tulu3_heldout** (never trained) | 2.515 | 1.288 | **-1.227** | -48.8% |
| tulu3_trained | 2.774 | 1.299 | -1.476 | -53.2% |

## Findings
1. **No catastrophic forgetting.** General held-out loss *decreased* (2.515 -> 1.288). The fine-tune
   did not degrade general modelling on data it never saw.
2. **Reproducible.** `targetA_heldout` measured -87.8% here vs -87.2% in the main run (0.6 pp).
3. **A large part of the headline number is a GLOBAL shift, not target-specific learning.** In
   absolute nats the never-trained general bucket improved slightly *more* (-1.227) than the target
   bucket (-1.137). Much of the -87% therefore reflects the model adapting to the pilot's rendering
   and packing, not specifically learning SQL-dialect conventions.
4. What survives as target-specific is **sharpness**: targetA lands at 0.158 (near-deterministic)
   while general data remains at 1.288. The target distribution is modelled far more confidently.

**Consequence for Gate 2:** a loss-only eval cannot separate format adaptation from capability gain.
The agentic benchmark (generation) is the measurement that can.

---

# Addendum 2: generation unblocked (limitation 2 closed)

Limitation 2 said the agentic/generation evidence was unreachable. It is now reachable.

## Why the obvious fix does not work
The MMQ bundle compiles only the TRAINING geometry — dense `M in {2048, 8192, 32768}` and
grouped-pair `r in {16384, 65536, 262144}` — and `exact_record()` does an exact-match lookup.
Generation presents `M = prompt_length` on prefill and `M = 1` per decode step. **Regenerating the
bundle cannot fix this in general**: prefill M is unbounded, so it would need a kernel per possible
prompt length.

A generic-dequant fallback was added for the *dense* path (see
`patches/recipe-fast_lora-mmq-generic-fallback.patch`), but the MoE expert path
(`grouped_mmq_pair`) has no generic counterpart in the recipe, and swapping to a stock transformers
experts implementation would **silently drop the expert LoRA**, making any comparison unfaithful.

## What does work: a fixed 2048-token window
`grouped-pair r=16384` is exactly 2048 tokens x top-8, so decoding inside a fixed 2048-token window
(right-padded, `use_cache=False`, next token read from the logit at the last real position) keeps
*every* op on a compiled shape with the **complete** adapter applied. Cost: one full-window forward
per token (~2s). Implemented in `dsbench.sftgen.gen_fixed_window`.

## Observed behaviour (greedy, base = B=0 vs trained adapter)

**Prompt: DuckDB flights-per-day for the last 7 days** (150 tokens)
- base: deliberates and never commits — "*I can use `date_trunc('day', ...)` or `dep_time::DATE`* ...
  `INTERVAL '7 days'` ... *Actually, 'last 7 days' usually means...*"
- adapter: commits and emits SQL — ```SELECT date_trunc('day', dep_time) AS flight_day, COUNT(*)```
  filtering with **`INTERVAL '7' DAY`** (canonical dialect form) rather than the base's
  `INTERVAL '7 days'`. Consistent with Target A (SQL-dialect date/time conventions).

**Prompt: steps before reporting a final accuracy number** (40 tokens)
- base: chain-of-thought preamble — "*Okay, so I'm trying to figure out... Let me start by breaking
  down the*"
- adapter: closes the thinking block and answers directly — "*`</think>` Here are the concrete steps
  ... 1. Data Exploration and Understanding - Load the CSV file and examine its structure*".
  Consistent with Target C (agentic ML-delivery discipline).

**Caveat: n=2 prompts, greedy, short budgets. This is qualitative corroboration, not a benchmark.**
The real measurement is the dsbench agentic suite, which this decoder now makes possible (at ~2s per
token, so it suits a small problem set rather than a k=5 sweep).
