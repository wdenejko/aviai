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
