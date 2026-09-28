# Training at 4096 and 8192 tokens: what was keyed on 2048, and what a longer step costs

**Date:** 2026-09-28 · raw numbers: `20260928-seq4096-enablement.json` · patches:
`patches/torch-ggml-ops-seq4096.patch`, `patches/recipe-seq4096-gmm-configs.patch` · box scripts:
`patches/seq4096/` · logs and audit reports: `~/benchlab/runs/2026-09-28-qwen36-seq4096/` on dashi

**Status: seq 4096 enabled and validated; seq 8192 needed no change.** The adapter trained at rank 4
with batch 1 now runs at 4096 or 8192 tokens per step, with complete gradients, at 239 and 182
tokens/s (296 at 2048). Memory is not the constraint: 17.8 GiB at 8192.

## Why

ADR-001's retrain trains with thinking on (Gate 2 item 5), and a prompt plus a real reasoning trace
plus the answer rarely fits in 2048 tokens. Gate 0 had found the recipe fails closed at 4096:
`torch-ggml-ops: MMQ ... unsupported exact deployment key ... M=4096`.

## What was keyed on 2048

Three tables in the stack accept exact shapes only. None of them is keyed on the sequence length;
all are keyed on token counts, and those were tuned for batch 1, 4 and 16 at 2048 tokens:

| Table | Key | Tuned values | A batch-1 seq-4096 step needs |
|---|---|---|---|
| torch-ggml-ops MMQ bundle, ordinary (dense) kernels | M = tokens per step | 2048, 8192, 32768 | M = 4096 |
| torch-ggml-ops MMQ bundle, grouped (MoE) kernels | R = routed rows = tokens × top-8 | 16384, 65536, 262144 | R = 32768 |
| the recipe's AITER GMM/PTGMM configs (LoRA and base expert matmuls) | routed rows | 16384, 65536, 262144 | 32768 |

**So batch-1 seq 8192 already had every shape it needs.** It is batch 4 at 2048 as far as these
tables go (8192 tokens, 65536 routed rows). Attention (the `aiter-flash-attn` Hub kernel) and the
GatedDeltaNet layers (fla) take any length. Only 4096 needed new entries.

## What changed

- **MMQ bundle: 38 new exact keys** (`add_seq4096_keys.py`). Each ordinary M=2048 key got an
  M=4096 twin and each top-8 grouped R=16384 key an R=32768 twin, on the same tuned kernel spec
  as its seq-2048 twin. The DeepSeek-only families (top-6 grouped, fixed-grouped Q8_0) were left
  alone. Built on a box-local branch `seq4096` of `~/src/torch-ggml-ops`: 190 kernels. The 152
  existing ones rebuilt byte-identical to the Gate-0 build, which is also the check that the
  toolchain matches.
- **GMM/PTGMM configs: 18 Qwen entries at 32,768 rows**, each reusing its 16,384-row config. The
  same entries went into torch-ggml-ops' own copy of those tables (`tools/aiter_gmm_heuristics.py`),
  which its test oracle uses. A Triton config sets speed, not results.
- **The recipe's audit** refuses sequences over 2048 because its dataset rows are 2048-token
  blocks. A copy (`make_audit_long_seq.py`) glues consecutive rows into longer sequences.

## Validation: one reused kernel spec was wrong

Every new key ran the checks of torch-ggml-ops' own deployment test
(`validate_seq4096_keys.py`):
- the public and the direct GGTensile route against the library's external reference;
- repeatability;
- dependence on the input.

Each key's seq-2048 twin ran alongside as a control.

| Keys | New (4096 / 32768) | seq-2048 twins |
|---|---:|---:|
| Ordinary, Qwen shapes | 16/16 pass | 16/16 pass |
| Ordinary, DeepSeek shapes | 12 not checkable | 12 not checkable |
| Grouped (Qwen) | 9/10 pass on the reused spec; 10/10 after one change | 10/10 pass |

The DeepSeek-shaped keys draw their test tensors from a DeepSeek-V4 GGUF that is not on the box,
so they could not be checked in either column. No Qwen path uses them.

One key failed on its reused spec: **paired-backward Q3_K at R=32768**, the input gradient of the
experts' gate/up projection in the 20 layers whose experts are Q3_K. The error against the reference
was NRMSE 0.38 (limit 0.04), with 42% of the [32768, 2048] gradient differing. The same spec passes
at R=16384. The R=65536 winner (spec 1) differs from it only in its wave tile (`MIWaveTile` [2,4]
instead of [1,4]) and passes at R=32768, so the key now uses spec 1. **Reusing a tuned spec at a new
row count is a starting point, not a guarantee; the numerical check is what decides.**

The seq-4096 audit had first run on the failing spec, and its losses and clip norm matched the
rerun on spec 1 to the last digit. The two specs keep the same reduction order, so where both are
right they compute identically. Spec 0's failure therefore needs a route pattern that the test
builds and this batch's routing did not. A training batch can route any way, so only the
validated spec is safe.

## Measurements

The recipe's training-step audit: rank 4, batch 1, the I-Mini GGUF base, two steps on pilot-mixture
rows, AdamW 8-bit, gradient checkpointing. The warmed (second) step:

| Sequence | Forward | Backward | Step | Tokens/s | Peak reserved | Loss, step 1 → 2 | LoRA-B updated |
|---:|---:|---:|---:|---:|---:|---|---:|
| 2048 | 2.04 s | 4.78 s | 6.92 s | **296** | 15.2 GiB | 2.138 → 2.070 | 330/330 |
| 4096 | 4.71 s | 12.36 s | 17.15 s | **239** | 16.0 GiB | 1.923 → 1.805 | 330/330 |
| 8192 | 10.86 s | 33.98 s | 44.92 s | **182** | 17.8 GiB | 1.797 → 1.694 | 330/330 |

The 2048 row is the control, and it reproduces Gate 0 to the hundredth of a second: 6.91 s,
forward 2.04 s, backward 4.78 s, loss 2.07, clip norm 4.4. Every length passed the audit's
gradient-completeness check. The loss falls with length because later tokens see more context.

Doubling the sequence more than doubles the backward (×2.6 to 4096, ×2.8 to 8192) and roughly
doubles the forward (×2.3 each time). The likely cause is attention's quadratic term: 10 of the 40
layers are full attention, the other 30 linear-time GatedDeltaNet. This was not profiled. At 8192
every kernel is a tuned winner, so it is not the reused specs.

## What it means for the retrain

| Sequence | Time per 10M tokens | Cost per token vs 2048 |
|---:|---:|---:|
| 2048 | 9.4 h | 1.0× |
| 4096 | 11.6 h | 1.24× |
| 8192 | 15.2 h | 1.62× |

Memory stays under 18 GiB at every length, so the choice is time against how many reasoning rows
fit whole. It depends on the length distribution of the regenerated reasoning traces, which is
the next measurement (ADR-001 Gate 2 item 5). Performance tuning of the 4096 keys (a
`tune_coefficient_prior_gmm.py` campaign, a GGTensile search for the MMQ keys) is possible later;
at 8192 there is nothing to tune.

## Ops notes

- **Build without the venv on PATH.** `source ~/ftgguf/bin/activate` puts TheRock's `hipcc` and
  `amdclang++` first. The bundle builder then paired them with the system `llvm-readobj`, whose
  LLVM-style header has no `Flags:` line, and every artifact failed the gfx1151 check. Gate 0's
  kernels were compiled by the system ROCm 7.2.4 clang; call `~/ftgguf/bin/python` directly
  (`build_ggml_ops_seq4096.sh`). The failed build left the installed kernels untouched: the
  builder stages and installs atomically.
- **A GPU Python process exits 0 on this box.** Once TheRock torch has initialized HIP,
  `sys.exit(3)` ends with status 0. The window's gate on the validation's exit status let a failed
  validation through; it now reads the summary in the output JSON. Any launcher that checks a
  GPU process's exit status has the same hole.
- **Anchor process checks on the executable.** `pgrep -f "llama-server"` matched the launching
  ssh session, whose command line merely contained the word, and the window refused to start;
  `pgrep -f '^[^ ]*/llama-server( |$)'` does not.
- **torch-ggml-ops' test oracle has its own GMM tables.** Extending the recipe's tables was not
  enough to validate the grouped keys: the oracle refused R=32768 until its copy got the same
  entries.
