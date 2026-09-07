# Squeezing LoRA training out of Strix Halo (gfx1151) — the evidence and the recipe

> **Question (S24, 2026-09-08):** what is the fastest, most effective way to run our Gemma-4-E4B
> rank-16 LoRA fine-tunes on dashi (Ryzen AI Max+ 395 / Radeon 8060S = gfx1151, 123 GiB unified)?
> **Method:** local measurements on the box (profiler, micro-benchmarks, sweeps) + three parallel
> web-research passes (GitHub issues/PRs, AMD/ROCm docs, HF, community guides; ~50 dated sources).
> Every gain below is labelled **measured** (on dashi) or **estimated** (from evidence elsewhere).

## Bottom line

Starting point: **0.30 examples/s (~40 s per optimizer step) → ~48 h per epoch** on the 49k set, with
plain `transformers`+`peft`+`trl`+SDPA. The step is **not FLOP-bound** — it is padding-, fusion- and
launch-bound — so most of the recoverable speed is software, plus one hardware setting only the
owner can flip.

| # | Lever | Gain | Confidence | Who |
|---|---|---|---|---|
| 1 | **Length-grouped batching** (`--group-by-length`) | **~2x** (49 % of linear compute was padding → 1 %) | measured on our data | done in trainer |
| 2 | **Performance power mode** (P-mode button: PL1 85 W → 120 W; GPU 2175 → ~2787 MHz) | up to ~1.28x on GEMM/attention share; ~1.1–1.2x on the step | strong external evidence (same box model) | **owner** |
| 3 | **torch.compile** (`--compile`, inductor default mode) | **1.42x** on a real step | measured | done in trainer |
| 4 | **No gradient checkpointing** (`--no-grad-checkpointing`) | ~1.3x; memory-safe once batches are length-grouped (except the TAF bucket) | measured | flag exists; validate memory |
| 5 | Launch-overhead env knobs (`HIP_FORCE_DEV_KERNARG=1`, `expandable_segments`) | 0–30 % | external (launch-heavy diffusion workload) | A/B pending |
| 6 | Bigger batch **with** grouping | unknown — earlier "worse" result was confounded by padding | to measure | A/B pending |

Composed estimate for 1+3+4 (+2): **~3.5–4.5x → the 49k epoch from ~48 h to ~10–14 h**, i.e. an
overnight run. See *Measured composed results* at the end for the sweep numbers as they land.

## Where the 40 s/step actually went (torch.profiler, one real microbatch, batch 6, ~500 tok)

| Bucket | Share | Notes |
|---|---|---|
| Base-model GEMMs (big Tensile tiles) | ~33 % | the irreducible core; hipBLASLt already in use |
| **Unfused elementwise + copies** (`mul`/`add`/`copy_`, ~9k launches) | **~35 %** | bandwidth waste; what compile fuses |
| **Rank-16 LoRA GEMMs** (MT32x32 tiles, ~1,600 launches) | **~12 %** | 0.44 % of params eating 12 % — launch-bound |
| Attention | ~15 % | EFFICIENT for the 35 sliding layers; MATH for the 7 head-dim-512 layers |

CPU time ≈ GPU time (~15k kernel launches per microbatch) — the classic launch-bound profile.
Raw ceiling: **27.9 TFLOP/s bf16 sustained** at 2175 MHz (→ ~37 at 2900 MHz, matching community
hipBLASLt numbers). We were at ~200 tok/s; the community-estimated fused-kernel ceiling for an 8B
LoRA on this chip is ~500–900 tok/s.

And the batch-length distribution that made padding the #1 lever (6.1k-row sample):
METAR p50 460 tok (mostly the prompt preamble) · TAF p50 1131 / p90 1497 · NOTAM-ext p50 315 ·
NOTAM-cls p50 143. Random batches: computed/real tokens = **1.96x**; length-grouped: **1.01x**.

## The recipe (what to run)

```bash
# inside toolbox llama-rocm-7.2.4_2, venv ~/fttorch (TheRock gfx1151 torch 2.12+rocm7.13)
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1   # required on this AOTriton 0.11.2b wheel (before 1st SDPA call)
export CC=$HOME/fttrain/bin/zigcc CXX=$HOME/fttrain/bin/zigcxx   # no-root C compiler for Triton/inductor
export PYTHONUNBUFFERED=1
# A/B candidates (pending): HIP_FORCE_DEV_KERNARG=1  PYTORCH_ALLOC_CONF=expandable_segments:True  TORCH_BLAS_PREFER_HIPBLASLT=1
~/fttorch/bin/python ~/scripts/avtext/train_lora_peft.py \
  --model unsloth/gemma-4-E4B-it --data ~/fttrain/train_all_lg.jsonl --out ~/fttrain/adapter-all-lg \
  --rank 16 --epochs 1 --batch 6 --grad-accum 2 --max-seq 2048 \
  --group-by-length --compile --no-grad-checkpointing      # drop --no-grad-checkpointing if the TAF bucket OOMs
```

Trainer notes: `--group-by-length` is a `LengthGroupedSampler` injected via a `SFTTrainer` subclass
(trl 1.x dropped `group_by_length`); `--compile` uses `SFTConfig(torch_compile=True)` with
`torch._dynamo.config.cache_size_limit=64` so variable sequence lengths resolve via automatic dynamic
shapes instead of recompiling per shape. First compile ≈ 3.5 min (cached afterwards; set
`TORCHINDUCTOR_CACHE_DIR` to a persistent path to reuse across runs).

**Owner action (hardware):** press the EVO-X2 P-mode button to *Performance* (PL1 120 W). Verify
under load: `pp_dpm_sclk` active state ≈ 2787 MHz and `power1_average` ≈ 119 W. **Watch
`temp1_input`** — an EVO-X2 running training at 120 W was reported rebooting above 90 °C; keep fans
at max and back off (ryzenadj `--tctl-temp=88` or PPT ~105 W) if it sits at 88–91 °C. ryzenadj
cannot raise above the OEM 120 W and its Strix Halo support is partial — the button is the lever.

## Hazards — read before changing anything

- **head_dim 512 has no fused attention on gfx11, by design** (AOTriton maintainer: limited
  LDS/register file; CK and aiter cap at 256). Gemma-4's 7 global layers always run MATH. Nothing
  in the ecosystem fixes this today.
- **Silent-garbage hazard on upgrade:** some 2026 ROCm wheels return *saturated wrong values* for
  head-dim 512 on gfx1151 instead of falling back (pytorch #189849; upstream cap fix 2026-08-28).
  Our wheel correctly rejects → MATH. **After any torch/AOTriton upgrade, re-run the SDPA
  head-dim-512 check** (`sdpa_kernel(FLASH/EFFICIENT)` must raise "No available kernel"; if it
  "succeeds", verify values against MATH).
- **torch.compile numerics on RDNA:** unsloth disabled compiled forward for Gemma-3 on gfx11 after
  NaN losses. Our Gemma-4 compiled step gave a sane loss, but **compare compiled vs eager loss over
  the first few hundred steps** of any real run and keep NaN guards. Triton 3.6/3.7 has known
  gfx1151 miscompiles (fixed in 3.8); `num_warps>4` can assert on RDNA.
- **Never** set `HSA_OVERRIDE_GFX_VERSION` (breaks native gfx1151 kernels) or
  `PYTORCH_HIP_ALLOC_CONF=backend:malloc` (crashes). Keep dataloader `num_workers=0` (Triton
  "invalid device ordinal" in forked workers on gfx1151).
- **`--packing` stays off** with plain SDPA: no varlen flash path → O(flattened_len²) MATH →
  310 s/step (measured) plus cross-sample contamination.

## Ruled out (with evidence)

| Option | Verdict | Why |
|---|---|---|
| Liger-Kernel | **no** | fused-linear-CE **8x slower** than plain matmul+CE on gfx1151 (6044 vs 762 ms, measured); model patches don't load on transformers 5.x; compute-for-memory is the wrong trade here |
| `torch.compile(mode="reduce-overhead")` / hipGraph capture | no | measured 0 gain over default; ROCm cudagraph trees broken/flaky; ~15k-node captures segfault on ROCm |
| QLoRA / 8-bit (bitsandbytes) | no | works on gfx1151 but **1.8x / 12x slower** (community, same chip); memory isn't our constraint |
| Bigger batch **without** grouping | no | batch 32 = 136 s/step (worse) — but confounded by padding; re-test *with* grouping pending |
| `TORCH_ROCM_FA_PREFER_CK`, AITER ASM, FA3/FA4, cuDNN, xformers | no | CDNA- or CUDA-only |
| torchtune / axolotl / LLaMA-Factory | no | unmaintained or MI-series-centric; nothing gfx1151 |

## Frontier (real gains, real effort — not done)

1. **Packing via flash-attn varlen** — CK backend gained RDNA *backward* (Jun/Jul 2026) with gfx1151
   in allowed archs; hdim ≤ 256 only. Needs a multi-hour `GPU_ARCHS=gfx1151` build plus a custom
   `AttentionInterface`: FA2-varlen (`window_size` for sliding) on the 35 hdim-256 layers,
   per-document MATH on the 7 hdim-512 layers. Removes padding entirely and makes sliding layers
   flash-speed at any length. Untested by anyone on gfx1151.
2. **FlexAttention** (`attn_implementation="flex_attention"`) — gfx1151-tuned tiles exist for
   hdim 256; HF's BlockMask handles packing + sliding windows sparsely, the only candidate that
   makes the hdim-512 layers sub-quadratic under packing (needs tiny `kernel_options` blocks).
   Experimental; compile fragility.
3. **unsloth[amd]** — officially "full support" for gfx1151 (Jul 2026; AMD playbook validated on
   gemma-4-E4B-it). Its own claim is ≤ ~1.4x vs TRL; it is the only packaged fix for the 12 % LoRA
   launch bucket (fused fast_lora). Worth an A/B as an alternative stack, not additive with compile.
   *Correction to our folklore:* the "10–30x faster unsloth" memory compared against a mis-configured
   run (packing with no varlen attention); the honest gap vs a tuned standard stack is ~1.3–1.5x.
4. **Newer TheRock nightly (post 2026-08-28)** — AOTriton 0.13b, gfx1151 non-experimental, the
   hdim cap fix; a full gfx1151 attention tuning DB (aotriton PR #205) is still open. Modest gain;
   re-run the head-dim-512 check.
5. **No-mask FLASH trick** — HF drops the attention mask only when there is no padding *and* the
   batch is shorter than the 512-token sliding window; `batch_size=1` + grad-accum would put every
   METAR-length row on FLASH. Probably loses to the 12x launch penalty of sequential microbatches;
   an idea for a custom packed pipeline, not for now.

## Measured composed results

### Sweep 1 — real training steps, 25 per config (batch 6 × accum 2, max_seq 2048)

`LengthGroupedSampler`'s first megabatch is a *random* 300 examples sorted longest→shortest (not the
300 longest), so 25 steps = one descending pass over a representative sample; "steady" = steps
15→25 (the shorter half, past compile/first-step warmup). Baseline for reference: random batching,
same settings, **40 s/step**.

| Config | steady s/step | vs A | total 25 steps |
|---|---|---|---|
| A `--group-by-length` (ckpt on) | 12.0 | — | 608 s |
| B A + `--compile` | 11.8 | ~1.0x | 787 s (incl. ~3.5 min compile) |
| **C B + `--no-grad-checkpointing`** | **7.0** | **1.7x** | 938 s (incl. compile + recompile) |

Readings: (1) compile's static-shape 1.42x mostly evaporates once sequence lengths vary — dynamo's
automatic dynamic-shape graph fuses less than the static graph did; a bucketed-padding strategy
(`pad_to_multiple_of`, static shapes per bucket, cached compiles) is the obvious follow-up.
(2) Dropping gradient checkpointing is worth 1.7x on top (no recompute *and* a simpler graph), and
**it did not OOM on the 2,048-token bucket at batch 6** — length grouping made it safe.
(3) Throughput on the short half in C ≈ 1,000 real tok/s, inside the community's fused-kernel
ceiling range for this chip.

**Numerical parity (the RDNA/Gemma NaN hazard) — passes.** Logged loss / grad_norm at steps 20 and
25: A 0.830 / 2.285 / 1.70 · B 0.828 / 2.278 / 1.76 · C 0.820 / 2.272 / 1.78; zero NaN lines in any
log. Compiled configs match eager to ~1 % (bf16 fusion-reordering noise). Config C is therefore a
trustworthy production setting.

_Per-length cost curve → expected full-epoch time per config (`perlen.py`)_ — **pending**.
_Sweep 2: launch-overhead env knobs; batch 12/24 with grouping_ — **pending**.

## Key sources

ROCm #6035 (EVO-X2 at 2787 MHz/119 W under training) · Notebookcheck EVO-X2 review (P-modes) ·
ROCm/aotriton #168 + `flash_disabled()` (hdim>256 disabled on gfx11) · pytorch #189849 / PR #194280
(hdim-512 silent garbage + cap) · ROCm #6034 (19x SDPA with the experimental flag; env pitfalls) ·
TheRock discussion #2845 (`HIP_FORCE_DEV_KERNARG`) · unsloth AMD docs + PR #4109 (Gemma NaN under
compile on RDNA) · AMD unsloth playbook (gemma-4-E4B-it on gfx1151) · kyuz0/amd-strix-halo-llm-
finetuning (8-bit 12x / QLoRA 1.8x slower) · h34v3nzc0dex strix-halo-llm-finetune-guide (Triton
RDNA caveats, env) · Dao-AILab PR #2675 / ROCm/flash-attention PR #184 (RDNA backward, varlen) ·
pytorch PR #177840 (Flex RDNA3 tiles profiled on gfx1151) · llm-tracker Strix Halo (hipBLASLt
36.9 TFLOPS) · AMD-Skills torch-compile notes (cudagraphs broken on ROCm) · triton #11378 (gfx1151
miscompile, fixed 3.8).
