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
| 2 | **Performance power mode** (P-mode button: PL1 85 W → 120 W; GPU 2175 → ~2787 MHz) | up to ~1.28x on GEMM/attention share; ~1.1–1.2x on the step — **but see hazards: at Balanced the APU already sits at Tctl 98–99 °C after 30 min of sustained training (firmware-throttled, stable); Performance mode needs better cooling first** | strong external evidence (same box model) | **owner** |
| 3 | **torch.compile** (`--compile`, inductor default mode) | **1.42x** on a real step | measured | done in trainer |
| 4 | **Length-adaptive checkpointing** (`--ckpt-above N`: recompute only the long tail) | 1.7x on the short half (no-ckpt speed) without the long-bucket OOM/thrash that plain `--no-grad-checkpointing` has | measured (sweeps 1–2) | done in trainer |
| 5 | Launch-overhead env knobs (`HIP_FORCE_DEV_KERNARG=1`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, `TORCH_BLAS_PREFER_HIPBLASLT=1`) | **1.13x** (and `expandable_segments` cures the long-bucket swap storm) | measured (sweep 2 D) | in recipe |
| 6 | Bigger batch **with** grouping | **none** — batch 12 = 992 vs 983 ms/example; GPU saturated at batch 6 | measured (sweep 2 E) | stay at 6 |

Measured composed (1+3+4+5), production run: **~48 h → 15.8 h per 49k epoch at Balanced (3.0x); ~13 h
with Performance P-mode** (lever 2, owner). See *Measured composed results* for the sweep numbers and the epoch arithmetic.

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
export HIP_FORCE_DEV_KERNARG=1 PYTORCH_ALLOC_CONF=expandable_segments:True TORCH_BLAS_PREFER_HIPBLASLT=1  # sweep 2: 1.13x
~/fttorch/bin/python ~/scripts/avtext/train_lora_peft.py \
  --model unsloth/gemma-4-E4B-it --data ~/fttrain/train_all_lg.jsonl --out ~/fttrain/adapter-all-lg \
  --rank 16 --epochs 1 --batch 6 --grad-accum 2 --max-seq 2048 \
  --group-by-length --compile --ckpt-above 1800 --log-mem   # adaptive checkpointing: only the 1800–2048 head recomputes
# Production (overnight) form — same args through the retry loop, checkpointing every 200 steps so
# a crash costs minutes: it waits for GPU memory to drain, then re-launches with --resume.
~/scripts/avtext/run_resilient.sh <same args as above> --save-steps 200
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
- **"GPU page faults" here are memory blow-ups in disguise — the APU has no clean OOM.** On a
  discrete GPU an over-allocation fails fast; on Strix Halo the "GPU" pool *is* host RAM (amdgpu GTT),
  so a process that outgrows it drags the whole box into a swap storm (signature: `mem_info_gtt_used`
  → 119 of 124 GiB, host free ~1 GiB, GPU 1 % busy, the process at 100 % CPU with a tens-of-MB RSS
  because it has been swapped out) and finally an amdgpu `Memory access fault … Page not present`.
  The S24 fault reproduced *deterministically* at a ~1,130-token batch — and the root cause was ours:
  the probe scripts never called `model.train()`. `from_pretrained` returns eval mode and HF applies
  gradient checkpointing only when `self.training`, so every probe was silently a no-checkpointing
  run holding **85 GiB live in the forward** at 1,130 tokens × batch 6; the same rows with `.train()`
  peak at 30 GiB allocated / 52 reserved. The Trainer calls `.train()`, which is why sweep 1 crossed
  the same lengths cleanly. Guards that stay: (1) `--mem-fraction` (default 0.85) caps the pool so a
  future over-allocation is a fast, retryable OOM instead of a hung box; (2) **wait for GTT to drain
  between GPU processes** — a just-exited (or faulted) process still holds 80–95 GiB for tens of
  seconds, so an immediate relaunch OOMs on its first big allocation (`gttwait` in
  `run_resilient.sh` and the sweep scripts); (3) `--save-steps 200` + `run_resilient.sh` remain cheap
  insurance against power/driver events; (4) any standalone probe: `model.train()`, and a sysfs GTT
  watchdog thread that `os._exit`s above ~90 GiB (`faultmap2.py`) beats a swap storm.
- **Thermal ceiling at Balanced already.** Sustained training (30+ min, GPU 99 % busy, 85 W) settles at
  Tctl 98–99 °C / GPU edge 93–98 °C with clocks modulating 2.07–2.34 GHz — the firmware holding its
  ~99 °C target, stable, no kernel thermal events, no fan sensor exposed to Linux. Short sweeps read
  83 °C because they never heat-soaked. Consequence: **do not press Performance mode (120 W) without
  improving cooling** — the reboots reported above 90 °C on this model were in that mode. Two
  hours in, Tctl overshot to 102 °C and the pace slid from 347 to 412 s per megabatch, so the production
  run carries a userspace governor (`~/fttrain/thermostat.sh`: SIGSTOP the trainer at ≥ 101 °C, SIGCONT
  at ≤ 97 °C — the GPU idles and the chip drops 4 °C in ~6 s; nothing else on the box is touched). The
  30-min pulse logs temperature, clock and pause count. A run that must be cooler goes to Quiet mode
  (54 W, slower), which is the owner's button, not a software setting.
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

### Sweep 2 — the decisions (same data; ms/example over quantile-matched windows)

`steady2.py` keys on the training bar's total and reports ms/example over a window that is the same
*quantile* of a megabatch for every batch size (batch-6 steps 15→25 ≡ batch-12 steps 30→50 = the
shortest 40 % of a random 300/600-example megabatch); raw s/step is not comparable across batch sizes.

| Config | window | vs reference |
|---|---|---|
| G eager, no-ckpt (is compile earning its warmup?) | **OOM at step 0** — the 2048-token bucket does not fit at batch 6 without compile's fusion | — |
| D C + env knobs (`HIP_FORCE_DEV_KERNARG=1`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, `TORCH_BLAS_PREFER_HIPBLASLT=1`) | **6.2 s/step = 517 ms/ex** | **1.13x vs C** (583); long-bucket steps healthy — `expandable_segments` removed the swap storm C had there |
| E batch 12 × accum 1, ckpt + compile | 11.9 s/step = 992 ms/ex | 1.0x vs B (983): **bigger batch buys nothing** — the GPU is saturated at batch 6 once batches are grouped |
| H D + `--ckpt-above 1100` (length-adaptive checkpointing, `--log-mem`) | **6.4 s/step = 533 ms/ex** | 1.0x vs D on the short half (no-ckpt speed); the 2048 head runs checkpointed at **23.7 GiB**; compiled no-ckpt peaks at **51 GiB at 1,059 tokens** (eager: >107 GiB) → production threshold **1,800** (est. ~76 GiB, 97 % of rows no-ckpt) |

Readings: (1) **No-checkpointing does not fit the long tail.** Re-reading sweep 1's per-step times:
C's steps 2–6 took 110–200 s (A: 48–65 s) — it "survived" the 2048 bucket by thrashing the unified
pool; eager G OOMs outright (an eager no-ckpt forward needs >107 GiB already at 1,130 tokens — compile's
fusion is what makes no-ckpt fit at all); D fits by a few GiB thanks to `expandable_segments`. Too
fragile for an unattended night. (2) Since HF checkpointing is a runtime flag checked per forward, the
trainer now flips it per microbatch on padded length (`--ckpt-above N`): the short/medium majority runs
at no-ckpt speed, only the TAF tail recomputes; `--log-mem` prints the per-microbatch peak so the
threshold is set from the compiled curve, not a probe. (3) The env knobs are a real 13 % and cost
nothing. (4) Batch stays at 6.

### Epoch arithmetic (what the night actually costs)

The 25-step sweeps are one megabatch = a random 300 examples, so *total time minus the compile step*
over 288 examples is an unbiased (slightly optimistic — it drops the 12 longest) per-example cost:

| Config | steps 2–25 | s/example | 49,214-example epoch |
|---|---|---|---|
| A grouping, eager, ckpt | 529 s | 1.84 | ~25 h (the old `perlen.py` 23.7 h agrees — it was, by accident, the eager no-ckpt curve, see hazards) |
| D grouping + compile + knobs, no-ckpt | 361 s | 1.25 | **~17 h** at Balanced; ~14 h at Performance P-mode |
| H D with adaptive ckpt (N=1100) | 454 s | 1.58 | ~22 h on this arithmetic — inflated by the second graph variant's recompiles inside the one megabatch; at N=1800 expect ≈ D. The production run's own rate is the number that counts (first megabatches, below) |

**Production run (config H, `--ckpt-above 1800`, seed 42 → same first megabatch as the sweeps):**
megabatch 1 incl. compile 827 s · megabatch 2 (recompiles for the second graph variant) 432 s ·
**megabatch 3 = 347 s for 300 examples = 1.16 s/example → 15.8 h per 49,214-example epoch** at
Balanced — steady state, the number to plan with (~13 h with Performance P-mode). No-ckpt peaks on the
fitted line (66–68 GiB at 1,520–1,566 tokens); the 1,854–2,048 head checkpoints at 22 GiB.

Random-batch baseline was ~48 h → **3.0x measured end to end**. The remaining structural lever is
bucketed static-shape compile (1.42x measured on a static step; frontier section).
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
