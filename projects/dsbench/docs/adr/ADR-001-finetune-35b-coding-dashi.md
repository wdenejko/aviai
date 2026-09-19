# ADR-001: Fine-tuning a 35B-class coding model on dashi (Strix Halo) - feasibility and plan

- **Status:** Proposed; **Gate 0 base comparison done (2026-09-19): base = `Qwen3.6-35B-A3B`** (dsbench v2 head-to-head vs Ornith-1.5 — see Gate 0 action items)
- **Date:** 2026-09-15
- **Deciders:** Wojtek Denejko (box owner)
- **Relates to:** ADR-002 (the dsbench evaluation harness that measures this plan's gates); aviai ADR-005 (compute inventory) and ADR-008 (ROCm training stack on dashi); memory notes `engram-graft-research`, `qwen3.6-27b-strix-halo-measured`, `flashnext-build-recipe`
- **Note:** aviai working copy for the research continuation; a verbatim copy also lives in the llama.cpp fork at `docs/adr/ADR-0001-...`. The benchmark that scores the gates below is specified in ADR-002 and already implemented at `projects/dsbench/`.
- **Research trail:** five web-research passes (model family, AMD training stack, no-regression methods, data sources, case studies) plus read-only inspection of dashi and this repo, 2026-09-14/15. Sources are listed in Appendix B; first-hand measurements in Appendix A.

## Question

> Using the dashi server, would it be possible to fine-tune an LLM in the 35-70B parameter range to improve its quality for coding (data science and data engineering in particular, software engineering in general) without losing the performance?

"Performance" is read in both senses the owner cares about on this box: (1) inference speed on the perf fork (decode and prefill tokens/s, MTP draft acceptance) and (2) general model quality (no regression on non-coding and general-coding tasks).

## Context

### What is being asked for

- Inspiration: `Qwen/Qwen3.5-35B-A3B` and its spin-offs `ornith-ai/Ornith-1.5-35B-A3B` and `huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated`.
- Target skills: data science (pandas/polars/numpy/sklearn/statistics/Jupyter), data engineering (SQL and text-to-SQL, dbt, Spark/PySpark, Airflow/Dagster, warehouses, streaming) and general software engineering (repo-level fixes, tool use, tests, review).
- Hard constraints: one machine (dashi, 128 GB Strix Halo, no discrete GPU); the box also runs the production `llama-server`; the result must serve on this fork at today's speed.

### The model family (Appendix B.1)

- Qwen3.5-35B-A3B was released 2026-02-24 and Qwen3.6-35B-A3B on 2026-04-16; both Apache-2.0, 35B total / 3B active, 262k context, with a vision tower and one MTP layer. The text architecture is identical (40 layers: 30 Gated DeltaNet + 10 gated attention, 256 experts top-8 + shared, hidden 2048, head_dim 256, vocab 248,320).
- There is no Qwen3.7 generation and no 35B-A3B in Qwen3.8 (only 27B dense, 2.4T-A95B and Flash-Next); there is no official Qwen3.5/3.6 "Coder". Qwen3.6-35B-A3B is itself positioned as the agentic-coding model (SWE-bench Verified 73.4 vs 69.2 for 3.5).
- Qwen discloses no post-training recipe; official fine-tuning guidance is "use Unsloth, ms-swift, LLaMA-Factory" with stale Qwen2.5-era hyperparameters and no MoE-specific advice.
- The derivative ecosystem is large (HF lists 259 fine-tunes of Qwen3.6-35B-A3B and 165 of Qwen3.5-35B-A3B). The coding gains that are actually measured come from RL with sandboxed execution at multi-node scale: Ornith-1.5 (self-improvement RL loop, MIT, SWE-bench Verified 79 self-reported, 4.15M GGUF downloads/month, no data/hardware disclosed) and Lego-X (GSPO RL inside the OpenHands SDK on 3 nodes x 8 GPUs: SWE-bench Verified 64.0 -> 70.4). Single-GPU community LoRA distills exist (rank 16-32, attention-only, 4k-10k Claude-trace samples, 1x H200, 2 epochs) but report only tiny or noisy evals. Abliteration (Huihui, Bahushruth, HauhauCS) is weight surgery, not training, and nobody publishes its capability cost. The closest data-engineering tune is `datajuicer/Juicer-35B-A3B` (SFT + RL over Data-Juicer operators). No derivative publishes training tokens/s, and no derivative documents keeping the MTP head.
- Ornith-1.5 is already on dashi (`~/models/Ornith-1.5-35B-A3B-GGUF`, BF16 71 GB + Q8_0 38 GB); its GGUF header says `qwen35moe`, 256 experts, `block_count` 41 (the MTP layer is present).

### What dashi can do today (Appendix A)

- A working LoRA training stack already exists on the box (AMD TheRock torch 2.12+rocm7.13 for gfx1151, transformers 5.16.1, peft 0.20.0, trl 1.12.0, Triton 3.7 via a zig-cc shim) and completed a real campaign on 7-11 Sep 2026: a rank-16 bf16 LoRA of Gemma-4-E4B over 32.1M tokens in 16 h 58 min (525 tokens/s end to end, 28.8 TFLOP/s bf16 matmul ceiling, no NaN, one attempt). The same campaign documented the box's failure modes: no clean OOM (GTT swap storm then GPU page fault), thermal soak to Tctl 99-102 C at 85 W with a userspace SIGSTOP governor, and a hard reset after ~45 h of continuous GPU load.
- The repo serves the target architecture (`qwen35moe`, MTP included), converts its checkpoints (fused 3-D experts and legacy layouts, `mtp.*` -> nextn), and applies LoRA adapters at runtime to dense projections and to the routed experts (`build_lora_mm_id`). The in-tree `llama-finetune` is FP32-only and not usable for a 35B MoE.

### The platform as of September 2026 (Appendix B.2)

- ROCm 10.0.0 (2026-08-26) lists gfx1151 as officially supported and AMD now publishes gfx1151 torch wheels (torch 2.11-2.13 on ROCm 7.14/10.0); pytorch.org still ships none. The `torch._grouped_mm` segfault on gfx1151 is fixed from torch 2.11.0+rocm7.13.0. AOTriton has no gfx1151 tuning database through 0.13.x; SDPA FLASH/EFFICIENT with backward works at head_dim <= 256 (the Qwen3.5 attention shape) with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; CK flash-attention backward is unsupported on RDNA. Unsloth lists gfx1151 as fully supported since July 2026, but Strix Halo users still pin the 12x-slower `native_torch` MoE backend to avoid segfaults, and Unsloth's fast Triton MoE path is undocumented on AMD.
- flash-linear-attention (needed for the 30 Gated DeltaNet layers) runs on ROCm: `chunk_gated_delta_rule` gradients agree with the reference to 1.8e-6 on MI355X, but the causal-conv1d backward returned silently wrong gradients on ROCm in fla 0.4.2 and 0.5.2 (issue #1156), fla's AMD warp guard is not wave32-aware (#1163), and fla 0.5.0 broke Qwen3.5 outright (#792). Without fla, transformers falls back to a pure-torch GDN path that is correct but slow.
- No public tokens/s figure exists for any >= 20B MoE LoRA on Strix Halo except one: the `woct0rdho/transformers5-qwen3.5-recipe` project (Apache-2.0, actively pushed 2026-09-13) trains Qwen3.6-35B-A3B rank-4 LoRA on a GGUF base with gfx1151-tuned native MMQ, grouped-MMQ, AITER GMM, FLA and fused-norm kernels and reports a warmed traced step of **5.33 s per batch-1 x 2048-token update (forward 1.43 s, backward 3.83 s of which 32% is checkpoint recompute), i.e. ~384 tokens/s, at 15.2 GiB peak allocation** with the whole model resident on the GPU. That is inside the 150-500 tokens/s band derived independently from the box's Gemma run (Appendix A6).
- Cloud reference (2026-09-15): H100 80GB PCIe $1.99-2.89/h, H100 SXM $2.69-3.49/h ($1.07 on Vast), RTX PRO 6000 96GB $1.69-2.09/h, B200 $5.98-6.99/h. Unsloth's own bf16 LoRA memory figure for Qwen3.5-35B-A3B is 74 GB, so an 80-96 GB single card suffices.

### What "without losing performance" means in practice

1. **Inference speed** is preserved by construction if the fine-tuned model is merged and re-quantized to the same tensor types as today's GGUF: the graph, the quant types and every fork optimization for `qwen35moe` are unchanged. The one moving part is the MTP draft head: transformers drops `mtp.*` on load, so the trained trunk drifts under a frozen head and draft acceptance can fall; it must be re-measured and, if needed, the head refreshed against the new trunk. A runtime LoRA adapter (`--lora`) keeps the base and MTP head byte-identical but adds two small matmuls per adapted projection per token; its decode cost on the Vulkan path is unmeasured and must be measured before it is used for anything but evaluation.
2. **General quality** cannot be guaranteed a priori; it is controlled by method (LoRA at modest rank on all layers, assistant-only loss, 25-30% general replay, conservative LR, one epoch) and verified by a before/after eval battery with explicit regression thresholds. The concrete harness is the dsbench project (ADR-002); the external benchmarks in Appendix B.3 complement it. The literature is consistent that LoRA forgets less than full fine-tuning at equal target gain and that domain gains from 10-50M SFT tokens are real, while general-coding jumps at this model size came from RL that this box cannot run.

## Decision (proposed)

**Yes, with a narrow shape.** Fine-tuning on dashi is feasible for parameter-efficient (LoRA) supervised fine-tuning of the 35B-A3B MoE family, at 10-30M training tokens per run, with the model served afterwards on the fork at unchanged speed. It is not feasible for full fine-tuning, for RL post-training at any useful scale, or for dense models above ~27B. Concretely:

1. **Base model:** `Qwen/Qwen3.6-35B-A3B` (Apache-2.0, the newest 35B-A3B, best SWE-bench of the family, MTP head, already supported by the fork). `Ornith-1.5-35B-A3B` (MIT, RL-tuned for agentic coding, already on disk) is evaluated as an alternative starting point at Gate 0; it wins only if its data-science/data-engineering scores are already higher and its agentic behaviour survives an SFT pilot. Dense 27-70B bases are rejected (section "Options").
2. **Target:** data-science and data-engineering competence (pandas/polars/SQL/dbt/Spark/Airflow tasks, notebook and tool-call formats, the owner's stack) plus format and tool-use robustness for agent harnesses. General SWE-bench-style gains are explicitly out of scope for the box: they come from RL with sandboxed execution at multi-node scale (Ornith, Lego-X). If that is wanted, start from Ornith or rent GPUs.
3. **Training path on dashi:** GGUF-base LoRA with gfx1151-native kernels (the `woct0rdho/transformers5-qwen3.5-recipe` + `torch-ggml-ops` stack) as the primary path because it is the only path with a measured 35B-A3B step on this silicon (5.33 s / 2048 tokens, 15 GiB). The existing bf16 stack (`~/fttorch`, PEFT `target_parameters` on the fused experts, `experts_implementation="grouped_mm"`, fla + causal-conv1d) and `unsloth[amd]` are A/B'd at Gate 0 as fallbacks. Cloud training (one H100/RTX PRO 6000 for a few hours, ~$5-15 per run) is the escape hatch if Gate 0 measures below ~100 tokens/s or cannot be made numerically sound.
4. **Method:** LoRA rank 16 (rank 8 on experts if memory-bound), alpha = 2x rank, all linear projections including routed experts and the shared expert, router frozen, router aux loss off, LR 1e-4 to 2e-4 cosine with 3% warmup, one epoch, assistant-only loss, sequence length 2048-4096, batch 1 x grad-accum 16-32 (~32-64k tokens per optimizer step), 25-30% general replay data, checkpoints every 200 steps with resume, the thermal governor and GTT drain gate from the Gemma campaign.
5. **Serving:** for evaluation, base GGUF + `--lora adapter.gguf` on the Vulkan build (exact training-time configuration, MTP head untouched). For production, merge into bf16, re-attach `mtp.*` from the base checkpoint, convert with this fork's converter, and re-quantize with imatrix to the same recipe as today's serving GGUF, then re-measure decode, prefill and MTP acceptance. Runtime LoRA becomes a production option only if its measured decode cost is under ~10%.
6. **Gates:** no long run before the stack smoke test (Gate 0), the end-to-end pipeline pilot with a 1M-token adapter (Gate 1) and a full before/after evaluation on a 10M-token run (Gate 2) pass their kill criteria (section "Action items").

## Options considered

### Option A - bf16 LoRA on dashi with the existing TheRock + PEFT/TRL stack

| Dimension | Assessment |
|---|---|
| Complexity | Medium: venv exists; add fla + causal-conv1d, PEFT `target_parameters`, chunked loss; validate GDN gradients on RDNA |
| Memory | 69 GB base + activations + adapter/optimizer: ~86 GB (dense-only LoRA) to ~110 GB (rank-16 expert LoRA) against a 105 GB usable cap; server must be stopped |
| Speed | Unmeasured for MoE; planning band 150-500 tok/s; the naive expert loop (256 experts, `index_add`) and the torch GDN fallback could push it far below |
| Risk | fla conv backward correctness on ROCm (#1156), Triton fork/worker bug, thermal, swap storm; expert-LoRA rank limited by memory |
| Team familiarity | High: same scripts, launcher, governor as the Gemma campaign |

**Pros:** provenance-clean (official bf16 weights), merge-to-bf16 is exact, everything already installed. **Cons:** no MoE measurement yet; the largest memory footprint of any option; bf16 expert weights (64 GB touched per step) sit on a 256 GB/s bus.

### Option B - GGUF-base LoRA on dashi (woct0rdho recipe + torch-ggml-ops) - **chosen**

| Dimension | Assessment |
|---|---|
| Complexity | Medium-high: forked transformers (`gguf` branch, 2026-09-12), native ROCm extension built from this repo's ggml sources (hipcc is in the TheRock venv), AITER/FLA/Liger pins; research-grade code, single author, 15 stars |
| Memory | 15.2 GiB peak at rank 4 (measured by the author); rank 16 adds ~1.5 GB; leaves >100 GB free (a served model could even coexist memory-wise, though not compute-wise) |
| Speed | 5.33 s per 2048-token step measured on gfx1151 (~384 tok/s incl. profiler overhead); FLA backward is the top cost (1.3 s), then grouped MMQ and GEMM |
| Risk | dependency churn, router aux loss disabled by design, rank-4/bs-1/seq-2048 is the validated envelope, quality of training against a 4-bit base (QLoRA-equivalent) |
| Team familiarity | Low-medium: new stack, but the trainer is a plain transformers `Trainer` subclass and adapters are ordinary PEFT files |

**Pros:** the only measured 35B-A3B step on this chip; memory-safe by a wide margin (no swap storms); adapters apply at runtime to the same GGUF in this fork, so evaluation is exact; supports Q3_K..Q6_K/Q8_0 bases, so the base can be our own imatrix quant of the official weights. **Cons:** QLoRA-style: merge-then-requant introduces a small mismatch that must be checked by greedy parity; higher ranks and longer sequences are untested by the author; upstream-ability of the transformers GGUF quantizer is open (transformers issue #40070).

### Option C - unsloth[amd] on dashi

| Dimension | Assessment |
|---|---|
| Complexity | Low to install, unknown to make fast: `grouped_mm` backend needs torch >= 2.11+rocm7.13; field reports pin `native_torch` (12x slower); Triton MoE backend undocumented on AMD |
| Memory | 74 GB bf16 claim for 35B-A3B (NVIDIA); Unsloth caps gfx1151 at 80% of the pool |
| Speed | Unmeasured on gfx1151 for MoE; on NVIDIA 1.4x over transformers for Qwen3-30B-A3B |
| Risk | segfaults reported on Strix Halo MoE paths; compile disabled on RDNA for some models; a second venv (the old `~/ft` was wiped) |
| Team familiarity | Medium: unsloth ran on this box in July-August 2026 |

**Pros:** LoRA on fused experts out of the box, corrected chat template, `save_pretrained_merged`. **Cons:** nothing measured on AMD for MoE; behaviour on this box unknown. Kept as a one-afternoon A/B at Gate 0.

### Option D - rent one cloud GPU for the training step, keep dashi for data, eval and serving

| Dimension | Assessment |
|---|---|
| Complexity | Low: mature CUDA kernels (Unsloth/LLaMA-Factory/ms-swift), same adapter format comes back |
| Cost | H100 80GB $2-3.5/h or RTX PRO 6000 96GB $1.7-2.1/h; a 30M-token LoRA run is a few hours: ~$5-15 per run, ~$50-150 for a full iteration campaign |
| Speed | 10-20x the box (ratio-based; absolute H100 tok/s for 30B-A3B LoRA is unpublished) |
| Risk | data leaves the box; account/credit setup; still needs the local conversion/serving pipeline |
| Team familiarity | Medium |

**Pros:** iteration speed; frees dashi to keep serving. **Cons:** violates the "using the dashi server" premise; cost is small but not zero. Retained as the escape hatch and as the honest comparison.

### Option E - no-training alternatives

- Adopt an existing tune after local evals: Ornith-1.5 (already on disk) is the strongest public 35B-A3B for agentic coding; `Juicer-35B-A3B` targets data pipelines. Cost: an evaluation day. Recommended as the Gate 0 baseline regardless.
- Prompting, skills and retrieval over dbt/Spark/Airflow/pandas docs for the owner's stack; zero training, immediate, orthogonal to fine-tuning (and the same eval battery measures it).
- Model merging (mergekit TIES/DARE) of community tunes: cheap but unmeasured on this architecture; useful later as a regression-recovery tool (merge the adapter back towards the base at a lower scale).
- Abliteration: does not improve coding; excluded from this decision.

### Option F - full fine-tuning or RL on dashi - rejected

Full fine-tuning needs ~16 bytes per parameter with AdamW (560 GB for 35B; kyuz0's Strix Halo matrix tops out at 12B full FT in 115 GB). RL (GRPO/GSPO) needs rollout servers, sandboxed test execution and many GPU-hours per step (Lego-X: 24 GPUs, 126 steps); the box would take weeks per experiment.

### Option G - dense 27-70B bases - rejected for the box

Qwen3.6-27B bf16 LoRA fits (54 GB weights) but every token costs ~9x the MoE's active compute (27B vs 3B) -> ~40-60 tok/s -> 10M tokens in 2-3 days, above the box's 45 h stability record per run; Llama-3.3-70B bf16 (140 GB) does not fit and its 4-bit path is 1.8x slower still. The MoE is the only 35-70B-class model that trains at a useful rate on 256 GB/s of unified memory.

## Trade-off analysis

- **Box vs cloud.** The box trains a 10M-token LoRA overnight and a 30M-token one over a weekend, for free, with data never leaving the house; the cloud does the same in 1-3 hours for $5-15. What the box cannot buy is iteration count: a full eval cycle (train + convert + eval battery) is ~1-2 days on the box vs ~half a day rented. The decision keeps the box as the primary path because the question is about the box, and because most of the calendar time in this project goes to data construction and evaluation, which run on the box either way.
- **GGUF-base (B) vs bf16 (A).** B is measured, memory-safe and evaluation-exact; A is provenance-clean and merge-exact but unmeasured and memory-tight. The mismatch B introduces at merge time is bounded by the quantization noise of a low-rank delta and is checked by greedy parity at Gate 1; if it fails, A is the fallback for the final production run only.
- **Rank and targets.** Expert LoRA is where the capacity is (92% of parameters); attention-only distills in the community show no convincing gains. Rank 16 on experts costs ~1.3B adapter parameters; on B this is a few GB, on A it is ~20 GB with fp32 optimizer state. The plan starts at rank 16 all-layers on B and drops to rank 8 on experts if memory or the A/B says so.
- **Quality vs speed of the served model.** Merge + requant keeps today's speed exactly; runtime LoRA keeps today's MTP head exactly. Which one ships is decided by two measurements at Gate 1 (decode cost of the adapter; MTP acceptance after merge), not by preference.
- **Scope honesty.** The achievable improvement on the box is domain-specific (the owner's DS/DE stack, formats, tool use) and measured against a battery the owner controls; it is not a general leap in coding ability, which at this size has been bought with RL compute the box does not have.

## Consequences

- **Easier:** a reusable LoRA pipeline for the `qwen35moe` family on the box (train -> adapter -> GGUF adapter -> Vulkan serving), per-domain adapters that can be swapped without re-quantizing the base, and an evaluation battery that also measures no-training alternatives (prompts, retrieval, other community tunes).
- **Harder:** the production `llama-server` must be stopped for every training window (the GPU is shared, and the box's power budget is the throttle); every long run needs the governor, checkpointing and a resumable launcher; the box's ~45 h continuous-load record caps a single run at ~30M tokens without a resume.
- **New maintenance surface:** a second Python environment pinned to a research-grade stack (forked transformers, native ops built from this repo's ggml sources, fla/causal-conv1d/AITER/Liger pins). Pin commits, keep the bf16 venv as the fallback, and re-run the numerical checks after any upgrade.
- **What must be revisited:** the MTP head (refresh it against the new trunk if acceptance drops), the quant recipe and imatrix for the merged model, and the eval thresholds once the first before/after numbers exist. If Gate 0 measures below ~100 tokens/s or the GDN gradients cannot be validated on RDNA, training moves to Option D and the box keeps data generation, evaluation and serving.
- **Data licensing:** only teacher-generated data whose licences permit distillation (Qwen, DeepSeek, Kimi, GLM, gpt-oss derived) goes into anything that might be published; Claude/OpenAI/Gemini-derived sets stay study-only (Appendix B.4).

## Action items (gated plan)

Each gate has a kill criterion; nothing below a gate starts before the gate passes. GPU windows require stopping the production server (owner action), the GTT drain gate (`~/fttrain/gttwait.sh`) between GPU processes, and the thermal governor (`~/fttrain/thermostat.sh`) for anything longer than an hour.

### Gate 0 - stack smoke test and throughput (1-2 GPU sessions, ~4-6 h)

1. [x] Baseline evals of `Qwen3.6-35B-A3B` and `Ornith-1.5` on the dsbench v2 agentic battery (ADR-003), pi harness @ 0.84.4, thinking high, temp 0, k=5, **Q8_0 GGUFs both** (apples-to-apples). **Result: a statistical tie** — Qwen3.6 **79/115** runs · 15/23 by majority; Ornith **75/115** · 16/23. On the *target* DS/DE domains Qwen3.6 leads **38/50 vs 34/50** (mainly `ds_delay_predict` 4/5 vs 1/5 and `de_taf_latest` 5/5 vs 3/5). Both share the same intrinsic gaps: ClickHouse `dayOfWeek`/timezone conventions, the `da_delay_attribution` denominator error (Qwen 40.9 / Ornith 42.8 vs 44.1 truth — two different wrong answers confirm a genuine family-wide reasoning gap, not an oracle artifact), and ML-output-delivery reliability. **Decision: fine-tune `Qwen3.6-35B-A3B`.** Per the base-selection rule Ornith wins only if its DS/DE scores are *higher* (they are not), so the official Apache-2.0 base is chosen — cleaner provenance, intact MTP head, and a cleaner SFT substrate than an opaque RL tune. Reports: `projects/dsbench/reports/agentic-runs/20260919-005921-ornith-23problem-k5.json` and `20260919-110559-qwen36-23problem-k5.json`. (Decode/prefill/MTP-acceptance speed metrics still to record once the training-base GGUF is fixed.)
2. [ ] New venv `~/ftgguf` (Python 3.12, TheRock gfx1151 torch as in `~/fttorch`): build `torch-ggml-ops` (`tools/generate_vendor.py --llama-cpp=~/src/llama-qwen4exp-src`, `pip install --no-build-isolation --no-deps -e .`), install `woct0rdho/transformers@gguf`, `peft`, `trl`, `flash-linear-attention` (pin the commit the recipe validates), `causal-conv1d`, `aiter`, `liger-kernel`, `bitsandbytes` (for `adamw_8bit` only). Keep `CC=~/fttrain/bin/zigcc`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, `HIP_FORCE_DEV_KERNARG=1`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, `num_workers=0`, `torch.cuda.set_per_process_memory_fraction(0.8)`.
3. [ ] Base GGUF for training: prefer our own Q4_K_M or Q5_K imatrix quant of the official bf16 weights made with this fork (provenance), else `mudler/Qwen3.6-35B-A3B-APEX-I-Mini.gguf` (the author's validated file).
4. [ ] Run the recipe's `audit_qwen3_5_training_step.py` at rank 4 (reproduce 5.33 s / 15 GiB), then rank 16 and sequence 4096; record s/step, tokens/s, peak GTT.
5. [ ] Numerical checks: GDN causal-conv backward vs the pure-torch fallback on one layer (fla #1156 class of bug), loss on 20 steps of real data decreasing, no NaN, compiled-vs-eager parity where compile is used.
6. [ ] A/B (one afternoon each, optional if step 4 is satisfactory): bf16 path in `~/fttorch` (PEFT `target_parameters` on `experts.gate_up_proj`/`down_proj`, `experts_implementation="grouped_mm"`, fla installed) and `unsloth[amd]` in a third venv with `UNSLOTH_MOE_BACKEND` = `grouped_mm` then `unsloth_triton`.
7. **Kill:** best path < 100 tokens/s at rank 16, or gradients not validated -> switch training to Option D; the rest of the plan is unchanged.

### Gate 1 - end-to-end pipeline on a 1M-token pilot (1-2 days)

1. [ ] Pilot mixture: ~1M tokens sampled from the Gate 2 mixture (Appendix B.4), rendered with the corrected Qwen3.5 chat template (tool-call `arguments` mapping fix), assistant-only labels, thinking disabled for non-reasoning rows.
2. [ ] Train ~500 steps; save the adapter; convert with `convert_lora_to_gguf.py --base <bf16 snapshot>` from this fork. Watch the GDN `in_proj_qkv`/`out_proj` V-reorder replay on the LoRA tensors; if it fails, exclude those two projections from LoRA (the GDN `in_proj_z`, attention and expert projections remain) or use the merge route.
3. [ ] Serve base GGUF + adapter on the Vulkan build; greedy parity vs the HF model on 10 fixed prompts; measure decode/prefill and MTP acceptance with and without the adapter.
4. [ ] Merge route dry run: merge into bf16 (`merge_and_unload`), copy `mtp.*` tensors from the base safetensors into the saved checkpoint, convert with `conversion/qwen.py` (MTP as a separate `mtp_only` GGUF), quantize with the production recipe and imatrix, and check greedy parity vs the runtime-adapter server on the same prompts.
5. [ ] Run the short eval battery (HumanEval+ and DS-1000 subsets, IFEval subset) on base vs adapter to validate the harness, not the model.
6. **Kill:** adapter cannot be applied or merged losslessly (parity failures that survive debugging), or runtime adapter costs > 15% decode with no merge path -> stop and re-plan.

### Gate 2 - first real run, 10M tokens (8-20 h GPU, ~1 day eval)

1. [ ] Mixture per Appendix B.4 (data science + data engineering + SWE + 25-30% general replay), decontaminated against every eval set (13-gram overlap) before training.
2. [ ] Hyperparameters from the Decision; checkpoint every 200 steps; governor on; resumable launcher (`prod_all_lg.sh` pattern).
3. [ ] Full battery before/after; acceptance: target-domain gains on at least two of DS-1000, BIRD-dev, the owner's DE suite, and the tool-call format tests; regression <= 1 point on IFEval/MMLU-Pro/GPQA subsets and <= 2 points on LiveCodeBench/HumanEval+; MTP acceptance drop <= 5 points at the served quant.
4. [ ] If a regression exceeds the threshold: halve the adapter scale at load (`--lora-scaled`) and re-evaluate, then raise replay to 40% and retrain; if it persists, the data slice responsible is identified by ablation before any scale-up.

### Gate 3 - scale-up and production (following weeks)

1. [ ] 30M tokens or a second epoch with fresh samples, optionally a small DPO/RFT pass on execution-verified own samples (the teacher can be Ornith/Flash-Next/DeepSeek-V4-Flash served on the box at ~35-100 tok/s: 10k verified samples of ~1k tokens is ~30-80 h of box inference, or a few dollars of DeepSeek/Qwen API).
2. [ ] Production artifact: merged + imatrix-requantized GGUF at today's recipe, MTP head refreshed if Gate 2 measured a drop, a new launcher in `scripts/run/` per the runner policy (one fixed config per file), guard for the binary.
3. [ ] Publish decision (HF release) is separate and needs the data-licence audit to pass.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Throughput far below the recipe's measurement (different GGUF types, rank 16, seq 4096) | Medium | Schedule x2-3 | Gate 0 measures before any long run; kill criterion at 100 tok/s |
| GDN kernels numerically wrong on RDNA (fla #1156/#1163 class) | Medium | Silent bad adapter | One-layer gradient check vs torch fallback at Gate 0; pin fla commit; NaN guards |
| Swap storm / GPU page fault on over-allocation | Low on Option B, medium on A | Box hang, owner reboot | Memory fraction 0.8, GTT drain gate, sysfs watchdog, `expandable_segments` |
| Thermal soak, firmware reset after ~45 h | Medium on long runs | Lost run | Governor, 200-step checkpoints, resumable launcher, runs <= 30M tokens |
| MTP head mismatch after merge (acceptance drop, or `mtp.*` lost on save) | High if unhandled | Decode speed loss | Re-attach `mtp.*` from base; measure acceptance; refresh head with frozen trunk if > 5 points drop |
| Adapter conversion fails on V-reordered GDN tensors | Medium | Runtime-LoRA route blocked | Exclude those projections or use the merge route |
| Chat template / tool-call format bugs in official template | High | Broken tool calls after SFT | Train and serve with the corrected template; add tool-call format tests to the battery |
| General regression from SFT | Medium | The point of the exercise | LoRA, replay, low LR, one epoch, eval thresholds, adapter scaling, merge-back |
| Benchmark contamination in public SFT sets | Medium | Fake gains | 13-gram decontamination against DS-1000, HumanEval+, LiveCodeBench, BIRD, Spider |
| Licence of teacher-derived data (Claude/OpenAI/Gemini) | High for some popular sets | Cannot publish | Prefer DeepSeek/Qwen/Kimi-derived and execution-verified sets; keep study-only sets out of anything released |
| Research-grade dependencies churn (forked transformers, native ops) | High | Rebuild cost | Pin commits, keep `~/fttorch` bf16 fallback venv, re-run the audit script after upgrades |
| Disk (342 GB free) | Low | Blocked download | bf16 snapshot 70 GB + quants 22-40 GB + data; move Ornith BF16 (71 GB) off the box if needed (owner decision) |

---

# Appendices

## Appendix A - First-hand evidence (measured on dashi or read from the repo, 2026-09-14)

Everything in this appendix was inspected directly (ssh dashi read-only, the local repos, the aviai project notes). Items marked *measured* come from logs on the box; nothing here is a web estimate.

### A1. The box

| Item | Value (inspected 2026-09-14 16:27 CEST) |
|---|---|
| Machine | GMKtec EVO-X2, AMD Ryzen AI MAX+ 395, Radeon 8060S = gfx1151 (RDNA 3.5, 40 CU), 32 threads |
| OS | Fedora 43 Server, kernel 7.1.3-101.fc43; no sudo for the agent account |
| Memory | 123 GiB usable; kernel `amdgpu.gttsize=126976 ttm.pages_limit=32505856` -> GTT 124.0 GiB, UMA VRAM 2 GiB. The GPU pool is host RAM |
| Disk | 1.9 TB root, 342 GB free (82% used); `~/models` 1.2 TB |
| Load at inspection | production `llama-server` (Flash-Next UD-Q4_K_XL-a1perm + MTP draft, :8080) holding 98.5 GB GTT / 94 GiB RAM; searxng; ComfyUI container (exited) |
| Containers | `llama-rocm-7.2.4_2` (training toolbox), `llama-vulkan-wdenejko` (serving), `llama-vulkan-wdenejko-build`, kyuz0 `rocm-7.14*` images |
| Stability record | hard reset 2026-09-09 18:47 after ~45 h of continuous GPU training load (no kernel trace); earlier GPU wedges from crash loops need a reboot only the owner can do |
| Already on disk | `~/models/Ornith-1.5-35B-A3B-GGUF` (Aug 19), Qwen3.6-27B-MTP, Qwen3.8-27B, Qwen3.8-Flash-Next (+MTP, +Uncensored), gpt-oss-120b, DeepSeek-V4-Flash GGUF (HF cache), Agnes-3.0-Flash safetensors |

### A2. The training stack that already exists on dashi

Venv `~/fttorch` (Python 3.12; the host 3.14 has no torch wheels), usable only inside toolbox `llama-rocm-7.2.4_2` (the host lacks `libatomic.so.1` and the GPU wiring):

| Package | Version | Note |
|---|---|---|
| torch | 2.12.0a0+rocm7.13.0a20260411 | AMD TheRock nightly for gfx1151 (`https://rocm.nightlies.amd.com/v2/gfx1151/`), native kernels, no `HSA_OVERRIDE_GFX_VERSION` |
| rocm-sdk / rocm_sdk_libraries_gfx1151 | 7.13.0a20260411 | bundled in the venv |
| triton | 3.7.0+git18f89f64.rocm7.13 | needs a C compiler: `ziglang` 0.16 + `~/fttrain/bin/zigcc` shim (no gcc in the toolbox, no sudo) |
| transformers | 5.16.1 | has `Qwen3_5MoeForCausalLM`; experts are fused 3-D `nn.Parameter` (`gate_up_proj`, `down_proj`); GDN uses `chunk_gated_delta_rule` from the kernels hub with a pure-torch fallback; `mtp.*` weights are ignored on load |
| peft | 0.20.0 | `LoraConfig.target_parameters` present (LoRA on 3-D expert params) |
| trl | 1.12.0 | SFTTrainer with assistant-only loss |
| accelerate / datasets | 1.14.0 / 5.0.1 | |
| liger_kernel | 0.8.2 | installed but ruled out (8x slower fused CE, measured) |
| NOT installed | `fla` (flash-linear-attention), `causal_conv1d`, `kernels`, `unsloth`, `bitsandbytes`, `flash_attn` | the GDN layers would run the slow torch fallback until `fla` is installed and verified on RDNA |

Trainer: `~/scripts/avtext/train_lora_peft.py` (187 lines, transformers+peft+trl; aviai repo `projects/avtext/scripts/dashi/`), production launcher `~/fttrain/prod_all_lg.sh`, GTT drain gate `~/fttrain/gttwait.sh`, thermal governor `~/fttrain/thermostat.sh`. Runbooks: aviai `projects/avtext/docs/DASHI_TRAINING_ENV.md` and `STRIX_HALO_TRAINING_SPEED.md` (the latter has ~50 dated sources). Earlier (Jul-Aug 2026) an unsloth[amd] venv `~/ft` also ran on this box (Gemma-3/Gemma-4 LoRA); it was wiped in August; `~/unsloth_compiled_cache` is its leftover.

### A3. Measured training campaign (Gemma-4-E4B-it, rank-16 bf16 LoRA, 7-11 Sep 2026)

| Metric | Measured value |
|---|---|
| Data | 49,214 aviation SFT rows (143-2048 tokens, p50 460-1131), `num_tokens` 3.209e7 |
| Run | 4,102 optimizer steps (batch 6 x accum 2), `train_runtime` 61,070 s = 16 h 58 min, 0.806 samples/s, 1 attempt, no NaN |
| Throughput | 525 tokens/s end to end (32.1M tokens / 61,070 s) for an ~8B-param (~4.5B FLOP-active) dense model |
| Raw compute | bf16 4096^3 matmul 28.8 TFLOP/s (27.9 sustained at 2175 MHz; ~37 at 2900 MHz) |
| Step profile (batch 6, ~500 tok) | 33% base GEMMs, 35% unfused elementwise/copies, 12% rank-16 LoRA GEMMs (launch-bound), 15% attention; ~15k kernel launches per microbatch, CPU time ~= GPU time |
| Peak memory | no-ckpt steps 66-68 GiB at ~1,550 tok x batch 6; ckpt steps 22-24 GiB at 2048 tok; reserved 76 GiB |
| Levers (measured) | length-grouped batching ~2x; `torch.compile` 1.42x; `HIP_FORCE_DEV_KERNARG=1 PYTORCH_ALLOC_CONF=expandable_segments:True TORCH_BLAS_PREFER_HIPBLASLT=1` 1.13x; adaptive grad-ckpt (`--ckpt-above N`) 1.7x on short rows; larger batch 0 gain (saturated at 6) |
| Attention | SDPA FLASH/EFFICIENT fwd+bwd work at head_dim 256 via AOTriton (93-99 ms vs MATH 280 ms); padding mask disables FLASH, EFFICIENT 2.2x slower; needs `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` on this wheel |
| Ruled out | Liger fused CE (8x slower), `reduce-overhead`/hipGraph (0 gain, flaky), bitsandbytes 4-bit (1.8x slower) and 8-bit (12x slower) on this chip, packing with plain SDPA (O(L^2) MATH, 310 s/step) |
| Memory failure mode | no clean OOM on the APU: over-allocation -> swap storm (gtt_used 119/124 GiB, host free ~1 GiB, GPU 1% busy) -> amdgpu "Memory access fault ... Page not present". Guards in use: `--mem-fraction 0.85`, GTT drain gate between GPU processes (a just-exited process holds 80-95 GiB for tens of seconds), `--save-steps 200` + resilient relaunch |
| Thermal | Balanced mode (85 W PL1) settles at Tctl 98-99 C, clocks 2.07-2.34 GHz; after ~2 h overshoots to 102 C and pace slides 347 -> 412 s per megabatch; governor SIGSTOPs the trainer at >= 101 C (31 pauses in the first hot hour). Performance mode (120 W) needs better cooling first |
| Triton | 3.7 runs (zig cc); known gfx1151 miscompiles in 3.6/3.7 fixed in 3.8; `num_warps > 4` can assert on RDNA; dataloader `num_workers=0` required |

Result quality of that campaign (for calibration of what a rank-16 LoRA can do): METAR EM 94.4%, TAF EM 93.3%, NOTAM classification 95.3% on full eval sets, from base scores of 24.8% / 7.2% / 78.2%; the merged model was released as `aviai-e4b`.

### A4. Target-model math (from `config.json` of Qwen/Qwen3.5-35B-A3B and Qwen/Qwen3.6-35B-A3B, fetched 2026-09-14; identical text architecture)

- hidden 2048; 40 layers = 30 `linear_attention` (Gated DeltaNet: 16 key heads x 128, 32 value heads x 128, conv kernel 4, fp32 SSM state) + 10 `full_attention` (every 4th layer; 16 q heads, 2 kv heads, head_dim 256, output gate, partial rotary 0.25, interleaved mRoPE).
- MoE: 256 experts, top-8, `moe_intermediate_size` 512, shared expert 512, `router_aux_loss_coef` 0.001. Vocab 248,320, untied embeddings. `mtp_num_hidden_layers` 1 (shares embeddings). 27-layer ViT (1152-d) vision tower. Context 262,144.
- Parameters (derived): experts 256 x 3 x (2048 x 512) = 805M per layer -> 32.2B (92% of the model); GDN layers ~33.7M each (1.01B); attention layers ~27.3M each (0.27B); shared experts 0.13B; routers 0.02B; embeddings + lm_head 1.02B; text total ~34.7B; MTP layer ~0.84B; vision ~0.41B.
- FLOP-active per token ~2.95B (8 experts 1.01B + shared 0.13B + GDN 1.01B + attention 0.27B + lm_head 0.51B).
- LoRA parameter budget: per expert per layer 3 x (2048 + 512) x r = 7,680 r -> all experts 78.6M x r (r=8: 0.63B, r=16: 1.26B, r=32: 2.5B). Dense projections (GDN + attention + shared expert): ~1.3M x r per layer -> ~50M at r=32. PEFT keeps adapters in fp32 by default (`autocast_adapter_dtype=True`): r=16 expert LoRA costs ~1.26B x (4 + 4 + 8) B = ~20 GB with AdamW; dense-only r=32 ~0.8 GB.

### A5. What this repo (fork master @ 966ed58bfc) already supports for serving the result

- Arch `qwen35moe` (`src/models/qwen35moe.cpp`) serves Qwen3.5/3.6-35B-A3B including the NEXTN (MTP) tensors; `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_XL (~22 GB) plus an MTP GGUF is known to run on the fork.
- Converter `conversion/qwen.py::Qwen3_5MoeTextModel` handles both expert layouts (fused 3-D `gate_up_proj`/`down_proj` as saved by transformers 5, and legacy per-expert tensors), remaps `mtp.*` to nextn layers, and can export the MTP head alone (`mtp_only`) or drop it (`no_mtp`). The GDN `in_proj_qkv` rows are reordered at conversion (`_LinearAttentionVReorderBase`).
- Runtime LoRA: `llama_adapter_lora`, `build_lora_mm` for dense projections and `build_lora_mm_id` for the routed experts (`src/llama-graph.cpp:1545`, used for gate/up/down exps in `build_moe_ffn`); the qwen35moe graph routes q/k/v/o, qkv/z/beta/alpha/out of the GDN block and the MoE FFN through them. `convert_lora_to_gguf.py` replays the base converter's `modify_tensors` on a `LoraTorchTensor` shim (permute, reshape, transpose, split, stack, cat); whether the V-reorder replays cleanly on the GDN LoRA tensors is untested.
- `llama-finetune` (`examples/training`) is FP32-only, no flash-attention backward, proven only on <= 1B models: not a candidate for a 35B MoE.
- The Flash-Next-only fork tricks (a1perm row permutation, K1 HC-collapse fusion, QSA) do not apply to qwen35moe, so a fine-tuned 35B-A3B is served on the plain qwen35moe path with standard `llama-quantize` + imatrix.

### A6. Memory and time envelope on dashi (derived from A3 + A4; to be replaced by Gate 0 measurements)

Training FLOPs per token for LoRA (the frozen base still needs activation gradients): forward ~5.9 GFLOP, backward ~5.9, checkpoint recompute ~5.9 => ~18 GFLOP/token (12 without recompute). Ceiling at 28 TFLOP/s: ~1,500 tok/s. The Gemma run realised ~40% of sustained peak; a 256-expert MoE dispatch (M ~ 128 rows per expert GEMM at 4k tokens) and unfused GDN kernels will realise less.

| Quantity | Value |
|---|---|
| Planning throughput band | 150-500 tok/s (must be measured at Gate 0) |
| 10M tokens | 5.5-18.5 h |
| 30M tokens | 17-56 h (above the box's 45 h continuous-load record -> needs checkpoint/resume) |
| 100M tokens | 2.3-7.7 days (not recommended on the box) |
| bf16 base, text-only load | 69.3 GB (leave the 0.8 GB vision tower and 1.7 GB MTP layer out of the training graph) |
| Activations, seq 4096 x batch 1, grad-ckpt | ~15 GB (dominated by 4096 x 248,320 fp32 logits + gradient ~8 GB) |
| Dense-only LoRA r=32 total | ~86 GB incl. ~5 GB allocator slack -> comfortable under `--mem-fraction 0.85` (105 GB) |
| Expert LoRA r=16 total | ~110 GB -> over the 105 GB cap; needs r <= 8-12, seq 2048, or an 8-bit optimizer |
| QLoRA (4-bit experts) | base ~25 GB but 1.8x slower on this chip (memory is not the binding constraint, time is) |

## Appendix B - Research findings (five web-research passes, 2026-09-15)

Full reports are archived in the session scratchpad (`research/A..F.md`). This appendix keeps the load-bearing facts and their sources. Confidence tags: H high, M medium, L low.

### B.1 The model family (report A)

- Releases: Qwen3.5-35B-A3B 2026-02-24, Qwen3.6-35B-A3B 2026-04-16, both Apache-2.0 (H). No Qwen3.7 line; no 35B-A3B in Qwen3.8; no official Qwen3.5/3.6 "Coder" (H). Qwen3.6-35B-A3B card: SWE-bench Verified 73.4, MMLU-Pro 85.2, LiveCodeBench v6 80.4 (H, self-reported).
- Post-training recipe undisclosed; official guidance is "use Unsloth / ms-swift / LLaMA-Factory", with only stale Qwen2.5 hyperparameters and no MoE advice (H).
- transformers implements experts as fused 3-D `nn.Parameter` (`gate_up_proj`, `down_proj`) with an expert loop; GDN uses `chunk_gated_delta_rule` from the kernels hub with a pure-torch fallback; `mtp.*` is dropped on load (`_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]`); router aux loss is computed only when `output_router_logits` is set, and the config ships it False with coef 0.001 (H).
- Ecosystem: 259 HF fine-tunes of Qwen3.6-35B-A3B, 165 of 3.5 (H). Measured coding gains come from RL at scale: Ornith-1.5 (self-improvement RL loop, MIT, SWE-bench Verified 79 self-reported, 4.15M GGUF downloads/mo, no data/hardware disclosed, M); Lego-X GSPO RL in the OpenHands SDK on 3 nodes x 8 GPUs -> 64.0 to 70.4 (M). Single-GPU community LoRA distills (rank 16-32, attention-only, 4-10k Claude traces, 1x H200, 2 epochs) report only tiny/noisy evals. `datajuicer/Juicer-35B-A3B` is the closest DE-oriented tune. No derivative reports training tokens/s or documents keeping the MTP head (H).
- Bugs to expect: PEFT shared-expert target rewrite (#3711); FA2 illegal memory access from 3-D `position_ids` (#44910, use sdpa/eager); fla 0.5.0 breaks Qwen3.5, downgrade to 0.4.2 (#792); tool-call template crash on `arguments | items` mapping (official repo discussion #4); stock llama.cpp GGUF conversion of the hybrid GDN tensors broken (#27019, #24737) though this fork's converter handles it (H).

### B.2 The platform (report B)

- ROCm 10.0.0 (2026-08-26) lists gfx1151 officially; AMD publishes gfx1151 torch wheels (2.11-2.13 on ROCm 7.14/10.0); pytorch.org ships none (H). `torch._grouped_mm` segfault on gfx1151 fixed from torch 2.11.0+rocm7.13.0 (H, Unsloth PR #5301).
- No gfx1151 AOTriton tuning DB through 0.13.x; SDPA FLASH/EFFICIENT with backward works at head_dim <= 256 (the Qwen3.5 shape) with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; CK flash-attention backward unsupported on RDNA (H).
- fla on ROCm: `chunk_gated_delta_rule` gradients match reference to 1.8e-6 (MI355X) but the causal-conv1d backward returned silently wrong gradients in fla 0.4.2/0.5.2 (#1156), and the AMD warp guard is not wave32-aware (#1163) (H/M). Without fla, transformers uses a correct-but-slow torch GDN path (M).
- Unsloth lists gfx1151 "full support" since Jul 2026 but Strix Halo users pin the 12x-slower `native_torch` MoE backend to dodge segfaults; the fast Triton MoE path is undocumented on AMD (M).
- **The only measured >= 20B MoE LoRA step on this silicon:** `woct0rdho/transformers5-qwen3.5-recipe` (Apache-2.0, pushed 2026-09-13) + `torch-ggml-ops` trains Qwen3.6-35B-A3B rank-4 LoRA on a GGUF base (Q3_K..Q6_K/Q8_0, native MMQ + grouped-MMQ + AITER GMM/PTGMM + FLA + Liger fused norms, 8-bit AdamW, router aux loss and KV cache disabled): warmed step 5.33 s / (batch 1 x 2048 tok) = forward 1.43 s + backward 3.83 s (32% checkpoint recompute), ~384 tok/s, peak 15.2 GiB with the whole model resident (H, single author, 15 stars). Top costs: FLA backward 1.3 s, then grouped-MMQ and GEMM.
- Community Strix Halo training (kyuz0 matrix, Gemma-3, no tok/s): 12B full FT 115 GB/25 min, 27B full FT and LoRA both OOM, GPT-OSS-20B LoRA 32-38 GB/~1 h; 8-bit LoRA 8-12x slower than bf16 (cross-validates our own measurement) (H). Independent Strix Halo report (report E #19): Qwen3.5-27B dense bf16 LoRA r=128, ~4 days / 448 steps, ~80 GB, flat eval loss, no downstream benchmark; APU-specific fixes needed (out-of-process eval to avoid a 1.2M/s TLB-shootdown storm, `--no-mmap`, warp cap) (M).
- Cloud reference (2026-09-15): H100 80GB PCIe $1.99-2.89/h, H100 SXM $2.69-3.49/h ($1.07 Vast), RTX PRO 6000 96GB $1.69-2.09/h, B200 $5.98-6.99/h; Unsloth's bf16 LoRA memory for Qwen3.5-35B-A3B is 74 GB (H).

### B.3 No-regression method (report C)

- Forgetting tracks KL(fine-tuned || base) on the new data; LoRA forgets less than full FT at equal domain gain and matches full FT up to ~100M SFT tokens, losing only at continued-pretraining scale (Biderman 2024, arXiv:2405.09673) (H).
- "LoRA Without Regret" (Thinking Machines 2025): apply LoRA to all matrices especially MLP/MoE, LR ~10x full FT and roughly rank-independent, keep effective batch < 32, r=256 for SFT / r=1-32 for RL, MoE = per-expert LoRA at rank/active-experts (H). On 256 experts PEFT's `r // num_experts` degenerates to rank 1, so either accept rank 1-4 per expert or adapt a subset ESFT-style (M).
- On-policy distillation restores general ability after domain training (IFEval 85 -> 45 -> 83 while keeping new knowledge) at 9-30x less compute than RL (Thinking Machines 2025) (H); KL-regularized self-generated replay "nearly eliminates" forgetting (arXiv:2605.26097) (M). RFT/RL with executable rewards naturally preserves prior skills (SWE-RL improved 5 OOD tasks where SFT degraded) (H).
- Replay 5% (weak shift) / 25% (strong); instruction-tuning practice clusters at 10-15% (H, arXiv:2403.08763).
- MoE practice: freeze the router (Unsloth default); PEFT `target_parameters` materializes the full per-expert delta (memory cost; the `baddbmm` fold that halves it, PR #3577, is unreleased), so `merge_and_unload` after training; ESFT beats plain LoRA by ~4.5-5.3 points within ~1 point of full FT but has never been applied to a Qwen3 MoE publicly (H/M).
- GDN specifics: LoRA the in/out projections, freeze `A_log`/`dt_bias`/short-conv (PEFT cannot name-target 1-D params anyway); packing leaks GDN state across documents unless `cu_seqlens` is supplied (Axolotl monkeypatch) (M). MTP/EAGLE head goes stale after trunk fine-tuning; retraining a head with a frozen trunk is tooled (SpecForge, ~200-500k samples) but no one publishes an acceptance-rate delta (M).
- Merging as insurance: merging a domain fine-tune back toward the base at alpha ~0.5 recovers general ability (LM-Cocktail, WiSE-FT) but is not a guarantee across pipelines; mergekit has no Qwen3.5/Qwen3-Next architecture definition (M).
- Evaluation: 1-sigma noise on a 164-198 item benchmark is ~3 points, so treat any single-run delta under ~6-8 points as noise; use paired per-question deltas (McNemar) on a fixed harness/template. lm-eval-harness MCQ tasks need a completion endpoint (`local-completions`), not chat-completions; EvalPlus/BigCodeBench/BFCL/Aider all drive an OpenAI-compatible base URL, so they run against the llama.cpp server (H).

### B.4 Data (report D, plus SWE and DS sub-agents)

- **Teacher licensing for a redistributable model** (H): Qwen (Apache-2.0), DeepSeek (MIT), GLM (MIT), gpt-oss (Apache-2.0), Kimi K2 (modified MIT) all permit training-and-redistribute. OpenAI / Anthropic / Google API outputs do not, and an MIT/CC tag on an HF repo does not override the upstream provider ToS. Taint list to exclude from anything published: SWE-smith-trajectories (Claude 3.7), SWE-Gym SFT + R2E-Gym (Claude/GPT-4o), Magicoder-OSS-Instruct (GPT-3.5), Code-Feedback (GPT-4), text_to_dbt (Claude), Think2SQL (Gemini).
- **Clean, released, execution-verified SFT sets:** SWE-Swiss-SFT-Merged-10K (DeepSeek-R1, MIT, 10,254 rows; SWE-Swiss-32B reached 60.2% Verified) and SWE-Dev `rft` split (DeepSeek-V3, verified, 2,276); Jupyter Agent Dataset (Apache-2.0, Qwen teachers, 51,389 x2, ~200M tok, E2B-executed); HuggingEnvs Data Agent SFT (MIT, deterministic grading, 4,677) and DataMind-12K (MIT, ~12k; DataMind-14B tops several DS benchmarks); TableLLM-SFT (MIT, executed, 73,157); OmniSQL/SynSQL-2.5M and Spider/BIRD train (H).
- **Cost of generating 50k verified samples** (H): non-thinking snippets ~$22-50 via DeepSeek/Qwen API vs ~20 days single-stream on the box; reasoning traces ~$81-188 vs ~86 days. Conclusion: the box is the wrong tool for generation and the right tool for verification (execute snippets in Docker, hours of CPU) and for the SFT run itself.
- **Chat-template rules** (H): ChatML; thinking on by default at 35B; for non-thinking rows emit an empty `<think></think>` and strip thinking from all but the last assistant turn; tools have two encodings (Hermes JSON vs Qwen3-Coder XML) and the shipped 3.5/3.6 template uses the XML form (`qwen3_xml` parser is stable, `qwen3_coder` regex breaks on `<`/`>` in code); tool results are user-role `<tool_response>`; assistant-only loss; Unsloth's published reasoning:non-reasoning ratio is 75:25.

**Concrete starter mixtures (report D section 5), decontaminated against every eval set before training:**

~10M tokens (Gate 2 first run), ~6,700 samples:

| Bucket | Source + filter | Tokens | % |
|---|---|---|---|
| DS notebooks | jupyter-agent-dataset both subsets, drop traces > 8k tok | 2.6M | 26 |
| Text-to-SQL | OmniSQL/SynSQL CoT + BIRD train + DuckDB-text2sql | 1.6M | 16 |
| Data engineering | data-engineering-sft rows + Airflow DAGs + dbt (self-gen, verified) | 1.2M | 12 |
| SWE agentic | SWE-Swiss / Orchard resolved, shortest quartile | 1.5M | 15 |
| General code | OpenCoder `package_instruct` + `educational_instruct` | 0.9M | 9 |
| Exec-feedback / tool-call | own-generated substitutes for the tainted GPT-4 sets | 0.7M | 7 |
| General replay | SmolTalk2 (60/40) + Dolci + Tulu-3 | 1.5M | 15 |

~50M tokens (Gate 3): scale each bucket ~5x and add depth (all Spider/BIRD/CoSQL/SParC train for SQL, DataMind-Analysis-SFT for DS, nebius SWE-rebench resolved trajectories for SWE, OpenCodeReasoning Python slice for thinking depth), replay ~15%.

Non-negotiables: drop Claude/GPT/Gemini-teacher rows from anything redistributable; exclude SWE-bench-Verified's 12 repos by name from every SWE source; n-gram-decontaminate SQL against Spider dev/test and BIRD dev; hold out data-eng-bench, DABStep and Spider2.0-DBT as eval only.

### B.5 Case studies (report E) - the calibration

- Every large SWE jump (SWE-Gym +13.6, SWE-smith +33, Nebius Qwen3-30B-A3B 25.7 -> 50.3, Scale-SWE 22 -> 64) came from multi-turn agent rollouts in real execution environments, at 16-64 GPU scale; real repos beat synthetic volume; imitation SFT saturates as the base gets stronger while RL does not (OmniSQL 2.5M samples / 20 A800-days = +0.0 at 32B on BIRD; Arctic-Text2SQL-R1 GRPO on the same base with 28k samples = +6.0) (H).
- Domain gains at LoRA/SFT scale are real and cheaper: SWE-Gym +13.6 points from ~500 trajectories; Jupyter Agent DABStep-easy 44 -> 70.8 from a 4B full FT on 51k notebooks (M/H). daVinci-Env measured positive transfer (+12 math, +5 science, no factual loss) - general collapse is not automatic (M).
- Nobody has published a coding/DS fine-tune trained on a 128 GB APU beyond the one 27B dense LoRA (flat loss, no benchmark), and nobody reports MTP/spec-decode acceptance change after a trunk fine-tune - both are genuine gaps this project would be entering (H).

### B.6 Key open questions carried into the plan

1. No measured throughput for rank-16 / seq-4096 35B-A3B LoRA on gfx1151 (only the recipe's rank-4 5.33 s/step) -> Gate 0 measures it.
2. fla causal-conv backward correctness on RDNA 3.5 wave32 is unverified -> Gate 0 one-layer gradient check.
3. MTP acceptance delta after trunk fine-tuning is unpublished -> measure at Gate 1/2; refresh the head if it drops.
4. Whether a LoRA on the fused experts round-trips through merge -> GGUF cleanly (llama.cpp #27019/#24737 open on stock; this fork's converter handles the base) -> Gate 1 parity test.
5. mergekit has no Qwen3.5 architecture def -> the merge-back insurance may need hand-rolled tensor arithmetic.
