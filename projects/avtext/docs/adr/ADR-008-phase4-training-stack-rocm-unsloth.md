# ADR-008: Phase 4 training stack — Unsloth bf16 LoRA on Strix Halo via ROCm

- **Status:** Accepted
- **Date:** 2026-07-30
- **Builds on:** ADR-007 (Phase 4 on dashi). Resolves its "open: the training stack" item.

## Context

dashi had **no training stack** — no torch/peft/unsloth anywhere (the `*-Unsloth` model dirs
are inference *quants*, not a trainer). The GPU (Radeon 8060S, gfx1151 / RDNA 3.5) drove
llama.cpp inference via **Vulkan**, but PyTorch training needs the ROCm/KFD compute path.
The user chose **Unsloth on the AMD GPU** — a frontier combination (Unsloth is CUDA-first,
gfx1151 is new silicon). This ADR records the stack that actually worked, as a runbook.

## Decision

Train with **Unsloth bf16 LoRA on PyTorch-ROCm**, in a dedicated venv (`~/ft`, Python 3.12),
running **on the host** (not a toolbox — torch-rocm bundles its own ROCm and the host has
`/dev/kfd` + amdgpu). bf16, not 4-bit QLoRA: the small models fit on 123 GiB, so no
quantised training is needed. The finetuned adapter is later merged and converted to **GGUF**
so eval runs the *same* llama.cpp path as the baseline (same-stack McNemar, ADR-007).

### Runbook (what worked, and the traps)

```bash
uv venv ~/ft --python 3.12                     # NOT system 3.14 — no ML wheels for it
uv pip install --python ~/ft/bin/python "torch>=2.11,<2.12" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm7.2   # matches ROCm 7.2.4; native gfx1151
uv pip install --python ~/ft/bin/python "unsloth[amd]"
# bitsandbytes: uv rejects the preview wheel's non-PEP440 version "1.33.7.preview";
# bootstrap pip and install with it instead:
~/ft/bin/python -m ensurepip --upgrade
~/ft/bin/python -m pip install --no-deps <bitsandbytes AMD preview wheel URL>
```

Traps hit, in order:
1. **System Python is 3.14** — no torch/unsloth wheels. Pin **3.12** via uv.
2. **`torch-rocm7.2` sees the GPU natively** — `cuda.is_available()==True`, device "AMD Radeon
   8060S", a real matmul runs. **No `HSA_OVERRIDE_GFX_VERSION` needed** (ROCm 7.2 has genuine
   gfx1151 support; the override is only a fallback if kernels fail to compile).
3. **uv won't install the bitsandbytes preview wheel** ("Must have a version" — its `1.33.7.preview`
   isn't PEP 440). Use `pip` (lenient); it lands `bitsandbytes 0.50.1.dev0`.
4. **bitsandbytes is required even for bf16** — Unsloth imports `Linear4bit` during model
   patching. It's imported but never *used* in bf16; its "Could not detect ROCm arch (no
   rocminfo)" warning is benign for our path.
5. **Unsloth's SFTTrainer needs a `text` field** — it doesn't auto-format `{"messages":[...]}`;
   apply the chat template into `text` yourself (see `finetune/train_lora.py`).

Verified end-to-end: gemma-3-1b LoRA, 3000 examples, loss logged and decreasing, adapter saved.

## Consequences

- **+** Real GPU LoRA on dashi, self-consistent with the ADR-007 eval (train + serve + score
  all on the same box / same llama.cpp path).
- **+** A reproducible runbook for PyTorch/Unsloth on Strix Halo — reusable for any model.
- **−** **Slow:** ~14 s/step for a 1B LoRA (early ROCm 7.2; no CK/flash-attn kernels on gfx1151
  yet). A 1-epoch 1B run is ~45–90 min; 4B will be multiples of that. Acceptable in the
  background, but not fast iteration.
- **−** bf16 only in practice (4-bit QLoRA blocked by the AMD bitsandbytes 4-bit NaN issue on
  older versions and arch-detection quirks) — fine for ≤4B on 123 GiB.
- **Open:** does Unsloth load **gemma-3n** (E2B/E4B, the multimodal PLE arch)? The dense Gemma
  path is proven; the 3n anchor is TBD — if unsupported, the finetune study runs on dense
  Gemma rungs and E4B stays a base-only reference.
