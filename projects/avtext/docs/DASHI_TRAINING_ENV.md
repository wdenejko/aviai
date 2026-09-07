# dashi training environment — the gfx1151 LoRA recipe

> **Why this file exists.** The original finetune recipe (an `unsloth`-based
> `train_lora.py`) was **lost** when dashi's `~/ft` working tree was wiped, and reconstructing
> it cost most of Session 23. This document is the byte-level recovery recipe so it is **never
> lost again**. If dashi is rebuilt, following this file top-to-bottom reproduces a working
> Gemma-4-E4B LoRA trainer that trains on the iGPU and converts to a servable GGUF.

## The machine

| | |
|---|---|
| Host | GMKtec EVO-X2 · AMD Ryzen AI Max+ 395 "Strix Halo" |
| iGPU | Radeon 8060S → **gfx1151** (this is the whole problem — see below) |
| RAM | 123 GiB unified (CPU+GPU share it; no discrete VRAM) |
| OS | Fedora 43, kernel 7.1.3 · host Python **3.14** (too new for torch) |
| Access | `ssh dashi` (192.168.0.131) |

## The one hard problem: gfx1151 has no stock kernels

`gfx1151` (Strix Halo) is **not** in any pytorch.org ROCm wheel. Installing the stock
`torch ... --index-url https://download.pytorch.org/whl/rocm6.4` gives a torch that loads but
dies at the first matmul with:

```
RuntimeError: HIP error: invalid device function      # a.k.a. hipErrorNoBinaryForGpu
```

The fix is **AMD's TheRock nightly, built per-arch for gfx1151**. No `HSA_OVERRIDE_GFX_VERSION`
is needed — these are *native* gfx1151 kernels, not a gfx1100 override.

## Recipe — rebuild from scratch

### 1. venv on Python 3.12 (host 3.14 is too new for torch)

```bash
# uv is already on dashi; make a 3.12 venv for torch
uv venv --python 3.12 ~/fttorch
```

### 2. TheRock gfx1151 torch (the critical step)

```bash
~/fttorch/bin/python -m pip install --pre torch \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
# yields: torch 2.12.0a0+rocm7.13...  (native gfx1151 kernels)
```

### 3. The rest of the finetune stack

```bash
~/fttorch/bin/python -m pip install \
  transformers==5.16.1 trl==1.12.0 peft==0.20.0 \
  datasets accelerate sentencepiece protobuf pillow
```

Pinned versions that are known-good together (captured 2026-09-07):

| pkg | version | pkg | version |
|---|---|---|---|
| torch | 2.12.0a0+rocm7.13 | datasets | 5.0.1 |
| transformers | 5.16.1 | accelerate | 1.14.0 |
| peft | 0.20.0 | numpy | 2.5.3 |
| trl | 1.12.0 | | |

### 4. Run **inside the toolbox**, not on the bare host

The bare host is missing `libatomic.so.1` (torch import fails) and does not wire the GPU
devices into an arbitrary process. Everything runs inside the kyuz0 Strix-Halo toolbox
container **`llama-rocm-7.2.4_2`**, which supplies `libatomic` + `/dev/kfd` + `/dev/dri`:

```bash
toolbox run --container llama-rocm-7.2.4_2 bash -lc '<command>'
```

Sanity check the whole stack in one line:

```bash
toolbox run --container llama-rocm-7.2.4_2 bash -lc \
  '~/fttorch/bin/python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
# -> True Radeon 8060S Graphics
```

## Train — `scripts/dashi/train_lora_peft.py`

Standard `transformers` + `peft` + `trl` SFT (no unsloth). Two Gemma-4-specific gotchas are
baked in, both non-obvious and both fatal if missed:

1. **Text-tower-only LoRA target regex.** Gemma-4-E4B is multimodal (text + vision + audio
   towers). The vision/audio projections are `Gemma4ClippableLinear` (a wrapper, *not*
   `nn.Linear`) → PEFT raises `Target module ... is not supported`. Restrict targets to the
   text tower:

   ```python
   TEXT_TARGETS = r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
   ```

2. **`processing_class=tok`, not an AutoProcessor.** trl otherwise auto-loads the multimodal
   `Gemma4Processor`, which hard-requires PIL and drags in the vision path. Passing the plain
   tokenizer as `processing_class` keeps it text-only. (`pillow` is still installed as a
   belt-and-braces import guard.)

Invoke via the wrapper (defaults to the right toolbox):

```bash
~/scripts/avtext/run_peft_train.sh \
  --model unsloth/gemma-4-E4B-it \
  --data ~/fttrain/train_all.jsonl \
  --out  ~/fttrain/adapter-all \
  --rank 16 --epochs 1 --batch 6 --packing
```

LoRA config that this produces: **rank 16, alpha 32 (2·rank), dropout 0**, the 7 text-tower
proj modules → **34.9M trainable params (0.44%)**.

## Convert adapter → GGUF (for serving/eval)

The PEFT adapter's safetensors keys carry the multimodal nesting
(`base_model.model.model.language_model.layers.N...`). **This is fine** —
`convert_lora_to_gguf.py` knows the `gemma4` arch and maps them to llama.cpp text names
(`blk.N.ffn_gate.weight.lora_a/b`). No key-stripping needed.

```bash
toolbox run --container llama-rocm-7.2.4_2 bash -lc '
  cd ~/gfx1151-fork
  PYTHONPATH=~/gfx1151-fork/gguf-py ~/fttorch/bin/python convert_lora_to_gguf.py \
    ~/fttrain/adapter-all \
    --base ~/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it/snapshots/<SNAP>/ \
    --outfile ~/fttrain/adapter-all-f16.gguf --outtype f16'
```

Serve it with llama-server `--lora adapter-all-f16.gguf` on top of the base GGUF, then eval
with the harness (see `models_notam` / `runner_notam`). ⚠️ Long eval runs hit the
`<unused49>` serving-degradation bug — use the chunked evaluator (`eval_notam_chunked.sh`).

## The five failures that cost Session 23 (each maps to a step above)

| symptom | cause | fix |
|---|---|---|
| `hipErrorNoBinaryForGpu` / `invalid device function` | stock rocm6.4 wheel has no gfx1151 kernels | TheRock gfx1151 nightly (step 2) |
| `libatomic.so.1: cannot open shared object file` | missing on bare host | run inside toolbox (step 4) |
| `Target module Gemma4ClippableLinear is not supported` | PEFT can't LoRA the vision/audio towers | text-tower target regex |
| `Gemma4Processor requires the PIL library` | trl auto-loaded the multimodal processor | `processing_class=tok` |
| adapter won't convert (feared) | *non-issue* — keys carry `model.language_model.` prefix | convert script handles gemma4 natively |
