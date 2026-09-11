---
license: apache-2.0
base_model: unsloth/gemma-4-E4B-it
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- aviation
- metar
- taf
- notam
- weather
- gemma-4
- structured-output
- fine-tuned
---

# gemma-4-e4b-avtext

A full fine-tune of **Gemma 4 E4B (instruction-tuned)** for structured decoding of aviation text:
**METAR** and **TAF** reports into canonical JSON, and **NOTAMs** into category-specific extraction
rows or one of 13 operational classes. One model covers all four tasks. The fine-tune was trained as a
rank-16 LoRA and merged into the base weights, so this repo is a plain Transformers checkpoint (plus
GGUF conversions for llama.cpp); the original adapter is included under `adapter/` for anyone who
prefers to apply it to the base themselves.

It is a research artifact from the *avtext* study (dataset engineering + evaluation harness +
fine-tuning on a single AMD Strix Halo box). It is **not** a certified aeronautical product: do not use
its output for operational or flight-safety decisions without independent verification.

## What's in the repo

| file(s) | format | size | use |
|---|---|---|---|
| `model-*.safetensors` + `config.json`, tokenizer and processor files | Transformers checkpoint, bf16, sharded | ~16 GB | `AutoModelForCausalLM.from_pretrained(<repo>)` |
| `gemma-4-e4b-avtext-Q8_0.gguf` | llama.cpp, 8-bit | ~8.5 GB | `llama-server -m …` (the quantization the study's numbers were measured with) |
| `gemma-4-e4b-avtext-f16.gguf` | llama.cpp, 16-bit | ~16 GB | for re-quantizing to other formats |
| `adapter/` | PEFT LoRA (rank 16) + GGUF LoRA | 140 MB + 70 MB | apply to `unsloth/gemma-4-E4B-it` instead of downloading merged weights |
| `prompts/` | text | — | the exact prompt templates the model was trained on (required, see *How to use*) |

The vision and audio towers of Gemma 4 E4B are carried over unchanged (the fine-tune touched only the
text tower); the model still loads with the multimodal classes but was trained and evaluated as a
text model.

## Results: before and after the adapter

Same frozen evals, same prompts, greedy decoding, same Q8_0 base served by llama.cpp; "before" is
the base model alone, "after" is the base with this adapter applied. One row per task, full sets.

| task | eval set | records | exact match (whole record) before → after | value recall before → after | hallucination before → after | invalid outputs before → after |
|---|---|---|---|---|---|---|
| METAR → JSON | avtext `v2` | 6,200 | 24.8 % → **94.4 %** (+69.6) | 88.8 % → 99.9 % | 21.4 % → 7.0 % | 0 → 0 |
| TAF → JSON | `taf-v1` (2048-token output cap) | 5,294 | _measuring_ → **93.3 %** | _measuring_ → 99.6 % | _measuring_ → 1.7 % | _measuring_ → 3 |
| NOTAM → extraction rows | `notam-v1` | 2,257 | 0.8 % → **82.3 %** (+81.5) | 11.0 % → 64.7 % | 14.9 % → 4.4 % | 64 → 154 |
| NOTAM → class (13) | `notam-cls-v1` | 4,047 | accuracy 78.2 % → **95.3 %** (+17.1); macro-F1 74.2 % → 94.3 % | — | — | 9 → 0 |

The base model knows the vocabulary but cannot hold a whole structured schema; the adapter teaches
the schema and the unit conventions, not new meteorology. The one number that moves the wrong way,
invalid NOTAM extractions (64 → 154), is the adapter attempting long "area" NOTAMs the base answers
with near-empty rows that parse trivially.

*Metric notes.* Exact match is per whole record. Value recall is the share of reference fields the
model reproduced with the correct value. Hallucination is the share of asserted values the reference
does not support. "Invalid" outputs (no parseable JSON) are scored as abstaining on every field. The
eval sets hold out unseen stations and unseen time windows. Exact definitions live in the avtext
harness (`score.py`, `score_taf.py`, `score_notam.py`).

*Protocol notes.* The base TAF row is being measured on the full set under the adapter's protocol
(the base's earlier 1,500-record score was 7.2 % exact match at a 1,024-token cap). The base
NOTAM-extraction run used 100-record llama.cpp sessions and the base METAR run the chat endpoint;
the adapter runs used 20-record sessions and the raw completion endpoint (the adapter is sensitive
to template drift, the base is not).

## How to use

The fine-tune is hard-tuned on **exact prompt templates**. Send the templates in `prompts/` verbatim
(the `{raw}` placeholder takes the report text); outputs drift off-distribution otherwise.

### Transformers

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "<this repo id>"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")

template = open("prompts/metar.txt", encoding="utf-8").read()
raw = "METAR EPGD 111200Z 27012KT 9999 FEW030 SCT045 18/09 Q1015 NOSIG"
msgs = [{"role": "user", "content": template.format(raw=raw)}]
enc = tok.apply_chat_template(
    msgs, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)
out = model.generate(**enc, max_new_tokens=400, do_sample=False)
text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
print(json.loads(text[text.index("{"): text.rindex("}") + 1]))
```

Use `max_new_tokens` ≥ 2048 for TAFs (long multi-period forecasts) and ≥ 512 for NOTAM extraction.

### llama.cpp

```bash
llama-server -m gemma-4-e4b-avtext-Q8_0.gguf --flash-attn on --reasoning-budget 0 -c 8192 --port 8080
```

Send the **raw** `/completion` endpoint the turn wrapper in `prompts/turn_wrapper.txt` around the
filled template (`<|turn>user\n{prompt}<turn|>\n<|turn>model\n`; the server prepends BOS) with
`temperature 0`. The chat endpoint's template engine renders Gemma 4's chat template slightly
differently from HF Transformers, and this fine-tune is sensitive to that drift.

For NOTAM extraction the study's harness additionally constrains decoding with a JSON grammar derived
from `prompts/notam_fields.json` (the row schema per category); without a grammar expect a few more
invalid outputs on long NOTAMs.

### Adapter instead of merged weights

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("unsloth/gemma-4-E4B-it", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, repo, subfolder="adapter")
```

## Training

| | |
|---|---|
| base | `unsloth/gemma-4-E4B-it` (weights mirror of `google/gemma-4-E4B-it`), bf16 |
| method | LoRA rank 16, alpha 32, dropout 0, on `q/k/v/o/gate/up/down_proj` of the **text tower only** (vision/audio towers untouched); 34.9 M trainable parameters, merged into the base weights after training (`merge_and_unload`) |
| data | 49,214 chat examples: 20,000 METAR, 16,000 TAF, 9,049 NOTAM extraction, 4,165 NOTAM classification |
| schedule | 1 epoch, AdamW (lr 2e-4, weight decay 0.01, linear decay, 10 warm-up steps), batch 6 × grad-accum 2, max sequence 2,048 tokens |
| tricks | length-grouped batching, `torch.compile`, length-adaptive gradient checkpointing (recompute only above 1,800 tokens) |
| hardware | one AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151, 123 GiB unified memory), ROCm/TheRock nightly PyTorch; 16 h 58 min |
| final train loss | ≈ 0.13 (mean of the last 100 steps; 0.62 over the first 100) |

## Limitations

- Research decode aid, not a certified aeronautical tool. Verify independently before any operational use.
- METAR shows a ~7 % hallucination floor shared by every fine-tune in the study (mostly optional fields
  the reference leaves empty).
- TAF: long multi-period forecasts need a ≥ 2,048-token output budget; below that ~3 % of outputs are truncated.
- NOTAM extraction: ~7 % of very long "area" NOTAMs (airspace restrictions with coordinate lists) yield
  no valid output; the model is also sensitive to prompt drift and server session length (restart
  llama.cpp sessions periodically on long batches).
- English / ICAO-format inputs only; trained on the four prompt templates shipped here.

## License and notices

These weights are released under the **Apache License 2.0** (see `LICENSE`). Gemma 4 E4B is
released by Google DeepMind under the Apache License 2.0 and subject to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), which also applies to
this derivative. See `NOTICE` for attributions. Gemma is a trademark of Google LLC; this project is not
affiliated with or endorsed by Google.
