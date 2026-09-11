---
license: apache-2.0
base_model: unsloth/gemma-4-E4B-it
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- aviation
- metar
- taf
- notam
- weather
- lora
- gemma-4
- structured-output
---

# gemma-4-e4b-avtext-lora-r16

A rank-16 LoRA adapter that turns **Gemma 4 E4B (instruction-tuned)** into a structured decoder for
aviation text: **METAR** and **TAF** reports into canonical JSON, and **NOTAMs** into category-specific
extraction rows or one of 13 operational classes. One adapter covers all four tasks.

It is a research artifact from the *avtext* study (dataset engineering + evaluation harness +
fine-tuning on a single AMD Strix Halo box). It is **not** a certified aeronautical product: do not use
its output for operational or flight-safety decisions without independent verification.

## Results (full frozen evals, greedy decoding)

| task | eval set | records | exact match (whole record) | value recall | hallucination | invalid outputs |
|---|---|---|---|---|---|---|
| METAR → JSON | avtext `v2` | 6,200 | **94.4 %** | 99.9 % | 7.0 % | 0 |
| TAF → JSON | `taf-v1` (2048-token output cap) | 5,294 | **93.3 %** | 99.6 % | 1.7 % | 3 |
| NOTAM → extraction rows | `notam-v1` (server restart every 20 records) | 2,257 | **82.3 %** | 64.7 % | 4.4 % | 154 |
| NOTAM → class (13) | `notam-cls-v1` | 4,047 | **accuracy 95.3 %**, macro-F1 94.3 % | — | — | 0 |

The served base model without the adapter scores 24.8 % exact match on the METAR eval (recall 88.8 %,
hallucination 21.4 %), 7.2 % on a 1,500-record TAF subset, and 78.2 % accuracy on NOTAM classification:
the adapter teaches the schema and the unit conventions, not new meteorology.

*Metric notes.* Exact match is per whole record. Value recall is the share of reference fields the
model reproduced with the correct value. Hallucination is the share of asserted values the reference
does not support. "Invalid" outputs (no parseable JSON) are scored as abstaining on every field. The
eval sets hold out unseen stations and unseen time windows, and the training set was audited to be
exactly disjoint from all of them. Exact definitions live in the avtext harness (`score.py`,
`score_taf.py`, `score_notam.py`).

## How to use

The adapter is a hard fine-tune on **exact prompt templates**. Send the templates in `prompts/`
verbatim (the `{raw}` placeholder takes the report text); outputs drift off-distribution otherwise.

### Transformers + PEFT

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "unsloth/gemma-4-E4B-it"          # mirror of google/gemma-4-E4B-it
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "<this repo id>")

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

### llama.cpp (GGUF LoRA)

`gemma-4-e4b-avtext-lora-r16-f16.gguf` applies on top of any GGUF conversion of the same base
(the study used a Q8_0 conversion of `unsloth/gemma-4-E4B-it`):

```bash
llama-server -m gemma-4-E4B-it-Q8_0.gguf --lora gemma-4-e4b-avtext-lora-r16-f16.gguf \
  --flash-attn on --reasoning-budget 0 -c 8192 --port 8080
```

Send the **raw** `/completion` endpoint the turn wrapper in `prompts/turn_wrapper.txt` around the
filled template (`<|turn>user\n{prompt}<turn|>\n<|turn>model\n`; the server prepends BOS) with
`temperature 0`. The chat endpoint's template engine renders Gemma 4's chat template slightly
differently from HF Transformers, and this fine-tune is sensitive to that drift.

For NOTAM extraction the harness additionally constrains decoding with a JSON grammar derived from
`prompts/notam_fields.json` (the row schema per category); without a grammar expect a few more invalid
outputs on long NOTAMs.

## Training

| | |
|---|---|
| base | `unsloth/gemma-4-E4B-it` (weights mirror of `google/gemma-4-E4B-it`), bf16 |
| adapter | LoRA rank 16, alpha 32, dropout 0, on `q/k/v/o/gate/up/down_proj` of the **text tower only** (vision/audio towers untouched); 34.9 M trainable parameters |
| data | 49,214 chat examples: 20,000 METAR, 16,000 TAF, 9,049 NOTAM extraction, 4,165 NOTAM classification |
| schedule | 1 epoch, AdamW (lr 2e-4, weight decay 0.01, linear decay, 10 warm-up steps), batch 6 × grad-accum 2, max sequence 2,048 tokens |
| tricks | length-grouped batching, `torch.compile`, length-adaptive gradient checkpointing (recompute only above 1,800 tokens) |
| hardware | one AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151, 123 GiB unified memory), ROCm/TheRock nightly PyTorch; 16 h 58 min |
| final train loss | ≈ 0.13 (mean of the last 100 steps; 0.62 over the first 100) |

## Data

- **METAR / TAF**: raw reports from the Iowa Environmental Mesonet ASOS/AFOS archive and NOAA's
  Aviation Weather Center (US Government works, public domain). Decode targets are parser-consensus
  decodes calibrated field by field against NOAA's official decodes (≥ 98.7 % agreement per field).
- **NOTAM classification**: `DEEL-AI/NOTAM` (MIT).
- **NOTAM extraction**: `Estrellajer/OpenNOTAM`, used with the author's permission
  (**GRANT REFERENCE: _to be filled by the maintainer_**).
- No records from the evaluation sets were used in training (record-level exact-match audit across
  all five eval files; 45 overlapping records were removed before training).

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

The adapter weights are released under the **Apache License 2.0** (see `LICENSE`). Gemma 4 E4B is
released by Google DeepMind under the Apache License 2.0 and subject to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), which also applies to
this derivative. See `NOTICE` for attributions. Gemma is a trademark of Google LLC; this project is not
affiliated with or endorsed by Google.
