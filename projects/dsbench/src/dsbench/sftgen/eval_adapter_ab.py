"""A/B eval: does assistant-only loss masking change what the fine-tune actually learns?

Gate 1 trained with loss over the WHOLE packed block, so ~46% of the gradient went on prompt, tool
and system text. The control bucket then showed never-trained general data improving as much as the
target data -- consistent with the run fitting the rendering rather than the behaviour. This script
compares the two adapters under identical conditions.

Two metrics per bucket, because they answer different questions:
  * full      -- loss over every token in the block (what Gate 1 originally reported)
  * assistant -- loss over assistant-authored tokens only (what the model actually has to generate)

All states are measured in ONE model load, on the SAME blocks, through the SAME packed-MMQ path.
The base state is a freshly injected LoRA (B=0), which is numerically the base model, so no
path/numerics difference can contaminate the comparison. Adapters are then loaded in turn.

Runs on the box (dashi). Blocks stay at 2048 -- the MMQ bundle's compiled geometry.

    ADAPTERS="unmasked=/path/a,masked=/path/b" python eval_adapter_ab.py
"""

import collections
import json
import os
import sys

import torch

RECIPE = "/home/wdenejko/src/transformers5-qwen3.5-recipe"
sys.path.insert(0, RECIPE)
sys.path.insert(0, os.path.expanduser("~"))

from attention_aiter_tuning import configure_qwen35_flash_attention_2  # noqa: E402
from fast_lora import register_fast_lora  # noqa: E402
from fast_moe_lora import register_fast_moe_lora  # noqa: E402
from fast_moe_ranking import configure_fast_moe_ranking  # noqa: E402
from fla_tuning import configure_qwen35_fla  # noqa: E402
from gguf_dequant_compile import configure_compiled_gguf_dequantize  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict  # noqa: E402
from qwen3_5_fused_norms import configure_qwen35_fused_norms  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from tokenize_masked import pack  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_DIR = "/home/wdenejko/models/qwen3.6"
GGUF = "Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
EVAL = os.environ.get("EVAL_BUCKETS", "/home/wdenejko/eval_buckets_ab.jsonl")
OUT = os.environ.get("EVAL_OUT", "/home/wdenejko/gate1_ab_result.json")
SEQ, NBLOCKS = 2048, 16

configure_compiled_gguf_dequantize()
configure_qwen35_flash_attention_2()
configure_qwen35_fla()

tok = AutoTokenizer.from_pretrained(MODEL_DIR, gguf_file=GGUF, local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

by_bucket = collections.defaultdict(list)
for line in open(EVAL):
    record = json.loads(line)
    by_bucket[record["bucket"]].append(record)

blocks = {}
for name, records in by_bucket.items():
    ids, labels, stats = pack(tok, records, SEQ, tok.eos_token_id)
    blocks[name] = (ids[:NBLOCKS], labels[:NBLOCKS])
    print(f"{name:18s} {len(ids[:NBLOCKS])} blocks | assistant tokens "
          f"{stats['trainable_token_pct']}%", flush=True)

print("== loading base + recipe LoRA injection ==", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, gguf_file=GGUF, gguf_mmap_policy="release", local_files_only=True,
    dtype=torch.bfloat16, device_map={"": "cuda:0"},
    attn_implementation=os.environ.get("QWEN35_ATTN_IMPL", "flash_attention_2"),
)
model.config.use_cache = False
model.config.output_router_logits = False
model.config.router_aux_loss_coef = 0.0
configure_fast_moe_ranking(model)
configure_qwen35_fused_norms(model)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj",
                    "in_proj_qkv", "in_proj_z", "out_proj", "experts"],
    r=4, lora_alpha=4, use_rslora=False,
)
register_fast_lora(lora_config, model)
register_fast_moe_lora(lora_config, model, expert_prior="qwen-learned")
model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
model.eval()


def measure() -> dict:
    out = {}
    for name, (ids, labels) in blocks.items():
        full_total, asst_total, counted = 0.0, 0.0, 0
        for block_ids, block_labels in zip(ids, labels, strict=True):
            tensor = torch.tensor([block_ids], dtype=torch.int64, device="cuda:0")
            mask = torch.ones_like(tensor)
            masked = torch.tensor([block_labels], dtype=torch.int64, device="cuda:0")
            with torch.no_grad():
                full = model(input_ids=tensor, attention_mask=mask, labels=tensor,
                             use_cache=False).loss
                asst = model(input_ids=tensor, attention_mask=mask, labels=masked,
                             use_cache=False).loss
            if not torch.isfinite(asst):  # a block with no assistant token contributes nothing
                continue
            full_total += float(full)
            asst_total += float(asst)
            counted += 1
        out[name] = {"full": full_total / max(counted, 1),
                     "assistant": asst_total / max(counted, 1), "blocks": counted}
        print(f"   {name:18s} full {out[name]['full']:.4f}  assistant "
              f"{out[name]['assistant']:.4f}", flush=True)
    return out


results = {}
print("== state: base (B=0) ==", flush=True)
results["base"] = measure()

for spec in os.environ.get("ADAPTERS", "").split(","):
    if "=" not in spec:
        continue
    label, path = spec.split("=", 1)
    print(f"== state: {label} ({path}) ==", flush=True)
    set_peft_model_state_dict(model, load_file(path.rstrip("/") + "/adapter_model.safetensors"))
    model.eval()
    results[label] = measure()

print("\n" + "=" * 92)
states = list(results)
print(f"{'bucket':18s} {'metric':10s} " + " ".join(f"{s:>12s}" for s in states) + "   vs-base")
print("-" * 92)
for bucket in blocks:
    for metric in ("full", "assistant"):
        row = [results[s][bucket][metric] for s in states]
        deltas = " ".join(f"{100 * (v - row[0]) / row[0]:+11.1f}%" for v in row[1:])
        print(f"{bucket:18s} {metric:10s} " + " ".join(f"{v:12.4f}" for v in row) + f"   {deltas}")
print("=" * 92)
json.dump(results, open(OUT, "w"), indent=2)
print("wrote", OUT)
