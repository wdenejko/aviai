"""Gate-1 fixed-M behavior-shift eval.

Free-form generation is impossible here (the torch-ggml-ops MMQ bundle is compiled only for the
training geometry M=2048), so we measure the fine-tune's effect with fixed-M=2048 forward loss —
the one shape the bundle supports.

Design: build the wrapped model ONCE exactly as train_qwen3_5_35b.py does. Freshly injected LoRA has
B=0, so the model is numerically the base model while still running the SAME packed-MMQ code path.
Measure per-bucket loss, then load the trained adapter weights and measure again. The only thing
that changed is the adapter, so the delta is purely the fine-tune (no path/numerics confound).

Buckets separate GENERALIZATION from MEMORIZATION: *_heldout rows were never trained on, *_trained
rows were in the pilot. tulu3_trained is a replay REFERENCE, not a control -- those rows were
trained on, so its delta includes memorisation and says nothing about forgetting.

Runs ON THE BOX (dashi), inside the build toolbox with the ftgguf venv, because it imports the
recipe modules from ~/src/transformers5-qwen3.5-recipe and needs the GGUF base on the GPU:
    PYTHONPATH=$HOME/src/aiter QWEN35_ATTN_IMPL=kernels-community/aiter-flash-attn \
        python eval_adapter.py [adapter_dir]
"""
import collections
import json
import os
import sys

import torch

RECIPE = "/home/wdenejko/src/transformers5-qwen3.5-recipe"
sys.path.insert(0, RECIPE)

from attention_aiter_tuning import configure_qwen35_flash_attention_2  # noqa: E402
from fast_lora import register_fast_lora  # noqa: E402
from fast_moe_lora import register_fast_moe_lora  # noqa: E402
from fast_moe_ranking import configure_fast_moe_ranking  # noqa: E402
from fla_tuning import configure_qwen35_fla  # noqa: E402
from gguf_dequant_compile import configure_compiled_gguf_dequantize  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict  # noqa: E402
from qwen3_5_fused_norms import configure_qwen35_fused_norms  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_DIR = "/home/wdenejko/models/qwen3.6"
GGUF = "Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else RECIPE + "/out_qwen36_35b/final"
EVAL = "/home/wdenejko/eval_buckets.jsonl"
SEQ = 2048           # MUST match the MMQ bundle's compiled geometry
BLOCKS_PER_BUCKET = 16

configure_compiled_gguf_dequantize()
configure_qwen35_flash_attention_2()
configure_qwen35_fla()

tok = AutoTokenizer.from_pretrained(MODEL_DIR, gguf_file=GGUF, local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
eos = tok.eos_token_id

# ---- pack each bucket into fixed 2048-token blocks (same packing as training) ----
streams = collections.defaultdict(list)
for line in open(EVAL):
    r = json.loads(line)
    try:
        text = tok.apply_chat_template(r["messages"], tools=r.get("tools"), tokenize=False,
                                       add_generation_prompt=False)
    except Exception:
        text = "\n".join(m.get("content") or "" for m in r["messages"]
                         if isinstance(m.get("content"), str))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    streams[r["bucket"]].extend(int(t) for t in ids)
    if eos is not None:
        streams[r["bucket"]].append(int(eos))

blocks = {}
for b, ids in streams.items():
    bl = [ids[i:i + SEQ] for i in range(0, len(ids) - SEQ + 1, SEQ)][:BLOCKS_PER_BUCKET]
    blocks[b] = bl
    print(f"bucket {b:18s}: {len(ids):7,} tok -> {len(bl)} blocks", flush=True)

print("== loading base + injecting recipe LoRA ==", flush=True)
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


def bucket_losses():
    out = {}
    for b, bl in blocks.items():
        tot = 0.0
        for blk in bl:
            ids = torch.tensor([blk], dtype=torch.int64, device="cuda:0")
            mask = torch.ones_like(ids)
            with torch.no_grad():
                r = model(input_ids=ids, attention_mask=mask, labels=ids, use_cache=False)
            tot += float(r.loss)
        out[b] = tot / max(len(bl), 1)
        print(f"   {b:18s} loss {out[b]:.4f}", flush=True)
    return out


print("== PASS 1: base (freshly injected LoRA, B=0 -> zero contribution) ==", flush=True)
base = bucket_losses()

print("== loading trained adapter ==", flush=True)
sd = load_file(ADAPTER + "/adapter_model.safetensors")
set_peft_model_state_dict(model, sd)
model.eval()
nz = sum(int(torch.count_nonzero(v)) for k, v in sd.items() if "lora_B" in k or "lora_B_" in k)
print(f"   adapter tensors: {len(sd)} | nonzero elements in B factors: {nz:,}", flush=True)

print("== PASS 2: with trained adapter ==", flush=True)
adpt = bucket_losses()

print("\n" + "=" * 74)
print(f"{'bucket':20s} {'base':>9s} {'adapter':>9s} {'delta':>9s} {'%':>8s}")
print("-" * 74)
for b in sorted(base):
    d = adpt[b] - base[b]
    print(f"{b:20s} {base[b]:9.4f} {adpt[b]:9.4f} {d:+9.4f} {100*d/base[b]:+7.1f}%")
print("=" * 74)
json.dump({"base": base, "adapter": adpt}, open("/home/wdenejko/gate1_eval_result.json", "w"),
          indent=2)
print("wrote /home/wdenejko/gate1_eval_result.json")
