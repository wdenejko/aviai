"""Faithful greedy generation on a training-shape-only MMQ bundle.

The bundle compiles kernels for the TRAINING geometry only: dense M=2048 and grouped-pair r=16384
(= 2048 tokens x top-8). Ordinary generation fails because prefill uses M=prompt_len and each decode
step uses M=1, neither of which is compiled -- and swapping the experts to a stock implementation
would silently drop the expert LoRA, making the comparison unfaithful.

So we decode inside a FIXED 2048-token window instead: the prompt is right-padded to exactly 2048,
every step re-runs the whole window (use_cache=False), and the next token is read from the logit at
the last real position. Every op therefore sees the compiled shape and the COMPLETE adapter
(attention + experts + GDN) is applied. The price is a full-window forward per token (~2s), which is
acceptable for a short qualitative check.

Base vs adapter uses the same B=0 trick as the loss eval: freshly injected LoRA is numerically the
base while already on the MMQ path, so the only thing that changes is the adapter weights.
"""
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
ADAPTER = os.environ.get("ADAPTER", RECIPE + "/out_qwen36_35b/final")
SEQ = 2048
MAX_NEW = int(os.environ.get("MAX_NEW", "40"))
# Optional: restrict to one prompt (index) so a longer budget can reach the actual answer.
_IDX = os.environ.get("PROMPT_IDX", "")

PROMPTS = [
    "Write a DuckDB SQL query returning the number of flights per day for the last 7 days from a "
    "table flights(dep_time TIMESTAMP, origin TEXT). Use DuckDB date functions.",
    "You are given a CSV of labeled training rows and asked to deliver a model. List the concrete "
    "steps you will take before reporting a final accuracy number.",
]

if _IDX != "":
    PROMPTS = [PROMPTS[int(_IDX)]]

configure_compiled_gguf_dequantize()
configure_qwen35_flash_attention_2()
configure_qwen35_fla()

tok = AutoTokenizer.from_pretrained(MODEL_DIR, gguf_file=GGUF, local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
pad_id, eos_id = tok.pad_token_id, tok.eos_token_id

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


def greedy(prompt: str) -> str:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    buf = list(tok(text, add_special_tokens=False)["input_ids"])
    if len(buf) >= SEQ:
        raise ValueError("prompt does not fit the fixed window")
    produced = []
    for _ in range(MAX_NEW):
        n = len(buf)
        window = buf + [pad_id] * (SEQ - n)
        ids = torch.tensor([window], dtype=torch.int64, device="cuda:0")
        mask = torch.tensor([[1] * n + [0] * (SEQ - n)], dtype=torch.int64, device="cuda:0")
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        nxt = int(torch.argmax(logits[0, n - 1]))
        if nxt == eos_id:
            break
        produced.append(nxt)
        buf.append(nxt)
        if len(buf) >= SEQ:
            break
    return tok.decode(produced, skip_special_tokens=True)


print("== PASS 1: base (B=0) ==", flush=True)
base_out = [greedy(p) for p in PROMPTS]
for i, o in enumerate(base_out):
    print(f"--- BASE[{i}] ---\n{o}\n", flush=True)

print("== loading trained adapter ==", flush=True)
set_peft_model_state_dict(model, load_file(ADAPTER + "/adapter_model.safetensors"))
model.eval()

print("== PASS 2: with adapter ==", flush=True)
adpt_out = [greedy(p) for p in PROMPTS]
for i, o in enumerate(adpt_out):
    print(f"--- ADAPTER[{i}] ---\n{o}\n", flush=True)

print("=" * 76)
for i, p in enumerate(PROMPTS):
    print(f"PROMPT {i}: {p[:80]}...")
    print(f"  base   : {base_out[i][:300]!r}")
    print(f"  adapter: {adpt_out[i][:300]!r}")
    print(f"  changed: {base_out[i].strip() != adpt_out[i].strip()}")
