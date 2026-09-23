"""Minimal OpenAI-compatible server for the GGUF-base + LoRA model, for agentic benchmarking.

The adapter cannot be served by llama.cpp (its fused-expert LoRA tensor names are not convertible),
and the MMQ bundle has no M=1 kernels, so there is no KV-cached decoding. Every generated token
therefore costs a full-context forward inside a FIXED window whose size must be one the bundle was
compiled for. Dense M and grouped r=M*8 are both compiled at 2048 and 8192, so we pick the smallest
window the prompt fits: 2048 turns stay cheap, longer ones still work at 4x the cost.

Exposes just enough of /v1/chat/completions for `dsbench.agentic.loop` (non-streaming; run the
harness with DSBENCH_STREAM=0). Qwen emits tool calls as <tool_call>{...}</tool_call>, which we
parse into OpenAI `tool_calls`.

    ADAPTER=/path/to/final PORT=18080 python serve_adapter.py     # omit ADAPTER for the base model
"""

import json
import os
import re
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

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
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

MODEL_DIR = "/home/wdenejko/models/qwen3.6"
GGUF = "Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
ADAPTER = os.environ.get("ADAPTER", "")
PORT = int(os.environ.get("PORT", "18080"))
MAX_NEW = int(os.environ.get("MAX_NEW", "320"))
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

configure_compiled_gguf_dequantize()
configure_qwen35_flash_attention_2()
configure_qwen35_fla()

tok = AutoTokenizer.from_pretrained(MODEL_DIR, gguf_file=GGUF, local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
PAD, EOS = tok.pad_token_id, tok.eos_token_id

print("loading base ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, gguf_file=GGUF, gguf_mmap_policy="release", local_files_only=True,
    dtype=torch.bfloat16, device_map={"": "cuda:0"},
    attn_implementation=os.environ.get("QWEN35_ATTN_IMPL", "flash_attention_2"),
)
model.config.use_cache = True
model.config.output_router_logits = False
model.config.router_aux_loss_coef = 0.0
configure_fast_moe_ranking(model)
configure_qwen35_fused_norms(model)
lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj",
                    "in_proj_qkv", "in_proj_z", "out_proj", "experts"],
    r=4, lora_alpha=4, use_rslora=False,
)
register_fast_lora(lora, model)
register_fast_moe_lora(lora, model, expert_prior="qwen-learned")
model = get_peft_model(model, lora, autocast_adapter_dtype=False)
if ADAPTER:
    set_peft_model_state_dict(model, load_file(ADAPTER.rstrip("/") + "/adapter_model.safetensors"))
    print("adapter loaded:", ADAPTER, flush=True)
else:
    print("NO adapter: serving the base model (LoRA B=0)", flush=True)
model.eval()


class _StopOnToolCall(StoppingCriteria):
    """A complete <tool_call> is all the harness needs this turn; stop paying for more tokens."""

    def __init__(self, prompt_len: int):
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = tok.decode(input_ids[0][self.prompt_len:], skip_special_tokens=False)
        return "</tool_call>" in text


def generate(prompt_ids: list[int], max_new: int) -> str:
    """Greedy decode with a KV cache.

    Cached decoding presents M=prompt_len on prefill and M=1 per step, neither of which the MMQ
    bundle compiles. Both the dense and the MoE expert paths now fall back to the generic
    compiled-dequant route for uncompiled shapes, so this works at any context length -- and decode
    only dequantizes the top_k routed experts per layer, which is far less work than the
    full-context forward the fixed-window decoder needed for every single token.
    """
    ids = torch.tensor([prompt_ids], dtype=torch.int64, device="cuda:0")
    mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
            pad_token_id=PAD, use_cache=True,
            stopping_criteria=StoppingCriteriaList([_StopOnToolCall(len(prompt_ids))]),
        )
    return tok.decode(out[0][len(prompt_ids):], skip_special_tokens=True)


def to_openai(text: str) -> dict:
    """Split Qwen's raw completion into OpenAI content + tool_calls."""
    calls = []
    for raw in TOOL_RE.findall(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue  # a malformed call is an observation, not a server error
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:8], "type": "function",
            "function": {"name": obj.get("name", ""),
                         "arguments": json.dumps(obj.get("arguments", {}))},
        })
    content = TOOL_RE.sub("", text).strip()
    message = {"role": "assistant", "content": content or None}
    if calls:
        message["tool_calls"] = calls
    return {"index": 0, "message": message,
            "finish_reason": "tool_calls" if calls else "stop"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console readable
        pass

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            text = tok.apply_chat_template(
                body["messages"], tools=body.get("tools"), tokenize=False,
                add_generation_prompt=True,
            )
            ids = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
            out = generate(ids, min(int(body.get("max_tokens", MAX_NEW)), MAX_NEW))
            payload = {"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion",
                       "model": body.get("model", "qwen36-lora"), "choices": [to_openai(out)]}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:  # surface as 500 so the client's retry/backoff can see it
            self.send_error(500, str(exc)[:200])


print(f"serving on 0.0.0.0:{PORT} (adapter={'yes' if ADAPTER else 'no'})", flush=True)
HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
