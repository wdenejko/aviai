"""Probe the training tokenizer's chat template: how it renders reasoning, history and prompts.

    python template_probe.py OUT_DIR

CPU only (tokenizer from the GGUF's metadata). Writes `chat_template.jinja`, which is the test
fixture `tests/fixtures/qwen36_chat_template.jinja` byte for byte, and `probe.json`: renders of
multi-turn chats, tool loops, and the thinking-on and thinking-off generation prompts, plus how
the tokenizer splits the text around `<think>`.
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

out = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(
    "/home/wdenejko/models/qwen3.6", gguf_file="Qwen3.6-35B-A3B-APEX-I-Mini.gguf",
    local_files_only=True)
(out / "chat_template.jinja").write_text(tok.chat_template)

specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
            "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]
res = {"eos": [tok.eos_token, tok.eos_token_id], "pad": [tok.pad_token, tok.pad_token_id],
       "ids": {s: tok.convert_tokens_to_ids(s) for s in specials}}


def r(messages, **kw):
    return tok.apply_chat_template(messages, tokenize=False, **kw)


chat = [{"role": "system", "content": "S"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2", "reasoning_content": "R2"}]
baked = [{"role": "user", "content": "U1"},
         {"role": "assistant", "content": "<think>\nR1\n</think>\n\nA1"}]
empty = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]
tools = [{"type": "function", "function": {
    "name": "get_time", "description": "Current time in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
loop = [{"role": "user", "content": "U1"},
        {"role": "assistant", "content": "", "reasoning_content": "R1",
         "tool_calls": [{"type": "function", "function": {
             "name": "get_time", "arguments": {"city": "Oslo"}}}]},
        {"role": "tool", "content": "12:00"},
        {"role": "assistant", "content": "A2", "reasoning_content": "R2"}]
loop_no_reason = [loop[0], {**loop[1], "reasoning_content": ""}, loop[2], loop[3]]

res["renders"] = {
    "chat_full_think_on": r(chat, enable_thinking=True),
    "chat_full_default": r(chat),
    "chat_prompt_turn4_think_on": r(chat[:4], add_generation_prompt=True, enable_thinking=True),
    "chat_prompt_turn4_think_off": r(chat[:4], add_generation_prompt=True, enable_thinking=False),
    "chat_prompt_turn4_default": r(chat[:4], add_generation_prompt=True),
    "baked_full": r(baked, enable_thinking=True),
    "empty_full": r(empty, enable_thinking=True),
    "loop_full": r(loop, tools=tools, enable_thinking=True),
    "loop_prompt_turn1": r(loop[:1], tools=tools, add_generation_prompt=True, enable_thinking=True),
    "loop_prompt_turn3": r(loop[:3], tools=tools, add_generation_prompt=True, enable_thinking=True),
    "loop_no_reason_full": r(loop_no_reason, tools=tools, enable_thinking=True),
}

# Token boundaries around the opener: is "\n" after <think> its own token?
text = "<|im_start|>assistant\n<think>\nOkay, the user wants\n</think>\n\nA2<|im_end|>\n"
enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
res["tokens"] = [[i, text[a:b]] for i, (a, b) in zip(enc["input_ids"], enc["offset_mapping"],
                                                     strict=True)]
(out / "probe.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print("ok", len(tok.chat_template), "chars of template")
