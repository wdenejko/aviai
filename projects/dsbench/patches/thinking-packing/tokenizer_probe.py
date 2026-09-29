"""Does the training tokenizer split text the way llama.cpp does? CPU only.

    python tokenizer_probe.py OUT_DIR

1. The GGUF's own token types: which tokens llama.cpp treats as atomic (CONTROL, USER_DEFINED).
2. The HF tokenizer built from the GGUF, as build_masked_dataset.py uses it, on sample strings.
3. The same tokenizer after registering the GGUF's atomic tokens as added tokens.
llama.cpp's own counts come from llama-server (validate_pilot_packing.py): llama-tokenize from
the qwen4exp build-v2 segfaults on this GGUF.
"""
import json
import sys
from pathlib import Path

from gguf import GGUFReader
from tokenizers import AddedToken
from transformers import AutoTokenizer

GGUF = "/home/wdenejko/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
out = Path(sys.argv[1])

reader = GGUFReader(GGUF)


def field(name):
    f = reader.fields[name]
    return [f.parts[i] for i in f.data]


tokens = [bytes(p).decode("utf-8", errors="replace") for p in field("tokenizer.ggml.tokens")]
types = [int(p[0]) for p in field("tokenizer.ggml.token_type")]
# llama.cpp token types: 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED, 5 UNUSED, 6 BYTE
atomic = [(i, tokens[i], types[i]) for i in range(len(tokens)) if types[i] in (3, 4)]

tok = AutoTokenizer.from_pretrained(str(Path(GGUF).parent), gguf_file=Path(GGUF).name,
                                    local_files_only=True)
samples = {
    "opener": "<|im_start|>assistant\n<think>\nOkay\n</think>\n\nA<|im_end|>\n",
    "tool": "<tool_call>\n<function=get_time>\n<parameter=city>\nOslo\n</parameter>\n"
            "</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n12:00\n"
            "</tool_response><|im_end|>\n",
}
res = {"n_vocab": len(tokens), "len_tok_before": len(tok),
       "atomic": [[i, t, ty] for i, t, ty in atomic],
       "hf_before": {k: tok(v, add_special_tokens=False)["input_ids"] for k, v in samples.items()}}

added = tok.add_tokens([AddedToken(t, normalized=False, special=True) for _, t, _ in atomic])
res["added_new"] = added
res["len_tok_after"] = len(tok)
res["hf_after"] = {k: tok(v, add_special_tokens=False)["input_ids"] for k, v in samples.items()}
res["think_id_after"] = tok.convert_tokens_to_ids("<think>")
(out / "tokenizer_probe.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
for k, v in samples.items():
    (out / f"sample_{k}.txt").write_text(v)
print("ok", len(atomic), "atomic tokens;", added, "new ids added")
