"""Validate the Revision 2 tokenisation on the reasoning pilot's real traces. CPU only; on the box.

    python validate_pilot_packing.py PILOT_RUN_DIR OUT_DIR

Three checks, on the 200 prompts and replies of reports/gate-evals/20260928-reasoning-pilot.md:

1. **Parity with llama.cpp.** The pilot recorded each prompt's token count from llama-server's own
   usage block. The HF tokenizer as the GGUF converter builds it, and the same tokenizer after
   `match_llama_tokenization`, must match it: the fixed one exactly.
2. **Label invariants, on real ids.** Every trained span starts right after `<think>` `\\n` and
   ends on `<|im_end|>`, with one `</think>` inside it.
3. **Packing at 8192.** Rows kept, rejected and too long; blocks, fill and trainable share. Replies
   that hit the generation cap are left out first, as the mixture's assembly will leave them out.

Needs `tokenize_masked.py` next to it, since dsbench is not installed on the box.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

from tokenize_masked import (
    IGNORE,
    RowRejected,
    gguf_atomic_tokens,
    match_llama_tokenization,
    pack_thinking,
    render_thinking,
    thinking_record,
)
from transformers import AutoTokenizer

GGUF = Path("/home/wdenejko/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf")
BLOCK = 8192

run, out = Path(sys.argv[1]), Path(sys.argv[2])
items = {}
for line in open(run / "pilot_items.jsonl"):
    item = json.loads(line)
    items[item["id"]] = item
gens = [json.loads(line) for line in open(run / "pilot_gen.jsonl")]
gens = [g for g in gens if not g.get("error")]

tok = AutoTokenizer.from_pretrained(str(GGUF.parent), gguf_file=GGUF.name, local_files_only=True)


def prompt_tokens(item):
    text = render_thinking(tok, item["messages"], item.get("tools"), add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False)["input_ids"])


# 1. Parity with llama-server's prompt counts, before and after the fix.
before = Counter(prompt_tokens(items[g["id"]]) - g["prompt_tokens"] for g in gens)
atomic = gguf_atomic_tokens(str(GGUF))
match_llama_tokenization(tok, atomic)
after = Counter(prompt_tokens(items[g["id"]]) - g["prompt_tokens"] for g in gens)

# 2. Records as the mixture will carry them, and label invariants on real ids.
think, end_think, im_end = (tok.convert_tokens_to_ids(t)
                            for t in ("<think>", "</think>", "<|im_end|>"))
# A byte-level vocabulary spells "\n" as "Ċ", so ask the tokenizer instead of the vocabulary.
(newline,) = tok("\n", add_special_tokens=False)["input_ids"]
records, capped = [], []
for g in gens:
    if g["finish_reason"] == "length":
        capped.append(g["id"])
        continue
    item = items[g["id"]]
    records.append({"messages": item["messages"] + [
        {"role": "assistant", "content": g["answer"], "reasoning_content": g["reasoning"]}],
        "meta": {"id": g["id"], "pool": g["pool"]}})

# The same records as a mixture file, for an end-to-end run of build_masked_dataset.py.
(out / "pilot_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

violations: Counter = Counter()
row_minus_server: Counter = Counter()
lengths = []
for record, g in zip(records, [g for g in gens if g["finish_reason"] != "length"], strict=True):
    try:
        ids, labels = thinking_record(tok, record)
    except RowRejected:
        continue  # counted by pack_thinking below
    lengths.append(len(ids))
    trained = [k for k, lab in enumerate(labels) if lab != IGNORE]
    first, last = trained[0], trained[-1]
    if ids[first - 2 : first] != [think, newline]:
        violations["label_does_not_follow_think_opener"] += 1
    if ids[last] != im_end:
        violations["label_does_not_end_on_im_end"] += 1
    if trained != list(range(first, last + 1)):
        violations["label_not_contiguous"] += 1
    if ids[first : last + 1].count(end_think) != 1:
        violations["not_one_think_close_in_label"] += 1
    # Row vs what the server saw: prompt + generated tokens (+ the template's newline after
    # <|im_end|>). Not exact by nature: the model may sample a non-canonical split of its text.
    row_minus_server[len(ids) - g["prompt_tokens"] - g["completion_tokens"]] += 1

# 3. Packing.
end = tok.convert_tokens_to_ids("<|endoftext|>")
id_blocks, label_blocks, stats = pack_thinking(tok, records, BLOCK, end, end)
assert all(len(b) == BLOCK for b in id_blocks), "a block is not BLOCK tokens"

result = {
    "prompts": len(gens),
    "hf_minus_server_prompt_tokens": {"converter_tokenizer": dict(before),
                                      "matched_tokenizer": dict(after)},
    "atomic_tokens": len(atomic),
    "capped_replies_left_out": capped,
    "label_invariant_violations": dict(violations),
    "row_minus_server_tokens": dict(sorted(row_minus_server.items())),
    "row_tokens_median": median(lengths) if lengths else None,
    "packing": {k: v for k, v in stats.items() if k != "block_rows"},
    "rows_per_block": dict(sorted(Counter(len(rows) for rows in stats["block_rows"]).items())),
}
(out / "validate_pilot_packing.json").write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k != "packing"}, indent=1))
print(json.dumps(result["packing"], indent=1))
