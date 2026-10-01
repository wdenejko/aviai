"""Measure Target C's pilot trajectories as the retrain will train them. CPU only; on the box.

    python measure_trajectories.py OUT.json TRAJECTORIES.jsonl [FAILED.jsonl ...]

Every trajectory, kept or not, goes through `thinking_record` (tokenize_masked.py, copied next to
this script, since dsbench is not installed on the box): the code that builds the retrain's
blocks, with the matched tokenizer, tool-call arguments rendered as objects, and every turn after
the user message labelled with its reasoning. Reported per trajectory: its tokens, its labelled
tokens, whether it fits an 8,192-token block, and why `thinking_record` rejects it, if it does.

Run in the ftgguf toolbox, whose torch needs a library the host lacks:
    toolbox run -c llama-rocm-unlimited-build bash -lc \\
        "cd DIR && ~/ftgguf/bin/python measure_trajectories.py ..." </dev/null
"""

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

from tokenize_masked import (
    IGNORE,
    RowRejected,
    encode,
    gguf_atomic_tokens,
    match_llama_tokenization,
    render_thinking,
    thinking_record,
    tool_arguments_as_objects,
)
from transformers import AutoTokenizer

GGUF = Path("/home/wdenejko/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf")
BLOCK = 8192

out = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(str(GGUF.parent), gguf_file=GGUF.name, local_files_only=True)
match_llama_tokenization(tok, gguf_atomic_tokens(str(GGUF)))

rows = []
for path in sys.argv[2:]:
    for line in open(path):
        rec = json.loads(line)
        meta, check = rec["meta"], rec["meta"]["verification"]
        row = {"id": meta["id"], "task": meta["task"], "passed": check["oracle_passed"],
               "status": check.get("status"), "steps": check["steps"],
               "assistant_turns": sum(m["role"] == "assistant" for m in rec["messages"]),
               "empty_reasoning_turns": sum(m["role"] == "assistant"
                                            and not (m.get("reasoning_content") or "").strip()
                                            for m in rec["messages"])}
        try:  # the rendered length, whether or not the row is trainable
            messages = tool_arguments_as_objects(rec["messages"])
            row["tokens"] = len(encode(tok, render_thinking(tok, messages, rec.get("tools"))))
        except RowRejected as err:
            row["tokens"] = None
            row["render_rejected"] = err.reason
        try:
            ids, labels = thinking_record(tok, rec)
            row["labelled"] = sum(label != IGNORE for label in labels)
            row["rejected"] = None
        except RowRejected as err:
            row["labelled"], row["rejected"] = None, err.reason
        row["kept"] = bool(row["passed"] and row["rejected"] is None
                           and row["tokens"] is not None and row["tokens"] <= BLOCK)
        rows.append(row)


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {"n": len(values), "median": median(values),
            "p90": values[min(len(values) - 1, int(0.9 * len(values)))], "max": values[-1]}


passed = [r for r in rows if r["passed"]]
summary = {
    "trajectories": len(rows), "oracle_passed": len(passed),
    "kept": sum(r["kept"] for r in rows),
    "passed_over_block": sum(r["tokens"] is not None and r["tokens"] > BLOCK for r in passed),
    "rejected_by_thinking_record": dict(Counter(r["rejected"] for r in rows if r["rejected"])),
    "tokens_passed": stats([r["tokens"] for r in passed]),
    "tokens_all": stats([r["tokens"] for r in rows]),
    "labelled_share_passed": (round(sum(r["labelled"] or 0 for r in passed)
                                    / max(1, sum(r["tokens"] or 0 for r in passed)), 3)),
    "by_task": {task: {"runs": sum(r["task"] == task for r in rows),
                       "passed": sum(r["task"] == task and r["passed"] for r in rows),
                       "kept": sum(r["task"] == task and r["kept"] for r in rows)}
                for task in sorted({r["task"] for r in rows})},
}
out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
print(json.dumps(summary, indent=1))
