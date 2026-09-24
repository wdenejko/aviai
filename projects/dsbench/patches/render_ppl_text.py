"""Render held-out eval records as plain role-labelled transcripts for `llama-perplexity`.

`llama-perplexity` tokenizes without parsing special tokens, so chat markup would be split into
ordinary characters. Plain role labels keep the text natural; it is for comparing export variants
on identical text, not for matching the training stack's chat-format loss.

    python3 render_ppl_text.py ~/gate2/eval_buckets_gate2.jsonl ~/gate2/ppl
"""
import json
import os
import sys

BUCKETS = ("targetA_heldout", "targetC_heldout", "general_heldout")


def render(message: dict) -> str:
    text = message.get("content") or ""
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        text += f"\n[call {function.get('name')}] {function.get('arguments')}"
    return text.strip()


def main() -> None:
    source, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    parts: dict[str, list[str]] = {bucket: [] for bucket in BUCKETS}
    for line in open(source):
        record = json.loads(line)
        if record["bucket"] not in parts:
            continue
        for message in record["messages"]:
            text = render(message)
            if text:
                parts[record["bucket"]].append(f"### {message['role']}\n{text}\n")
    for bucket, chunks in parts.items():
        with open(os.path.join(out_dir, f"{bucket}.txt"), "w") as handle:
            handle.write("\n".join(chunks))
        print(f"{bucket:16s} {sum(len(c) for c in chunks):8,} chars")


if __name__ == "__main__":
    main()
