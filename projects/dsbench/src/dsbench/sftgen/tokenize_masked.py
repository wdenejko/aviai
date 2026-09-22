"""Tokenise SFT records into fixed-length blocks with ASSISTANT-ONLY loss labels.

Every pilot record carries `loss_mask_roles: ["assistant"]` (ADR-004 designed the data that way),
but the recipe's collator ignores it and computes loss over the whole packed block. Measured on the
Gate-1 pilot, 36.6% of the content is user/tool/system text, so over a third of the gradient was
spent learning to predict prompts the model never has to generate. That is the most likely mechanism
behind the Gate-1 finding that never-trained *general* data improved as much as the target data --
the run was partly fitting the rendering, not the behaviour.

This module emits `labels` alongside `input_ids`: real token ids on spans authored by a role in
`loss_mask_roles`, and -100 everywhere else.

Finding the spans: two obvious mechanisms do NOT work here. `return_assistant_tokens_mask` needs a
`{% generation %}` block, which this template lacks (it returns an all-zero mask). Prefix rendering
fails too, because the template appends an empty `<think></think>` block to whichever assistant
message is LAST in the list it is given -- so rendering messages[:i] is not a prefix of the full
render whenever an assistant turn is non-final (11.9% of pilot records).

Instead we parse the rendered text structurally. The template is ChatML, so assistant regions are
exactly `<|im_start|>assistant\n ... <|im_end|>`; we locate those spans in the rendered string and
map them onto tokens via the fast tokenizer's offset mapping. This is immune to the think-tag quirk,
handles tool calls and multi-turn, and the closing `<|im_end|>` is kept trainable so the model
learns to stop.

Runs on the box (dashi) because the tokenizer comes from the GGUF.
"""

from __future__ import annotations

import json
from typing import Any

# Sentinel understood by the HF loss: positions that must not contribute a gradient.
IGNORE = -100

# ChatML markers used by this model's template.
ASSISTANT_OPEN = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"


def render(tok: Any, messages: list[dict], tools: Any = None) -> str:
    """Render a conversation with the model's chat template (string form, not token ids)."""
    return tok.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False
    )


def encode(tok: Any, text: str) -> list[int]:
    """Encode already-rendered text; the template has emitted its own special tokens."""
    return [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]


def assistant_spans(
    text: str, marker: str = ASSISTANT_OPEN, end: str = TURN_END
) -> list[tuple[int, int]]:
    """Character spans of assistant-authored regions in a ChatML-rendered conversation.

    The closing `<|im_end|>` is included: stopping is part of what the model must learn.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start == -1:
            return spans
        content = start + len(marker)
        stop = text.find(end, content)
        if stop == -1:
            spans.append((content, len(text)))
            return spans
        spans.append((content, stop + len(end)))
        cursor = stop + len(end)


def masked_record(tok: Any, record: dict) -> tuple[list[int], list[int], bool]:
    """Return (input_ids, labels, exact) for one record.

    `exact` is False when no trainable span could be located; the caller should count those, since
    their labels fall back to "train on everything" rather than silently training on nothing.
    """
    messages = record["messages"]
    train_roles = set(record.get("loss_mask_roles") or ["assistant"])
    text = render(tok, messages, record.get("tools"))

    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(t) for t in encoded["input_ids"]]
    offsets = encoded["offset_mapping"]

    spans: list[tuple[int, int]] = []
    if "assistant" in train_roles:
        spans.extend(assistant_spans(text))
    if not spans:
        return ids, list(ids), False

    labels = [IGNORE] * len(ids)
    for position, (begin, finish) in enumerate(offsets):
        if begin == finish:  # zero-width (special) token carries no text
            continue
        for span_start, span_end in spans:
            if begin >= span_start and finish <= span_end:
                labels[position] = ids[position]
                break
    if all(value == IGNORE for value in labels):
        return ids, list(ids), False
    return ids, labels, True


def pack(
    tok: Any,
    records: list[dict],
    block: int,
    eos_id: int | None,
) -> tuple[list[list[int]], list[list[int]], dict]:
    """Pack records into fixed `block`-length streams of ids + assistant-masked labels.

    Blocks with no trainable token at all are dropped: their loss would be NaN, and they teach
    nothing. That can only happen for a block made entirely of prompt text.
    """
    id_stream: list[int] = []
    label_stream: list[int] = []
    stats = {"records": 0, "inexact": 0, "dropped_empty_blocks": 0}

    for record in records:
        ids, labels, exact = masked_record(tok, record)
        if not ids:
            continue
        stats["records"] += 1
        stats["inexact"] += 0 if exact else 1
        id_stream.extend(ids)
        label_stream.extend(labels)
        if eos_id is not None:
            # The EOS that ends an assistant turn is itself worth learning.
            id_stream.append(int(eos_id))
            label_stream.append(int(eos_id) if labels and labels[-1] != IGNORE else IGNORE)

    id_blocks, label_blocks = [], []
    for start in range(0, len(id_stream) - block + 1, block):
        chunk_ids = id_stream[start : start + block]
        chunk_labels = label_stream[start : start + block]
        if all(value == IGNORE for value in chunk_labels):
            stats["dropped_empty_blocks"] += 1
            continue
        id_blocks.append(chunk_ids)
        label_blocks.append(chunk_labels)

    trainable = sum(1 for b in label_blocks for v in b if v != IGNORE)
    total = sum(len(b) for b in label_blocks)
    stats["blocks"] = len(id_blocks)
    stats["trainable_token_pct"] = round(100.0 * trainable / total, 1) if total else 0.0
    return id_blocks, label_blocks, stats


def read_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
