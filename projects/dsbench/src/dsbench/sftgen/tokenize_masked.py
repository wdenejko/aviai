"""Tokenise SFT records into fixed-length blocks with ASSISTANT-ONLY loss labels.

Two generations of the same job live here.

**Gate 1/2: `pack` and `masked_record`.** They are kept so those runs and their loss evals
(`eval_adapter_ab.py`) stay reproducible.

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

**ADR-004 Revision 2: `pack_thinking`, for the thinking-on retrain.** Four changes, each answering
a measured Gate-2 failure (reports/gate-evals/20260924-gate2-battery.md and its addendum):

1. **The label starts where the served prompt ends.** With thinking on, the prompt ends in
   `<|im_start|>assistant\\n<think>\\n`. The model never generates that opener, so it gets no label;
   Gate 2 labelled it in 27,690 turns, and its adapter opened blocks of its own. The span is found
   by rendering the conversation up to each trained turn with the generation prompt: the text the
   server sends. That text must be a prefix of the full rendering, as characters and as tokens, so
   what the model is trained to continue is, by construction, what serving asks it to continue.
   (Prefix rendering works here because it ends in the generation prompt; the failure above came
   from ending it in an assistant message.)
2. **Only turns after the last user query are trained, each with its reasoning.** The template
   drops the reasoning of earlier assistant turns. Labelled without it, they teach short answers
   with no thinking; Gate 2 trained 2,595 of them. A trained turn with empty reasoning rejects its
   row, because its label would teach closing the block at once.
3. **Rows are bin-packed whole** (first-fit decreasing) instead of cut from one stream. Gate 2's
   stream cut 2,158 turns at a block edge, and their ends trained without their prompts.
4. **Tool calls keep their arguments** (`tool_arguments_as_objects`). The records carry them as
   JSON strings, and the template renders parameters only from a mapping, so every real tool call
   Gate 2 trained on was an empty `<function=...>` block. The server parses the string first;
   training now does the same. A boolean or null argument is passed as its JSON text, which the
   model wrote; the template would print Python's `True` or `None`.
5. **Each block records where its rows end** (`stats["block_seq_lens"]`): every row with its
   separator, then the padding. Packed without them, a row attends to the rows before it in its
   block, and the GatedDeltaNet state and convolution carry over from one row into the next.
   `build_masked_dataset` writes them as `seq_lens`, and the recipe's collator turns them into row
   boundaries (patches/recipe-train-packed-rows.patch).

**The tokenizer.** One more fix turned up while building this path. transformers' GGUF converter
registers only a few ChatML control tokens as added tokens, so `<think>`, `</think>`,
`<tool_call>` and the other USER_DEFINED tokens came out as BPE pieces (`<think>` = `<th` `ink`
`>`). llama.cpp keeps them whole. On the reasoning pilot's 200 prompts, HF counted exactly 2 tokens
more than llama-server on every one, and 0 after `match_llama_tokenization`
(reports/gate-evals/20260929-thinking-rendering-packing.md). Gate 1 and Gate 2 trained on the split
form; the legacy path keeps it on purpose, so those runs stay reproducible.

Runs on the box (dashi) because the tokenizer comes from the GGUF.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

# Sentinel understood by the HF loss: positions that must not contribute a gradient.
IGNORE = -100

# ChatML markers used by this model's template.
ASSISTANT_OPEN = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"
# With thinking on, the served prompt ends in ASSISTANT_OPEN + THINK_OPEN.
THINK_OPEN = "<think>\n"
THINK_CLOSE = "</think>"
TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE = "<tool_response>", "</tool_response>"

# llama.cpp's token types (the GGUF's `tokenizer.ggml.token_type`): 3 = CONTROL, 4 = USER_DEFINED.
# Its tokenizer never breaks a token of either type into pieces: USER_DEFINED always, CONTROL
# whenever it parses special tokens, which the server does for every prompt.
LLAMA_ATOMIC_TYPES = (3, 4)


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


# ---- Tokenizer parity with llama.cpp ----


def atomic_tokens(names: list[str], types: list[int]) -> list[str]:
    """The vocabulary entries llama.cpp keeps whole: its CONTROL and USER_DEFINED tokens."""
    return [name for name, kind in zip(names, types, strict=True) if kind in LLAMA_ATOMIC_TYPES]


def gguf_atomic_tokens(gguf_path: str) -> list[str]:
    """`atomic_tokens` read from a GGUF's own vocabulary (the `gguf` package is on the box)."""
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    names_field = reader.fields["tokenizer.ggml.tokens"]
    types_field = reader.fields["tokenizer.ggml.token_type"]
    names = [bytes(names_field.parts[i]).decode("utf-8") for i in names_field.data]
    types = [int(types_field.parts[i][0]) for i in types_field.data]
    return atomic_tokens(names, types)


def match_llama_tokenization(tok: Any, atomic: list[str]) -> None:
    """Make `tok` keep llama.cpp's atomic tokens whole, with the ids they already have.

    Each of them is already in the vocabulary, so registering it as an added token changes how
    text is split, not which ids exist. A vocabulary that grows means a mismatched tokenizer.
    """
    from tokenizers import AddedToken

    size = len(tok)
    tok.add_tokens([AddedToken(token, normalized=False, special=True) for token in atomic])
    if len(tok) != size:
        raise ValueError(f"atomic tokens added {len(tok) - size} new ids: wrong vocabulary?")


# ---- ADR-004 Revision 2: thinking-on rows ----


class RowRejected(ValueError):
    """A record that must not become training text. `reason` is counted in the build report."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts: only the text ones matter here
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def last_query_index(messages: list[dict]) -> int:
    """Index of the last real user query, by the chat template's own rule.

    A user message that only wraps tool output (`<tool_response>...</tool_response>`) is not a
    query. The template renders an assistant turn with its reasoning exactly when the turn comes
    after this index; with no query at all it falls back to the last message.
    """
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") != "user":
            continue
        content = _content_text(messages[index].get("content")).strip()
        if not (content.startswith(TOOL_RESPONSE_OPEN) and content.endswith(TOOL_RESPONSE_CLOSE)):
            return index
    return len(messages) - 1


def tool_arguments_as_objects(messages: list[dict]) -> list[dict]:
    """The messages with every tool call's `arguments` as a JSON object, as the server sends them.

    The OpenAI format, which every generator here records, carries `arguments` as a JSON string.
    This template renders a call's parameters only from a mapping (`tool_call.arguments is
    mapping`); a string renders as `<function=run_sql>\\n</function>`, a call with no arguments.
    llama-server parses the string into an object before it applies the template
    (`func_args_not_string` in llama.cpp's common/chat.cpp), so serving shows the model its
    parameters. The HF tokenizer doesn't, so the training text has to do it here. Gate 2 didn't:
    all 6,061 real tool calls in its training tokens rendered empty (2026-10-01).

    A call whose arguments aren't a JSON object rejects the row: it could not have been served.
    Each argument's value is passed as the model wrote it (`_as_written`).
    """
    out = []
    for message in messages:
        calls = message.get("tool_calls")
        if not calls:
            out.append(message)
            continue
        fixed = []
        for call in calls:
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    raise RowRejected("tool_arguments_not_json") from None
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise RowRejected("tool_arguments_not_an_object")
            function["arguments"] = {name: _as_written(value) for name, value in arguments.items()}
            fixed.append({**call, "function": function})
        out.append({**message, "tool_calls": fixed})
    return out


def _as_written(value: Any) -> Any:
    """A top-level tool-call argument, spelled for the template as the model wrote it.

    The template prints an argument with `tojson` when it is a mapping or a list, and with Jinja's
    `string` filter otherwise. That filter is Python's `str`, which spells a boolean `True` and a
    null `None`. llama-server's grammar held the model to the parameter's JSON schema, so it wrote
    `true`, `false` or `null`, and that text is what the label must hold. Found 2026-10-01 in the
    Target C pilot, where the base ends runs with `finish(answer=true)`; 30 of the tool rows' 295
    fit prompts call a tool with a boolean parameter. Numbers print the same both ways, and nested
    values go through `tojson`, which already writes JSON.
    """
    if isinstance(value, bool):  # before anything numeric: True == 1 in Python
        return "true" if value else "false"
    if value is None:
        return "null"
    return value


def render_thinking(
    tok: Any, messages: list[dict], tools: Any = None, add_generation_prompt: bool = False
) -> str:
    """Render as the server does with thinking on (`enable_thinking` is a template variable)."""
    return tok.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
    )


def thinking_record(tok: Any, record: dict) -> tuple[list[int], list[int]]:
    """`input_ids` and `labels` for one thinking-on row, or `RowRejected`.

    Labelled: every assistant turn after the last user query, from the end of the served prompt
    (`...<|im_start|>assistant\\n<think>\\n`) through its `<|im_end|>`. That covers the reasoning,
    `</think>`, the answer or tool call, and the stop token.

    A turn marked `"loss": False` stays in the row as context and is not labelled: ADR-004
    decision 7's Target C turns whose call failed (`assemble_rev2.py`). It is still checked like
    the others, so a row is rendered the same either way.
    """
    messages, tools = tool_arguments_as_objects(record["messages"]), record.get("tools")
    text = render_thinking(tok, messages, tools)
    query = last_query_index(messages)
    turns = [i for i in range(query + 1, len(messages)) if messages[i].get("role") == "assistant"]
    if not turns:
        raise RowRejected("no_turn_after_last_query")

    spans: list[tuple[int, int]] = []
    for i in turns:
        prompt = render_thinking(tok, messages[:i], tools, add_generation_prompt=True)
        if not prompt.endswith(ASSISTANT_OPEN + THINK_OPEN):
            raise RowRejected("prompt_lacks_think_opener")
        if not text.startswith(prompt):
            raise RowRejected("prompt_not_a_prefix")
        end = text.find(TURN_END, len(prompt))
        if end == -1:
            raise RowRejected("turn_not_closed")
        reasoning, closed, _ = text[len(prompt) : end].partition(THINK_CLOSE)
        if not closed:
            raise RowRejected("reasoning_not_closed")
        if not reasoning.strip():
            raise RowRejected("empty_reasoning")
        if messages[i].get("loss") is not False:
            spans.append((len(prompt), end + len(TURN_END)))
    if not spans:
        raise RowRejected("no_labelled_turn")

    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(t) for t in encoded["input_ids"]]
    # Character offset -> the token that begins / ends there. Zero-width tokens carry no text;
    # a character split over several byte tokens maps to its first (begin) and last (end) one.
    begins: dict[int, int] = {}
    ends: dict[int, int] = {}
    for position, (begin, finish) in enumerate(encoded["offset_mapping"]):
        if begin != finish:
            begins.setdefault(begin, position)
            ends[finish] = position

    labels = [IGNORE] * len(ids)
    for start, stop in spans:
        first, last = begins.get(start), ends.get(stop)
        if first is None or last is None:
            raise RowRejected("label_edge_splits_a_token")
        # The served prompt must tokenize to exactly the tokens this row puts before the label.
        if encode(tok, text[:start]) != ids[:first]:
            raise RowRejected("prompt_tokens_differ")
        labels[first : last + 1] = ids[first : last + 1]
    return ids, labels


def bin_pack(lengths: list[int], capacity: int) -> tuple[list[list[int]], list[int]]:
    """First-fit decreasing: each item, longest first, goes into the first bin with room.

    Returns the bins (item indices in placement order) and the items longer than any bin can hold.
    Deterministic: equal lengths keep their input order.
    """
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    bins: list[list[int]] = []
    room: list[int] = []
    too_long: list[int] = []
    for i in order:
        if lengths[i] > capacity:
            too_long.append(i)
            continue
        for b, free in enumerate(room):
            if lengths[i] <= free:
                bins[b].append(i)
                room[b] -= lengths[i]
                break
        else:
            bins.append([i])
            room.append(capacity - lengths[i])
    return bins, sorted(too_long)


def pack_thinking(
    tok: Any, records: list[dict], block: int, sep_id: int, pad_id: int
) -> tuple[list[list[int]], list[list[int]], dict]:
    """Revision 2 blocks: whole thinking-on rows, bin-packed, each followed by `sep_id`, padded.

    Every block is exactly `block` tokens, because the training kernels are keyed on exact token
    counts. The separator and the padding get no label. A row must leave room for its separator,
    so the longest row that fits is `block - 1` tokens.

    `stats["block_seq_lens"]` holds each block's segments in order: each row's length plus its
    separator, then the padding, if any. They sum to `block`.
    """
    rows: list[tuple[list[int], list[int]]] = []
    names: list[str] = []
    rejected: Counter[str] = Counter()
    for n, record in enumerate(records):
        try:
            rows.append(thinking_record(tok, record))
        except RowRejected as err:
            rejected[err.reason] += 1
            continue
        names.append(str((record.get("meta") or {}).get("id") or f"#{n}"))

    bins, too_long = bin_pack([len(ids) + 1 for ids, _ in rows], block)
    id_blocks: list[list[int]] = []
    label_blocks: list[list[int]] = []
    seq_lens: list[list[int]] = []
    for members in bins:
        ids: list[int] = []
        labels: list[int] = []
        for i in members:
            ids += rows[i][0] + [sep_id]
            labels += rows[i][1] + [IGNORE]
        padding = block - len(ids)
        id_blocks.append(ids + [pad_id] * padding)
        label_blocks.append(labels + [IGNORE] * padding)
        seq_lens.append([len(rows[i][0]) + 1 for i in members] + ([padding] if padding else []))

    total = block * len(id_blocks)
    content = sum(len(rows[i][0]) + 1 for members in bins for i in members)
    trained = sum(value != IGNORE for labels in label_blocks for value in labels)
    stats = {
        "records": len(records),
        "rows": len(rows),
        "rejected": dict(sorted(rejected.items())),
        "too_long": [names[i] for i in too_long],
        "block": block,
        "blocks": len(id_blocks),
        "rows_packed": sum(len(members) for members in bins),
        "rows_per_block_max": max((len(members) for members in bins), default=0),
        "fill_pct": round(100.0 * content / total, 1) if total else 0.0,
        "trainable_token_pct": round(100.0 * trained / total, 1) if total else 0.0,
        "block_rows": [[names[i] for i in members] for members in bins],
        "block_seq_lens": seq_lens,
    }
    return id_blocks, label_blocks, stats
