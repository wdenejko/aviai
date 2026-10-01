"""Tests for the thinking-on tokenisation of ADR-004 Revision 2: labels, history, packing.

The chat template is the model's own, byte for byte (`fixtures/qwen36_chat_template.jinja`, read
from the training GGUF on 2026-09-29), rendered with jinja2 the way transformers renders it. The
tokenizer is a stand-in that splits text the way the served one does where it matters here:
special tokens stay whole, and newlines are tokens of their own. What the real tokenizer does on
real rows was checked on the box (reports/gate-evals/20260929-thinking-rendering-packing.md).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from dsbench.sftgen import tokenize_masked as tm
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE = (Path(__file__).parent / "fixtures" / "qwen36_chat_template.jinja").read_text()
SPECIAL = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
           "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")
TOOLS = [{"type": "function", "function": {
    "name": "get_time", "description": "Current time in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                      sort_keys=sort_keys)


def _raise(message):
    raise ValueError(message)


class TemplateTokenizer:
    """The real template, and a toy tokenizer with offsets.

    `glue_newlines` joins a newline to the word after it: a tokenizer whose token straddles the
    edge between the served prompt and the label.
    """

    def __init__(self, glue_newlines: bool = False):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        self.template = env.from_string(TEMPLATE)
        word = r"\n\w+|\w+" if glue_newlines else r"\w+"
        special = "|".join(re.escape(s) for s in SPECIAL)
        self.pattern = re.compile(rf"{special}|{word}|\n+|[^\S\n]+|[^\w\s]")
        self.vocab: dict[str, int] = {}

    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=False, **kwargs):
        assert not tokenize
        return self.template.render(messages=messages, tools=tools,
                                    add_generation_prompt=add_generation_prompt, **kwargs)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        found = [(m.group(), m.start(), m.end()) for m in self.pattern.finditer(text)]
        assert sum(end - start for _, start, end in found) == len(text)
        out = {"input_ids": [self.vocab.setdefault(p, len(self.vocab)) for p, _, _ in found]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(start, end) for _, start, end in found]
        return out

    def convert_tokens_to_ids(self, token):
        return self.vocab.setdefault(token, len(self.vocab))

    def text(self, ids):
        pieces = {i: p for p, i in self.vocab.items()}
        return "".join(pieces[i] for i in ids)


def _labelled(tok, ids, labels):
    return tok.text([i for i, lab in zip(ids, labels, strict=True) if lab != tm.IGNORE])


def _prompt(tok, ids, labels):
    """The unlabelled tokens before the first labelled one."""
    first = next(k for k, lab in enumerate(labels) if lab != tm.IGNORE)
    return tok.text(ids[:first])


def _chat(reasoning="Two plus two is four.", answer="4", rid=None):
    record = {"messages": [{"role": "system", "content": "Be brief."},
                           {"role": "user", "content": "What is 2+2?"},
                           {"role": "assistant", "content": answer,
                            "reasoning_content": reasoning}]}
    if rid:
        record["meta"] = {"id": rid}
    return record


def _loop(first_reasoning="I need the time in Oslo."):
    return {"tools": TOOLS, "messages": [
        {"role": "user", "content": "What time is it in Oslo?"},
        {"role": "assistant", "content": "", "reasoning_content": first_reasoning,
         "tool_calls": [{"type": "function",
                         "function": {"name": "get_time", "arguments": {"city": "Oslo"}}}]},
        {"role": "tool", "content": "12:00"},
        {"role": "assistant", "content": "It is noon.",
         "reasoning_content": "The tool says 12:00."},
    ]}


def test_the_fixture_renders_as_the_training_tokenizer_does():
    # Expected strings: transformers' render of the same messages on the box (template probe).
    tok = TemplateTokenizer()
    chat = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "U2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"}]
    assert tm.render_thinking(tok, chat) == (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU1<|im_end|>\n"
        "<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nU2<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nR2\n</think>\n\nA2<|im_end|>\n")
    assert tm.render_thinking(tok, chat[:4], add_generation_prompt=True).endswith(
        "<|im_start|>user\nU2<|im_end|>\n<|im_start|>assistant\n<think>\n")


def test_the_label_starts_where_the_served_prompt_ends_and_stops_on_im_end():
    tok = TemplateTokenizer()
    ids, labels = tm.thinking_record(tok, _chat())
    assert _prompt(tok, ids, labels).endswith("<|im_start|>assistant\n<think>\n")
    assert _labelled(tok, ids, labels) == "Two plus two is four.\n</think>\n\n4<|im_end|>"
    assert labels[-1] == tm.IGNORE  # the template's newline after <|im_end|> is not generated


def test_turns_before_the_last_query_are_context_only():
    tok = TemplateTokenizer()
    record = {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "Greet back."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "Simple sum."}]}
    ids, labels = tm.thinking_record(tok, record)
    assert _labelled(tok, ids, labels) == "Simple sum.\n</think>\n\n4<|im_end|>"
    assert "Greet back." not in tok.text(ids)  # the template drops history reasoning


def test_a_tool_loop_trains_every_turn_after_the_query_with_its_reasoning():
    tok = TemplateTokenizer()
    ids, labels = tm.thinking_record(tok, _loop())
    trained = _labelled(tok, ids, labels)
    assert trained == (
        "I need the time in Oslo.\n</think>\n\n<tool_call>\n<function=get_time>\n"
        "<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call><|im_end|>"
        "The tool says 12:00.\n</think>\n\nIt is noon.<|im_end|>")
    assert "12:00\n</tool_response>" not in trained


def _openai_loop(arguments='{"city": "Oslo"}'):
    """`_loop` as the generators record it: OpenAI format, `arguments` a JSON string."""
    record = _loop()
    call = record["messages"][1]["tool_calls"][0]
    call["function"] = {**call["function"], "arguments": arguments}
    return record


def test_a_tool_call_recorded_as_a_json_string_trains_with_its_arguments():
    # The template renders parameters only from a mapping. Rendered as recorded, this call is
    # `<function=get_time>\n</function>`, as all of Gate 2's tool calls were.
    tok = TemplateTokenizer()
    record = _openai_loop()
    raw = tm.render_thinking(tok, record["messages"], TOOLS)
    assert "<function=get_time>\n</function>" in raw and "<parameter=city>" not in raw
    assert tm.thinking_record(tok, record) == tm.thinking_record(tok, _loop())
    assert "<parameter=city>\nOslo\n</parameter>" in _labelled(tok, *tm.thinking_record(
        tok, record))
    # the record itself is left as it was
    assert record["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Oslo"}'


def test_a_boolean_or_null_argument_trains_as_the_json_the_model_wrote():
    # The template prints a top-level argument with Jinja's `string` filter, which is Python's
    # `str`: `True`, `None`. Served, the grammar held the model to JSON, so it wrote `true`.
    tok = TemplateTokenizer()
    record = _openai_loop('{"city": "Oslo", "dst": true, "zone": null, "offset": 1, '
                          '"opts": {"exact": false}}')
    trained = _labelled(tok, *tm.thinking_record(tok, record))
    assert "<parameter=dst>\ntrue\n</parameter>" in trained
    assert "<parameter=zone>\nnull\n</parameter>" in trained
    assert "<parameter=offset>\n1\n</parameter>" in trained  # 1 == True in Python; not here
    assert '<parameter=opts>\n{"exact": false}\n</parameter>' in trained  # tojson: JSON already
    assert "True" not in trained and "None" not in trained


def test_reasoning_baked_into_the_content_trains_the_same():
    # render.py's rows carry `<think>...</think>` inside the content; the template splits it out.
    tok = TemplateTokenizer()
    baked = _chat(reasoning=None, answer="<think>\nTwo plus two is four.\n</think>\n\n4")
    del baked["messages"][-1]["reasoning_content"]
    assert tm.thinking_record(tok, baked) == tm.thinking_record(tok, _chat())


@pytest.mark.parametrize(("record", "reason"), [
    (_chat(reasoning=""), "empty_reasoning"),
    (_chat(reasoning="   "), "empty_reasoning"),
    (_loop(first_reasoning=""), "empty_reasoning"),
    ({"messages": [{"role": "user", "content": "Hi"}]}, "no_turn_after_last_query"),
    # A call the server couldn't have rendered either:
    (_openai_loop('{"city": "Os'), "tool_arguments_not_json"),
    (_openai_loop('["Oslo"]'), "tool_arguments_not_an_object"),
])
def test_rows_that_would_teach_the_wrong_thing_are_rejected(record, reason):
    with pytest.raises(tm.RowRejected) as err:
        tm.thinking_record(TemplateTokenizer(), record)
    assert err.value.reason == reason


def test_a_label_edge_inside_a_token_rejects_the_row():
    # "<think>" + "\nTwo" as one token: the served prompt would end mid-token.
    with pytest.raises(tm.RowRejected) as err:
        tm.thinking_record(TemplateTokenizer(glue_newlines=True), _chat())
    assert err.value.reason == "label_edge_splits_a_token"


def test_last_query_index_follows_the_template():
    messages = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "call"},
                {"role": "user", "content": "<tool_response>\n42\n</tool_response>"},
                {"role": "tool", "content": "43"}, {"role": "assistant", "content": "done"}]
    assert tm.last_query_index(messages) == 2
    assert tm.last_query_index([{"role": "assistant", "content": "x"}]) == 0


def test_atomic_tokens_are_the_control_and_user_defined_ones():
    names = ["a", "<|im_end|>", "<think>", "b", "<0x0A>"]
    assert tm.atomic_tokens(names, [1, 3, 4, 1, 6]) == ["<|im_end|>", "<think>"]


def test_bin_pack_is_first_fit_decreasing():
    bins, too_long = tm.bin_pack([5, 3, 7, 2, 11, 4], capacity=10)
    # 11 fits nowhere; 7 opens a bin, 5 another, 4 joins the 5, 3 fills the 7's, 2 opens a third.
    assert bins == [[2, 1], [0, 5], [3]]
    assert too_long == [4]


def test_pack_thinking_fills_exact_blocks_with_whole_rows():
    tok = TemplateTokenizer()
    records = [_chat(reasoning="step " * n, rid=f"r{n}") for n in (1, 5, 9, 13)]
    records.append(_chat(reasoning="", rid="empty"))
    records.append(_chat(reasoning="long " * 400, rid="huge"))
    sizes = {r["meta"]["id"]: len(tm.thinking_record(tok, r)[0]) for r in records[:4]}
    block = max(sizes.values()) * 2
    end = tok.convert_tokens_to_ids("<|endoftext|>")

    id_blocks, label_blocks, stats = tm.pack_thinking(tok, records, block, end, end)

    assert all(len(b) == block for b in id_blocks + label_blocks)
    assert stats["rejected"] == {"empty_reasoning": 1}
    assert stats["too_long"] == ["huge"]
    assert sorted(r for rows in stats["block_rows"] for r in rows) == sorted(sizes)
    for ids, labels, names in zip(id_blocks, label_blocks, stats["block_rows"], strict=True):
        cursor = 0
        for name in names:  # each row sits whole, then its unlabelled separator
            cursor += sizes[name]
            assert ids[cursor] == end and labels[cursor] == tm.IGNORE
            cursor += 1
        assert all(lab == tm.IGNORE for lab in labels[cursor:])  # padding trains nothing
    expected = sum(sum(lab != tm.IGNORE for lab in tm.thinking_record(tok, r)[1])
                   for r in records[:4])
    assert sum(lab != tm.IGNORE for b in label_blocks for lab in b) == expected
