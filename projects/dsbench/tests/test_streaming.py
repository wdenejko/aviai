"""Tests for the stream reassembler shared by the agentic loop, the battery and Revision 2's
generation (`dsbench.streaming`)."""
from __future__ import annotations

import json

from dsbench.streaming import reassemble_stream


class _Stream:
    def __init__(self, events: list[dict]):
        self.lines = [f"data: {json.dumps(e)}" for e in events] + ["", "data: [DONE]"]

    def iter_lines(self):
        return iter(self.lines)


def _delta(**delta) -> dict:
    return {"choices": [{"delta": delta}]}


def test_the_reasoning_the_content_and_the_calls_all_come_back():
    events = [
        _delta(reasoning_content="The user wants "), _delta(reasoning_content="Oslo."),
        _delta(tool_calls=[{"index": 0, "id": "c0",
                            "function": {"name": "get_local_time", "arguments": '{"city": '}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": '"Oslo"}'}}]),
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 40, "completion_tokens": 12}},
    ]
    out = reassemble_stream(_Stream(events))
    msg = out["message"]
    assert msg["reasoning_content"] == "The user wants Oslo."
    assert msg["tool_calls"] == [{"id": "c0", "type": "function", "function": {
        "name": "get_local_time", "arguments": '{"city": "Oslo"}'}}]
    assert (out["finish_reason"], out["usage"]["completion_tokens"]) == ("tool_calls", 12)


def test_a_reply_without_reasoning_has_no_reasoning_field():
    out = reassemble_stream(_Stream([_delta(content="Hello."),
                                     {"choices": [{"delta": {}, "finish_reason": "stop"}]}]))
    assert out["message"] == {"role": "assistant", "content": "Hello.", "tool_calls": []}
