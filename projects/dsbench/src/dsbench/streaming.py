"""Reassemble a streamed chat completion into one non-stream-shaped `choice` dict.

WHY stream: the fork's NON-stream endpoint runs a strict json::parse over the model's tool-call
arguments and 500s the whole request if the model emitted one malformed/truncated tool call deep
in a trajectory (empirically what kills ds_notam_classify ~step 15 — NOT a double-quote bug; a
controlled probe showed quotes and 4KB args parse fine). Streaming delivers the arguments as
deltas we concatenate ourselves, so a malformed call comes back to us as text we can turn into a
recoverable tool error instead of aborting the run. See reference-ornith-agentic-behavior.

Shared by the agentic loop and the acceptance battery (the BFCL tool-call tests need the same
recovery: a malformed call must score as a format failure, not crash the request).
"""

from __future__ import annotations

import json

import httpx


def reassemble_stream(r: httpx.Response) -> dict:
    content = ""
    finish = None
    usage: dict = {}
    timings: dict = {}
    calls: dict[int, dict] = {}  # index -> {id, name, arguments}
    for line in r.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            continue  # keep-alive / non-JSON line
        usage = ev.get("usage") or usage
        timings = ev.get("timings") or timings
        ch = (ev.get("choices") or [{}])[0]
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
        delta = ch.get("delta") or {}
        if delta.get("content"):
            content += delta["content"]
        for tc in (delta.get("tool_calls") or []):
            idx = tc.get("index", 0)
            slot = calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function", {}) or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            slot["arguments"] += fn.get("arguments") or ""
    tool_calls = [
        {"id": c["id"], "type": "function",
         "function": {"name": c["name"], "arguments": c["arguments"]}}
        for _, c in sorted(calls.items())
    ]
    return {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
            "finish_reason": finish, "usage": usage, "timings": timings}
