"""Unit tests for the breadth-source normalisers (pure; no network / no `datasets`).

These guard the two things most likely to silently corrupt the pilot mixture: (1) the jupyter-agent
tool-call rewrite (dict `arguments` -> JSON string, ids added + linked), and (2) the Tulu-3
clean-source licensing filter. Everything here runs on synthetic rows, so it is CI-safe.
"""
from __future__ import annotations

import json

from dsbench.sftgen.breadth.acquire import approx_tokens
from dsbench.sftgen.breadth.sources import (
    _norm_instruction_output,
    _norm_jupyter_agent,
    _norm_text_to_sql,
    _norm_trajectory,
    _tulu_row_ok,
    _valid_messages,
)


def test_valid_messages():
    assert _valid_messages([{"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "yo"}])
    assert not _valid_messages([{"role": "user", "content": "hi"}])          # no assistant
    assert not _valid_messages([{"role": "assistant", "content": "yo"}])     # no user
    assert not _valid_messages("not a list")
    # an assistant turn with neither content nor tool_calls is not a usable target
    assert not _valid_messages([{"role": "user", "content": "hi"},
                                {"role": "assistant", "content": ""}])


def test_jupyter_agent_rewrites_dict_arguments():
    row = {
        "messages": [
            {"role": "user", "content": "count rows"},
            {"role": "assistant", "content": "loading",
             "tool_calls": [{"function": {"name": "run", "arguments": {"code": "df.shape"}}}]},
            {"role": "tool", "content": "(100, 3)"},
            {"role": "assistant", "content": "done",
             "tool_calls": [{"function": {"name": "final_answer",
                                          "arguments": {"answer": "100", "code": None}}}]},
        ],
        "tools": [{"type": "function", "function": {"name": "run"}}],
        "id": "nb_1", "edu_score": 5, "executor_type": "e2b",
    }
    rec = _norm_jupyter_agent(row)
    assert rec is not None
    assert rec["loss_mask_roles"] == ["assistant"]
    assert rec["tools"] == row["tools"]
    # every tool-call argument is now a JSON *string*, with id + type
    for m in rec["messages"]:
        for tc in m.get("tool_calls", []):
            assert tc["type"] == "function"
            assert isinstance(tc["function"]["arguments"], str)
            json.loads(tc["function"]["arguments"])  # parses
            assert tc["id"]
    # the tool response is linked to the preceding call id
    tool_msg = next(m for m in rec["messages"] if m["role"] == "tool")
    first_call = rec["messages"][1]["tool_calls"][0]
    assert tool_msg["tool_call_id"] == first_call["id"]
    assert rec["meta"]["edu_score"] == 5


def test_instruction_output_and_reject_empty():
    rec = _norm_instruction_output({"instruction": "Write add()", "output": "def add(a,b): ...",
                                    "seq_id": 1, "testcase": ["assert add(1,1)==2"]})
    assert rec is not None
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]
    assert rec["meta"]["has_testcase"] is True
    assert _norm_instruction_output({"instruction": "", "output": "x"}) is None


def test_trajectory_and_text_to_sql():
    traj = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert _norm_trajectory({"trajectory": traj, "db_id": "d"}) is not None
    assert _norm_trajectory({"trajectory": []}) is None
    rec = _norm_text_to_sql({"question": "how many?", "sql": "SELECT count(*) FROM t",
                             "external_knowledge": "t has rows"})
    assert rec is not None
    assert "```sql" in rec["messages"][-1]["content"]
    assert _norm_text_to_sql({"question": "q"}) is None   # no SQL -> drop


def test_tulu_clean_source_filter():
    assert _tulu_row_ok({"source": "ai2-adapt-dev/oasst1_converted"})
    assert _tulu_row_ok({"source": "ai2-adapt-dev/flan_v2_converted"})
    assert not _tulu_row_ok({"source": "ai2-adapt-dev/personahub_math"})   # GPT-4o teacher
    assert not _tulu_row_ok({"source": "ai2-adapt-dev/wildchat_gpt4"})     # GPT teacher
    assert not _tulu_row_ok({"source": ""})


def test_approx_tokens_counts_content_and_toolcall_args():
    rec = {"messages": [
        {"role": "user", "content": "x" * 35},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"arguments": "y" * 35}}]},
    ]}
    # 70 chars / 3.5 == 20
    assert approx_tokens(rec) == 20
