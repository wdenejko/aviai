"""Unit tests for the breadth-source normalisers (pure; no network / no `datasets`).

These guard the things most likely to silently corrupt the pilot mixture: (1) the jupyter-agent
tool-call rewrite (dict `arguments` -> JSON string, ids added + linked), (2) the Tulu-3
clean-source licensing filter, and (3) GSM8K's registration (train split only) and its gold answer.
Everything here runs on synthetic rows, so it is CI-safe.
"""
from __future__ import annotations

import json

from dsbench.sftgen.breadth.acquire import approx_tokens
from dsbench.sftgen.breadth.sources import (
    SOURCES_BY_KEY,
    _norm_gsm8k,
    _norm_instruction_output,
    _norm_jupyter_agent,
    _norm_text_to_sql,
    _norm_trajectory,
    _swe_row_ok,
    _tulu_row_ok,
    _valid_messages,
    public_sources,
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


def test_tulu_filter_takes_exact_subsets_and_leaves_out_no_robots():
    for kept in ("ai2-adapt-dev/tulu_v3.9_aya_100k", "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
                 "ai2-adapt-dev/tulu_hard_coded_repeated_10"):
        assert _tulu_row_ok({"source": kept})
    # Human-written, but CC BY-NC 4.0: not for a redistributable model.
    assert not _tulu_row_ok({"source": "ai2-adapt-dev/no_robots_converted"})
    # GPT-4-written subsets, under their real names.
    for tainted in ("ai2-adapt-dev/tulu_v3.9_wildchat_100k",
                    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
                    "ai2-adapt-dev/tulu_v3.9_table_gpt_5k"):
        assert not _tulu_row_ok({"source": tainted})
    # A familiar word in a new or renamed subset is not enough.
    assert not _tulu_row_ok({"source": "ai2-adapt-dev/aya_gpt4_translated"})
    assert not _tulu_row_ok({"source": "ai2-adapt-dev/oasst1_converted_v2"})


def test_gsm8k_is_registered_as_public_replay_from_its_train_split():
    src = SOURCES_BY_KEY["gsm8k"]
    assert (src.hf_id, src.config, src.split) == ("openai/gsm8k", "main", "train")  # never test
    assert (src.bucket, src.licence, src.redistributable) == ("replay", "MIT", True)
    assert src in public_sources()


def _gold(answer: str) -> str:
    return _norm_gsm8k({"question": "q", "answer": answer})["meta"]["gold_answer"]


def test_gsm8k_strips_calculator_annotations_and_keeps_the_gold_number():
    # A synthetic row in GSM8K's format: steps with `<<...>>` annotations, then `#### <number>`.
    rec = _norm_gsm8k({
        "question": " A baker made 30 rolls and sold a third of them. How many are left? ",
        "answer": "She sold 30/3 = <<30/3=10>>10 rolls.\n"
                  "So 30-10 = <<30-10=20>>20 are left.\n#### 20",
    })
    assert rec is not None and rec["loss_mask_roles"] == ["assistant"]
    user, assistant = rec["messages"]
    assert user == {"role": "user",
                    "content": "A baker made 30 rolls and sold a third of them. How many are left?"}
    assert assistant["content"] == ("She sold 30/3 = 10 rolls.\nSo 30-10 = 20 are left.\n\n"
                                    "The answer is 20.")
    assert rec["meta"] == {"gold_answer": "20"}
    # Final answers are integers, some with thousands commas or a sign (79 and 3 of train's 7,473).
    assert _gold("x\n#### 1,080") == "1080"
    assert _gold("x\n#### -7") == "-7"


def test_gsm8k_drops_rows_without_one_final_line():
    assert _norm_gsm8k({"question": "q", "answer": "no final line"}) is None
    assert _norm_gsm8k({"question": "q", "answer": "#### 3\n#### 4"}) is None
    assert _norm_gsm8k({"question": "q", "answer": "x\n#### 4\nmore text"}) is None
    assert _norm_gsm8k({"question": " ", "answer": "x\n#### 4"}) is None


def test_swe_excludes_swebench_verified_repos():
    # a trajectory that names a SWE-bench-Verified repo is dropped (eval hygiene)
    def msg(c):
        return {"messages": [{"role": "user", "content": c}]}
    assert not _swe_row_ok(msg("fix bug in django/django #123"))
    assert not _swe_row_ok(msg("patch scikit-learn/scikit-learn regression"))
    assert _swe_row_ok(msg("fix acme/widget in file x.py"))   # unrelated repo is kept


def test_approx_tokens_counts_content_and_toolcall_args():
    rec = {"messages": [
        {"role": "user", "content": "x" * 35},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"arguments": "y" * 35}}]},
    ]}
    # 70 chars / 3.5 == 20
    assert approx_tokens(rec) == 20
