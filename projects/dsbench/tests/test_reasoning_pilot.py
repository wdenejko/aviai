"""Tests for the reasoning pilot's own logic: item selection, parsing, the length summary.

Generation (a llama-server) and verification (the ClickHouse sandbox) run against real services;
what is ours, and tested here, is everything around them.
"""

from __future__ import annotations

import pytest
from dsbench.sftgen import reasoning_pilot as rp


def _row(pool, rid, user="q?", system=None, assistant="a", dialect=None, truth=None, tools=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    meta = {"mix_pool": pool}
    if rid:
        meta["id"] = rid
    if dialect:
        meta.update(dialect=dialect, verification={"truth": truth})
    row = {"messages": messages, "meta": meta}
    if tools:
        row["tools"] = tools
    return row


def _target_a(dialect, i):
    return _row("targetA", f"A-weekday-numbering-{dialect}-payments-{i}", user=f"q{i}",
                system="SQL for ClickHouse. Answer with a single SQL query in a ```sql code block, "
                       "then the numeric result.",
                assistant="```sql\nSELECT count(*) FROM payments\n```\n\nAnswer: 7",
                dialect=dialect, truth=7)


@pytest.fixture
def mixture():
    rows = [_target_a(d, i) for d in rp.TARGET_A_DIALECTS for i in range(40)]
    for pool, n in rp.POOLS.items():
        if pool != "targetA":
            rows += [_row(pool, None, user=f"{pool} prompt {i}") for i in range(n + 5)]
    rows.append(_row("tulu3", "multi", user="x", tools=[{"type": "function"}]))  # never picked
    return rows


def test_items_follow_the_pool_plan_and_drop_the_numeric_answer(mixture):
    items = rp.build_items(mixture)
    counts = {pool: sum(i["pool"] == pool for i in items) for pool in rp.POOLS}
    assert counts == rp.POOLS
    dialects = [i["verify"]["dialect"] for i in items if i["pool"] == "targetA"]
    assert {d: dialects.count(d) for d in rp.TARGET_A_DIALECTS} == rp.TARGET_A_DIALECTS
    target_a = next(i for i in items if i["pool"] == "targetA")
    assert target_a["messages"][0]["content"].endswith("```sql code block.")
    assert "numeric result" not in target_a["messages"][0]["content"]
    assert target_a["verify"]["gold_sql"] == "SELECT count(*) FROM payments"
    assert all(m["role"] != "assistant" for i in items for m in i["messages"])
    assert len({i["id"] for i in items}) == len(items)


def test_items_are_the_same_every_run(mixture):
    assert rp.build_items(mixture) == rp.build_items(list(reversed(mixture)))


def test_last_sql_takes_the_final_block():
    assert rp.last_sql("draft ```sql\nSELECT 1\n``` fixed: ```SQL\nSELECT 2\n```") == "SELECT 2"
    assert rp.last_sql("no code here") is None


def test_split_reasoning_prefers_the_server_field_and_falls_back_to_the_tag():
    assert rp.split_reasoning({"reasoning_content": " think ", "content": " ans "}) == (
        "think", "ans")
    assert rp.split_reasoning({"content": "<think>t</think>\n\nans"}) == ("t", "ans")
    assert rp.split_reasoning({"content": "ans"}) == ("", "ans")


def test_request_body_turns_thinking_on_with_a_stable_seed():
    item = {"id": "tulu3:abc", "messages": [{"role": "user", "content": "hi"}]}
    body = rp.request_body(item, 16384)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["temperature"] == 0.6 and body["max_tokens"] == 16384
    assert body["seed"] == rp.request_body(item, 1)["seed"]
    assert "stream" not in body and "tools" not in body


def test_a_tool_item_sends_its_tools_and_streams():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    item = {"id": "tool_fit:0001", "messages": [{"role": "user", "content": "hi"}],
            "tools": tools}
    body = rp.request_body(item, 8192)
    assert body["tools"] == tools and body["stream"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_target_a_id_parsing_keeps_hyphenated_families():
    assert rp.parse_target_a_id("A-timezone-direction-clickhouse-gym_checkins-618807") == (
        "clickhouse", "gym_checkins", 618807)


def test_summary_counts_fits_limits_and_yield():
    def gen(pool, prompt, completion, reasoning, finish="stop", t0=0.0, t1=10.0):
        return {"id": f"{pool}:{prompt}:{completion}", "pool": pool, "error": "",
                "prompt_tokens": prompt, "completion_tokens": completion,
                "reasoning_tokens": reasoning, "answer_tokens": completion - reasoning,
                "finish_reason": finish, "decode_tok_s": 12.0, "started": t0, "finished": t1}
    gens = [gen("tulu3", 100, 1000, 900), gen("tulu3", 100, 5000, 4900),
            gen("tulu3", 100, 16384, 16384, finish="length", t1=20.0),
            {"id": "x", "pool": "tulu3", "error": "boom"}]
    verified = [{"status": "verified"}, {"status": "wrong"}, {"status": "rebuild_mismatch"}]
    s = rp.summarize(gens, verified)
    tulu = s["pools"]["tulu3"]
    assert tulu["n"] == 3 and tulu["hit_limit"] == 1 and s["errors"] == 1
    assert (tulu["fits_4096"], tulu["fits_8192"]) == (1, 2)  # 1101, 5101 and 16485 tokens
    assert s["generation"]["tokens_per_second"] == round(22384 / 20.0, 1)
    assert s["target_a_clickhouse"] == {"counts": {"verified": 1, "wrong": 1,
                                                   "rebuild_mismatch": 1},
                                        "judged": 2, "yield": 0.5}
