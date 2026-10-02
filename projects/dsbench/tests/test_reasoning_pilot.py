"""Tests for the reasoning pilot's own logic: item selection, parsing, the length summary.

Generation (a llama-server) and verification (the ClickHouse sandbox) run against real services;
what is ours, and tested here, is everything around them.
"""

from __future__ import annotations

import json

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


# --- the prefilled path (sftgen/prefill.py) ------------------------------------------------------


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _Server:
    """/apply-template, /completion and /tokenize, answered as llama-server answers them."""

    def __init__(self, content, stop_type="eos", prompt="<|im_start|>assistant\n<think>\n"):
        self.content, self.stop_type, self.prompt, self.sent = content, stop_type, prompt, {}

    def post(self, path, json):
        self.sent[path] = json
        if path == "/apply-template":
            return _Response({"prompt": self.prompt})
        if path == "/completion":
            return _Response({"content": self.content, "stop_type": self.stop_type,
                              "tokens_evaluated": 30, "tokens_predicted": 12,
                              "timings": {"predicted_per_second": 20.0}})
        return _Response({"tokens": json["content"].split()})  # /tokenize: a token a word


PREFILLED = {"id": "targetA_start:A-x#1", "pool": "targetA_start",
             "messages": [{"role": "user", "content": "Sundays?"}],
             "prefill": "Sunday is 7 in ClickHouse."}


def test_a_prefilled_item_continues_its_own_thinking():
    server = _Server(" So `= 7`.\n</think>\n\n```sql\nSELECT 1\n```")
    record = rp.run_one(server, PREFILLED, 512)
    assert server.sent["/apply-template"]["chat_template_kwargs"] == {"enable_thinking": True}
    body = server.sent["/completion"]
    assert body["prompt"] == "<|im_start|>assistant\n<think>\nSunday is 7 in ClickHouse."
    assert body["preserved_tokens"] == ["<think>", "</think>"]  # or </think> prints as nothing
    assert body["n_predict"] == 512 and body["temperature"] == 0.6
    assert body["seed"] == rp.request_body(PREFILLED, 1)["seed"]  # as the chat path seeds it
    assert record["reasoning"] == "Sunday is 7 in ClickHouse. So `= 7`."  # the prefill included
    assert record["answer"] == "```sql\nSELECT 1\n```"
    assert record["finish_reason"] == "stop" and record["error"] == ""
    assert (record["prompt_tokens"], record["completion_tokens"]) == (30, 12)
    assert record["reasoning_tokens"] == 8  # the whole trace, as the training row holds it


def test_a_prefilled_reply_that_runs_out_is_all_reasoning_and_unfinished():
    record = rp.run_one(_Server(" and so on", stop_type="limit"), PREFILLED, 4)
    assert (record["reasoning"], record["answer"]) == ("Sunday is 7 in ClickHouse. and so on", "")
    assert record["finish_reason"] == "length"


def test_a_template_that_does_not_open_the_thinking_block_is_an_error():
    record = rp.run_one(_Server("x", prompt="<|im_start|>assistant\n"), PREFILLED, 4)
    assert "thinking block" in record["error"]


@pytest.mark.parametrize("stop_type, finish", [("eos", "stop"), ("word", "stop"),
                                               ("limit", "length"), ("none", "length"),
                                               (None, "length")])
def test_the_completion_stop_type_maps_to_the_chat_finish_reason(stop_type, finish):
    assert rp.finish_reason(stop_type) == finish


# --- volume runs: the block budget and whole rows (ADR-004 Revision 2) ---------------------------


class _ChatServer(_Server):
    """The chat path too: /v1/chat/completions answers one message, and a prompt is 1,000 words."""

    def __init__(self, fail_template=False):
        super().__init__("", prompt="w " * 1000)
        self.fail_template = fail_template

    def post(self, path, json):
        if path == "/apply-template" and self.fail_template:
            raise RuntimeError("no template here")
        if path == "/v1/chat/completions":
            self.sent[path] = json
            return _Response({"choices": [{"message": {"reasoning_content": "r",
                                                       "content": "a"},
                                           "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 1000, "completion_tokens": 9}})
        return super().post(path, json)


def test_a_reply_may_run_only_as_far_as_its_row_still_fits_the_block():
    tools = [{"type": "function", "function": {"name": "f"}}]
    item = {"id": "tool_fit:1", "pool": "tool_fit", "tools": tools,
            "messages": [{"role": "user", "content": "hi"}]}
    server = _ChatServer()
    assert rp.reply_budget(server, item, 16384, 8192) == (8192 - 1000 + rp.BLOCK_SLACK, 1000)
    # rendered as the server renders the request: thinking on, with the item's tools
    assert server.sent["/apply-template"] == {"messages": item["messages"], "tools": tools,
                                              "chat_template_kwargs": {"enable_thinking": True}}
    assert rp.reply_budget(server, item, 4096, 8192) == (4096, 1000)  # max_tokens still caps
    assert rp.reply_budget(server, item, 4096, 0) == (4096, None)  # no block, no count
    # a prefill is part of the row's text before the model's own
    prefilled = {**item, "prefill": "one two three"}
    assert rp.reply_budget(_ChatServer(), prefilled, 16384, 8192)[1] == 1003
    # a prompt that can't be counted keeps max_tokens: a wasted reply, never a lost row
    assert rp.reply_budget(_ChatServer(fail_template=True), item, 16384, 8192) == (16384, None)


def test_a_volume_reply_records_its_budget_and_a_prompt_over_the_block_runs_nothing():
    item = {"id": "oasst1:1", "pool": "oasst1", "messages": [{"role": "user", "content": "hi"}]}
    server = _ChatServer()
    record = rp.run_one(server, item, 16384, block=8192)
    assert server.sent["/v1/chat/completions"]["max_tokens"] == 7208
    assert (record["max_tokens"], record["rendered_prompt_tokens"]) == (7208, 1000)
    assert (record["answer"], record["finish_reason"]) == ("a", "stop")
    small = _ChatServer()
    record = rp.run_one(small, item, 16384, block=1000)
    assert record["finish_reason"] == "prompt_over_block" and record["error"] == ""
    assert "/v1/chat/completions" not in small.sent
    assert "max_tokens" not in rp.run_one(_ChatServer(), item, 16384)  # a pilot's row is as before


def test_a_resumed_run_skips_a_torn_row_and_appends_whole_ones(tmp_path, monkeypatch):
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps({"id": i, "pool": "p", "messages": []}) + "\n"
                             for i in ("a", "b", "c", "d")))
    out = tmp_path / "gen.jsonl"
    # a: answered; b: failed, so it runs again; c: torn at the end of the file (a power cut)
    out.write_text(json.dumps({"id": "a", "error": ""}) + "\n"
                   + json.dumps({"id": "b", "error": "boom"}) + "\n" + '{"id": "c", "rea')
    assert rp.answered(out) == {"a"}
    ran = []
    monkeypatch.setattr(rp, "run_one", lambda client, item, max_tokens, block: ran.append(
        (item["id"], max_tokens, block)) or {"id": item["id"], "error": ""})
    rp.generate(items, out, "http://127.0.0.1:9", workers=2, max_tokens=99, block=8192)
    assert sorted(ran) == [("b", 99, 8192), ("c", 99, 8192), ("d", 99, 8192)]
    lines = out.read_text().splitlines()
    assert lines[2] == '{"id": "c", "rea'  # the torn row is left as it was, on a line of its own
    assert rp.answered(out) == {"a", "b", "c", "d"}


def test_a_checker_reads_one_row_an_item_the_answered_one(tmp_path):
    gen = tmp_path / "gen.jsonl"
    gen.write_text("".join(json.dumps(row) + "\n" for row in [
        {"id": "a", "error": "ReadTimeout"}, {"id": "a", "error": "", "answer": "second"},
        {"id": "b", "error": "", "answer": "kept"}, {"id": "b", "error": "late"},
        {"id": "c", "error": "boom"}, {"id": "c", "error": "boom again"}]))
    rows = rp.latest_rows(gen)
    assert (rows["a"]["answer"], rows["b"]["answer"], rows["c"]["error"]) == (
        "second", "kept", "boom again")
    assert list(rows) == ["a", "b", "c"]  # in the order the items first appear
