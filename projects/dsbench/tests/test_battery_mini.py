"""Tests for the thinking-on mini-battery: its requests, its subsets and its comparison."""

from __future__ import annotations

import json

from dsbench.battery import extract, generate, mini
from dsbench.battery.generate import request_body, thinking_fields
from dsbench.battery.items import Item, load_items, save_items, write_jsonl


def _item(bench="humaneval_plus", i="0", **gen) -> Item:
    return Item(bench, i, [{"role": "user", "content": "q"}], gen, {"test": "SECRET"})


# --- requests ----------------------------------------------------------------------------------


def test_a_thinking_item_samples_with_one_seed_per_item_and_another_for_the_aa_pass():
    item = _item(thinking=True, max_tokens=12288)
    base, adapter, rep = (request_body(item, s, "m") for s in ("base", "adapter", "base_rep"))
    assert "SECRET" not in json.dumps(base)
    assert base["chat_template_kwargs"] == {"enable_thinking": True}
    assert (base["temperature"], base["top_p"], base["top_k"], base["min_p"]) == (0.6, 0.95, 20, 0)
    assert base["max_tokens"] == 12288
    # base and adapter draw the same random numbers; the A/A pass draws others
    assert base["seed"] == adapter["seed"] == request_body(item, "adapter_half", "m")["seed"]
    assert rep["seed"] == (base["seed"] + 1) % 2**31
    assert request_body(_item(i="1", thinking=True), "base", "m")["seed"] != base["seed"]
    # a battery item without the flag is unchanged: greedy, thinking off, seed 0
    plain = request_body(_item(max_tokens=2048), "base", "m")
    assert (plain["temperature"], plain["seed"], "top_k" in plain) == (0.0, 0, False)
    assert plain["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_reply_with_reasoning_and_nothing_after_it_never_closed():
    assert thinking_fields({"reasoning_content": "so", "content": ""})["unclosed"]
    assert thinking_fields({"reasoning_content": "so", "content": " \n"})["unclosed"]
    assert not thinking_fields({"reasoning_content": "so", "content": "def f(): ..."})["unclosed"]
    assert not thinking_fields({"reasoning_content": "so", "tool_calls": [{"id": 1}]})["unclosed"]
    assert not thinking_fields({"content": "an answer, no reasoning"})["unclosed"]


def test_a_thinking_row_keeps_its_reasoning(monkeypatch):
    reply = {"message": {"reasoning_content": "first, the edge cases", "content": "code"},
             "finish_reason": "stop", "usage": {"completion_tokens": 9}}
    monkeypatch.setattr(generate, "call", lambda client, body: reply)
    row = generate.run_item(None, _item(thinking=True), "base", "m", attempts=1)
    assert (row["reasoning"], row["unclosed"], row["reasoning_chars"]) == (
        "first, the edge cases", False, 21)
    assert "reasoning" not in generate.run_item(None, _item(), "base", "m", attempts=1)


def test_rows_are_appended_whole_one_write_each(tmp_path, monkeypatch):
    out = tmp_path / "gen.jsonl"
    out.write_text(json.dumps({"id": "0"}) + "\n")  # a pass resumed: earlier rows stay
    writes = []
    real_write = generate.os.write
    monkeypatch.setattr(generate.os, "write", lambda fd, data: writes.append(len(data))
                        or real_write(fd, data))
    fd = generate.os.open(out, generate.os.O_WRONLY | generate.os.O_APPEND)
    big = {"id": "1", "reasoning": "x" * 60_000}  # past any buffer a writer would split at
    generate.append_row(fd, big)
    generate.os.close(fd)
    assert len(writes) == 1
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ["0", "1"]


# --- subsets -----------------------------------------------------------------------------------


def _battery_items(tmp_path):
    d = tmp_path / "items"
    save_items(d / "ifeval.jsonl", [_item("ifeval", str(i), max_tokens=4096) for i in range(300)])
    save_items(d / "bfcl.jsonl", [
        Item("bfcl", f"{cat}_{i}", [{"role": "user", "content": "q"}],
             {"max_tokens": 1024, "tools": [{"t": 1}]}, {}, {"category": cat})
        for cat, n in (("simple", 30), ("irrelevance", 50)) for i in range(n)])
    levels = ["simple"] * 120 + ["moderate"] * 60 + ["challenging"] * 20
    save_items(d / "bird.jsonl", [Item("bird", str(i), [{"role": "user", "content": "q"}],
                                       {"max_tokens": 1024}, {}, {"difficulty": lv})
                                  for i, lv in enumerate(levels)])
    save_items(d / "humaneval_plus.jsonl", [_item(i=str(i), max_tokens=2048) for i in range(10)])
    return d


def test_prepare_draws_the_same_subsets_every_time_and_asks_for_thinking(tmp_path):
    items = _battery_items(tmp_path)
    first = mini.prepare(items, tmp_path / "a")
    mini.prepare(items, tmp_path / "b")
    for bench in mini.SIZES:
        assert (tmp_path / "a" / f"{bench}.jsonl").read_bytes() == (
            tmp_path / "b" / f"{bench}.jsonl").read_bytes()
    assert {b: e["n"] for b, e in first["benches"].items()} == {
        "ifeval": 200, "bfcl": 50, "bird": 150, "humaneval_plus": 10}
    assert first["benches"]["bird"]["strata"] == {"challenging": 15, "moderate": 45, "simple": 90}
    bfcl = load_items(tmp_path / "a" / "bfcl.jsonl")
    assert {it.meta["category"] for it in bfcl} == {"irrelevance"}
    assert bfcl[0].gen == {"max_tokens": 12288, "tools": [{"t": 1}], "thinking": True}


def test_thinking_drops_stop_strings_which_would_end_the_reasoning():
    # llama-server matches stop strings against everything it generates, the reasoning included:
    # DS-1000's "</code>" would end a reply whose reasoning names the tag. Its extraction cuts at
    # the same markers itself.
    item = _item("ds1000", max_tokens=1024, stop=["</code>", "# SOLUTION END"])
    assert mini.thinking(item).gen == {"max_tokens": 12288, "thinking": True}
    assert mini.thinking(item).ref == item.ref
    assert extract.ds1000_code("<code>x = 1</code>\nreasoning about </code> later") == "x = 1"


def test_each_level_keeps_its_share_by_largest_remainder():
    items = [Item("bird", str(i), [], {}, {}, {"difficulty": lv})
             for i, lv in enumerate(["a"] * 925 + ["b"] * 464 + ["c"] * 145)]  # BIRD-dev's levels
    chosen = mini.proportional(items, "difficulty", 150, seed=1)
    counts = {lv: sum(it.meta["difficulty"] == lv for it in chosen) for lv in "abc"}
    assert counts == {"a": 91, "b": 45, "c": 14}  # 90.4, 45.4, 14.2: the largest remainder is a's
    assert [it.id for it in chosen] == sorted((it.id for it in chosen), key=int)  # order kept


# --- summary -----------------------------------------------------------------------------------


def _write_state(run, bench, state, rows):
    """rows: per item (passed, status, reasoning_chars, unclosed)."""
    write_jsonl(run / "scores" / f"{bench}.{state}.jsonl",
                [{"id": str(i), "passed": p, "status": st} for i, (p, st, _, _) in enumerate(rows)])
    write_jsonl(run / "gen" / f"{bench}.{state}.jsonl",
                [{"id": str(i), "content": "" if u else "x", "reasoning_chars": r, "unclosed": u,
                  "finish_reason": "length" if u else "stop", "completion_tokens": r // 4}
                 for i, (_, _, r, u) in enumerate(rows)])


def test_a_reply_that_never_closed_fails_whatever_its_scorer_says():
    # BFCL irrelevance passes when no call is decoded, which a reply looping in its reasoning
    # satisfies; served, it answers nothing
    looping = {"unclosed": True, "content": "", "finish_reason": "length"}
    assert not mini.conditions("bfcl", {"passed": True, "status": "ok"}, looping)["passed"]
    assert mini.conditions("bfcl", {"passed": True, "status": "ok"}, {"content": "No tool fits."})[
        "passed"]


def test_summary_flags_what_got_worse_and_reads_the_aa_pass(tmp_path):
    run = tmp_path / "run"
    save_items(run / "items" / "bird.jsonl", [_item("bird", str(i)) for i in range(20)])
    save_items(run / "items" / "humaneval_plus.jsonl", [_item(i=str(i)) for i in range(11)])
    # BIRD: the base's SQL runs (2 wrong); the state's fails to run on 10 more items, and it
    # reasons 0.4 as long
    base = [(True, "ok", 1000, False)] * 18 + [(False, "wrong", 1000, False)] * 2
    state = [(True, "ok", 400, False)] * 8 + [(False, "error", 400, False)] * 10 + base[18:]
    for s, rows in (("base", base), ("adapter", state), ("base_rep", base)):
        _write_state(run, "bird", s, rows)
    # HumanEval+: the state leaves 8 of 10 measurable replies unclosed; item 10's reference fails
    he_base = [(True, "ok", 2000, False)] * 11
    he_state = [(False, "wrong", 50000, True)] * 8 + [(True, "ok", 2000, False)] * 3
    for s, rows in (("base", he_base), ("adapter", he_state), ("base_rep", he_base)):
        _write_state(run, "humaneval_plus", s, rows)
    # the gold scores cover the whole battery: item 11 fails too, but isn't in the subset
    write_jsonl(run / "scores" / "humaneval_plus.gold.jsonl",
                [{"id": str(i), "passed": i < 10} for i in range(12)])

    result = mini.summary(run, "adapter")
    assert set(result["flags"]) == {"bird:passed", "bird:sql_error", "bird:brevity",
                                    "humaneval_plus:passed", "humaneval_plus:unclosed",
                                    "humaneval_plus:truncated"}
    assert result["screen"] == "flagged" and result["aa_flags"] == []
    assert result["missing"] == ["ifeval", "bfcl"]
    bird = result["benches"]["bird"]
    assert bird["states"]["adapter"]["checks"]["sql_error"] == {
        "base": 0.0, "other": 50.0, "worse": 10, "better": 0, "p": 2 / 2**10, "flag": True}
    # 18 items at 0.4 of the base's length, 2 unchanged
    assert bird["states"]["adapter"]["checks"]["brevity"]["ratio_gmean"] == round(0.4 ** 0.9, 3)
    assert bird["aa_flips"] == 0
    assert bird["cost_base"]["completion_tokens"] == {"total": 5000, "median": 250, "p90": 250,
                                                      "max": 250}
    assert bird["states"]["adapter"]["cost"]["reasoning_chars_median"] == 400
    he = result["benches"]["humaneval_plus"]
    assert (he["n"], he["unmeasurable"]) == (10, ["10"])
    # unclosed replies sit out the length ratio: only the 2 measurable closed items remain
    assert he["states"]["adapter"]["checks"]["brevity"]["n"] == 2
    assert "sql_error" not in he["states"]["adapter"]["checks"]
    assert "| bird | sql_error | 20 | 0.0 | 50.0 | 10/0 |" in mini.markdown(result)


def test_a_pass_cut_short_is_compared_on_the_items_it_answered(tmp_path):
    run = tmp_path / "run"
    save_items(run / "items" / "ifeval.jsonl", [_item("ifeval", str(i)) for i in range(10)])
    _write_state(run, "ifeval", "base", [(True, "ok", 900, False)] * 10)
    # the A/A pass stopped at the time limit after 4 items: the scorer still writes 10 rows
    _write_state(run, "ifeval", "base_rep", [(True, "ok", 900, False)] * 4)
    write_jsonl(run / "scores" / "ifeval.base_rep.jsonl",
                [{"id": str(i), "passed": i < 4, "status": "ok" if i < 4 else "no_generation"}
                 for i in range(10)])
    check = mini.summary(run, "base_rep", aa="none")["benches"]["ifeval"]["states"]["base_rep"]
    assert check["checks"]["passed"]["worse"] == 0 and check["flags"] == []
