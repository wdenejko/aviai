"""Tests for Revision 2's prompt selector (ADR-004 Revision 2, action item 6).

Synthetic pools in a temporary data zone and a one-item battery: no network, no real pools.
"""
from __future__ import annotations

import json

import pytest
from dsbench.battery.contamination import build_index
from dsbench.battery.items import Item
from dsbench.sftgen.select_prompts import (
    Pool,
    even_allocation,
    prompt_messages,
    quota,
    select,
)

BATTERY_TEXT = "the quick brown fox jumps over the lazy dog while seven wise owls watch the moon"


def _row(user: str, meta: dict | None = None, history: tuple = ()) -> dict:
    msgs = [*history, {"role": "user", "content": user}, {"role": "assistant", "content": "ans"}]
    return {"messages": msgs, "loss_mask_roles": ["assistant"], "meta": dict(meta or {})}


def _write(path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_prompt_ends_at_the_last_user_turn_and_keeps_its_history():
    msgs = [{"role": "system", "content": "be brief"},
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "x": 1},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    assert prompt_messages(msgs) == [
        {"role": "system", "content": "be brief"}, {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]


def test_even_allocation_is_equal_where_the_groups_allow():
    assert even_allocation({"a": 100, "b": 100, "c": 100}, 10) == {"a": 3, "b": 3, "c": 4}
    assert even_allocation({"rare": 1, "b": 100, "c": 100}, 10) == {"rare": 1, "b": 4, "c": 5}
    assert sum(even_allocation({"a": 2, "b": 3}, 10).values()) == 5  # they run out


def test_quotas_normalise_the_shares_and_cover_the_expected_losses():
    pools = (Pool("x", "replay", "f", 0.5), Pool("y", "replay", "f", 0.5, keep_rate=0.8),
             Pool("z", "replay", "f", 0.0))
    rows = {"replay": 100}
    assert [quota(p, pools, rows) for p in pools] == [53, 63, 0]  # ceil(50/0.95), ceil(50/0.8)
    wider = (*pools[:2], Pool("z", "replay", "f", 1.0))  # a new share comes from the others
    assert [quota(p, wider, rows) for p in wider] == [27, 32, 53]


POOLS = (
    Pool("a", "replay", "mix.jsonl", 0.5, keep_rate=1.0, source="A"),
    Pool("b", "replay", "mix.jsonl", 0.0, keep_rate=1.0, source="B"),
    Pool("aya", "replay", "aya.jsonl", 0.25, keep_rate=1.0, balance="language"),
    Pool("gsm", "replay", "gsm.jsonl", 0.25, keep_rate=1.0),
    Pool("code", "code", "code.jsonl", 1.0, keep_rate=1.0, requires="testcase"),
)
ROWS = {"replay": 12, "code": 4}  # quotas: a 6, b 0, aya 3, gsm 3, code 4


@pytest.fixture
def zone(tmp_path):
    history = ({"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"})
    mix = [_row(f"question {i}", {"source": "A", "id": f"a{i}"}) for i in range(5)]
    mix += [_row("and then?", {"source": "A", "id": "a5"}, history),  # 6 eligible
            _row("question 0", {"source": "A", "id": "a-dup"}),
            _row("x" * 400, {"source": "A", "id": "a-long"}),
            _row(f"please {BATTERY_TEXT} now", {"source": "A", "id": "a-battery"})]
    mix += [_row(f"flan {i}", {"source": "B", "id": f"b{i}"}) for i in range(3)]
    _write(tmp_path / "mix.jsonl", mix)
    _write(tmp_path / "aya.jsonl", [_row(f"{lang} {i}", {"language": lang})
                                    for lang, n in (("x", 2), ("y", 10), ("z", 10))
                                    for i in range(n)])
    _write(tmp_path / "gsm.jsonl", [_row(f"sum {i}", {"gold_answer": str(i)}) for i in range(4)])
    code = [_row(f"write f{i}", {"testcase": [f"assert f{i}()"], "entry_point": f"f{i}"})
            for i in range(3)]
    _write(tmp_path / "code.jsonl", [*code, _row("write g", {})])
    return tmp_path


def _select(zone, **kw):
    index = build_index([Item("bfcl", "1", [{"role": "user", "content": BATTERY_TEXT}])])
    return select(zone, index, pools=POOLS, bucket_rows=ROWS, max_prompt_tokens=100, **kw)


def test_selection_filters_balances_and_carries_the_checks(zone):
    items, report = _select(zone)
    by = {e["pool"]: e for e in report["pools"]}
    assert by["a"]["dropped"] == {"duplicate": 1, "long": 1, "battery": 1}
    assert (by["a"]["eligible"], by["a"]["selected"]) == (6, 6)
    assert (by["b"]["quota"], by["b"]["selected"], by["b"]["eligible"]) == (0, 0, 3)  # FLAN's 0
    assert by["aya"]["by_language"] == {"x": 1, "y": 1, "z": 1}
    code = by["code"]
    assert (code["dropped"], code["selected"], code["shortfall"]) == ({"no testcase": 1}, 3, 1)

    pools = {item["id"]: item for item in items}
    assert len(pools) == len(items) == 6 + 3 + 3 + 3
    assert all(item["messages"][-1]["role"] == "user" for item in items)
    assert pools["a:a5"]["messages"][:2] == [{"role": "user", "content": "hi"},
                                             {"role": "assistant", "content": "hello"}]
    gsm = [item for item in items if item["pool"] == "gsm"]
    assert all(item["verify"]["gold_answer"] == item["messages"][0]["content"][4:] for item in gsm)
    assert all("gold_answer" not in item["meta"] for item in gsm)
    assert {tuple(item["verify"]["tests"]) for item in items if item["pool"] == "code"} == {
        ("assert f0()",), ("assert f1()",), ("assert f2()",)}


def test_selection_is_reproducible_and_a_top_up_skips_earlier_picks(zone):
    first, _ = _select(zone)
    again, _ = _select(zone)
    assert [item["id"] for item in first] == [item["id"] for item in again]
    earlier = frozenset(item["id"] for item in first if item["pool"] == "gsm")
    top_up, report = _select(zone, exclude=earlier)
    gsm = next(e for e in report["pools"] if e["pool"] == "gsm")
    assert (gsm["dropped"], gsm["selected"], gsm["shortfall"]) == ({"selected before": 3}, 1, 2)
    assert not earlier & {item["id"] for item in top_up}


def test_a_missing_pool_says_how_to_get_it(tmp_path):
    with pytest.raises(SystemExit, match="acquire it first"):
        select(tmp_path, None, pools=POOLS[:1], bucket_rows=ROWS)
