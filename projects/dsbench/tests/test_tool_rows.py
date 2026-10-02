"""Tests for Revision 2's tool rows (ADR-004 Revision 2, action item 4): the catalog, the generator
and the gold-call check. No network, no battery items: CI-safe.
"""
from __future__ import annotations

import json
import random

import pytest
from dsbench.sftgen import decontaminate as dc
from dsbench.sftgen import tool_rows as tr


@pytest.fixture(scope="module")
def generated():
    return tr.generate(seed=7)


def _call(name: str, args: dict) -> list[dict]:
    return [{"id": "c0", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]


def _gold_args(item: dict) -> dict:
    """The arguments a perfect reply passes: each stated value, and no optional left unstated."""
    return {k: v[0] for k, v in item["verify"]["gold"]["arguments"].items() if v[0] != ""}


# --- the catalog -------------------------------------------------------------------------------

def test_every_tool_is_a_valid_schema_its_templates_can_fill():
    assert len({t.name for t in tr.TOOLS}) == len(tr.TOOLS)
    for tool in tr.TOOLS:
        assert set(tool.required) <= set(tool.properties), tool.name
        for spec in tool.properties.values():
            if "enum" in spec and "default" in spec:
                assert spec["default"] in spec["enum"], tool.name
        for template in tool.templates:
            stated = set(tr.slots(template)) | set(template.fixed)
            assert set(tr.slots(template)) <= set(tool.properties), (tool.name, template.text)
            assert set(tool.required) <= stated, (tool.name, template.text)  # never guess one
            for seed in range(20):
                vals = tool.draw(random.Random(seed))
                assert set(tr.slots(template)) <= set(vals), (tool.name, template.text)
                gold = tr.gold_call(tool, vals, template)
                for key, accepted in gold["arguments"].items():
                    spec = tool.properties[key]
                    for value in accepted:
                        if value != "":  # every accepted value has the schema's type
                            assert tr._same(value, value, spec), (tool.name, key, value)
                        if value != "" and "enum" in spec:
                            assert value in spec["enum"], (tool.name, key, value)


def test_values_the_schema_formats_are_written_the_way_people_write_them():
    rng = random.Random(3)
    dates = {tr._date(rng)[1].say for _ in range(60)}
    assert any("," in d for d in dates) and any(d[:4] == "2027" for d in dates)
    times = [tr._time(rng) for _ in range(60)]
    assert all(len(t.value) == 5 and t.value[2] == ":" for t in times)
    assert any(t.say.endswith(("am", "pm")) for t in times)


# --- generation --------------------------------------------------------------------------------

def test_generation_is_reproducible_and_sized(generated):
    items, report = generated
    again, _ = tr.generate(seed=7)
    assert items == again
    assert (report["fit"], report["decline"]) == (295, 223)  # ceil(250/0.85), ceil(200/0.9)
    assert len({i["id"] for i in items}) == len(items)
    fit_counts = report["fit_by_tool"].values()
    assert max(fit_counts) - min(fit_counts) <= 1  # the targets cycle through the tools


def test_a_fit_row_offers_its_tool_and_a_decline_row_offers_nothing_that_fits(generated):
    items, _ = generated
    fit = [i for i in items if i["pool"] == "tool_fit"]
    decline = [i for i in items if i["pool"] == "tool_decline"]
    for item in fit:
        names = [t["function"]["name"] for t in item["tools"]]
        assert item["verify"]["gold"]["name"] in names
    fit_lists = {tuple(i["meta"]["listed"]) for i in fit}
    for item in decline:
        listed = [tr.TOOLS_BY_NAME[n] for n in item["meta"]["listed"]]
        assert tuple(item["meta"]["listed"]) in fit_lists  # the same lists, called and not
        assert not any(t.broad for t in listed)  # web search could answer anything
        if item["meta"]["decline"] == "other_tool":
            asked = tr.TOOLS_BY_NAME[item["meta"]["asks_for"]]
            assert asked.domain not in {t.domain for t in listed}


def test_general_requests_stay_away_from_the_tools_they_are_close_to(generated):
    items, _ = generated
    near = {text.split("{")[0]: domain for text, _, domain in tr.GENERAL if domain}
    for item in items:
        if item["meta"].get("decline") != "no_tool":
            continue
        domains = {tr.TOOLS_BY_NAME[n].domain for n in item["meta"]["listed"]}
        for prefix, domain in near.items():
            if item["messages"][0]["content"].startswith(prefix):
                assert domain not in domains


def test_every_item_passes_dsbench_s_gate(generated):
    items, _ = generated
    deny = dc.build_denylist()
    assert all(dc.scan_text(dc._row_text(item), deny) is None for item in items)


def test_a_gate_rejection_is_counted_and_replaced():
    seen = []

    def gate(item):
        seen.append(item["id"])
        return "test" if len(seen) % 5 == 0 else None

    items, report = tr.generate(seed=2, n_fit=20, n_decline=10, gate=gate)
    assert len(items) == 30 and sum(report["rejected_by_gate"].values()) == len(seen) - 30


# --- the check ---------------------------------------------------------------------------------

def test_the_gold_call_passes_on_every_fit_row(generated):
    items, _ = generated
    for item in items:
        if item["pool"] == "tool_fit":
            gold = item["verify"]["gold"]
            assert tr.check(item, _call(gold["name"], _gold_args(item))) == (True, ""), item["id"]


@pytest.fixture
def forecast():
    tool = tr.TOOLS_BY_NAME["get_weather_forecast"]
    template = tool.templates[0]  # states city and days, not units
    vals = {"city": tr.Val("Oslo", "Oslo", ("Oslo, Norway",)), "days": tr.Val(3, "three"),
            "units": tr.Val("celsius", "Celsius")}
    return {"tools": [tool.schema(), tr.TOOLS_BY_NAME["get_local_time"].schema()],
            "verify": {"kind": "call", "gold": tr.gold_call(tool, vals, template)}}


@pytest.mark.parametrize("args, ok", [
    ({"city": "Oslo", "days": 3}, True),
    ({"city": "oslo, norway", "days": 3}, True),  # a variant, compared as BFCL does
    ({"city": "Oslo", "days": 3, "units": "celsius"}, True),  # an unstated default
    ({"city": "Oslo", "days": 3, "units": "fahrenheit"}, False),  # an unstated choice
    ({"city": "Oslo", "days": 3.0}, False),  # an integer must be an integer
    ({"city": "Oslo", "days": "3"}, False),
    ({"city": "Bergen", "days": 3}, False),
    ({"city": "Oslo"}, False),  # a required parameter missing
    ({"city": "Oslo", "days": 3, "hourly": True}, False),  # not in the schema
])
def test_arguments_are_checked_by_name_type_and_value(forecast, args, ok):
    assert tr.check(forecast, _call("get_weather_forecast", args))[0] is ok


def test_the_call_itself_must_be_right(forecast):
    good = {"city": "Oslo", "days": 3}
    assert tr.check(forecast, _call("get_local_time", {"city": "Oslo"}))[1].startswith("called")
    assert tr.check(forecast, _call("get_weather_forecast", good) * 2)[0] is False
    assert tr.check(forecast, [])[1] == "expected one call, got 0"
    broken = [{"function": {"name": "get_weather_forecast", "arguments": "{'city': "}}]
    assert tr.check(forecast, broken)[1].startswith("format")


def test_a_stated_optional_value_must_be_passed():
    tool = tr.TOOLS_BY_NAME["get_weather_forecast"]
    vals = {"city": tr.Val("Oslo", "Oslo"), "days": tr.Val(3, "3"),
            "units": tr.Val("fahrenheit", "Fahrenheit")}
    item = {"tools": [tool.schema()],
            "verify": {"kind": "call", "gold": tr.gold_call(tool, vals, tool.templates[1])}}
    assert tr.check(item, _call(tool.name, {"city": "Oslo", "days": 3}))[1].startswith("left out")
    assert tr.check(item, _call(tool.name, {"city": "Oslo", "days": 3, "units": "fahrenheit"}))[0]


def test_numbers_accept_an_integer_and_lists_ignore_order():
    loan = tr.TOOLS_BY_NAME["calculate_loan_payment"]
    vals = {"principal": tr.Val(250000, "250,000"), "annual_rate_percent": tr.Val(4.5, "4.5"),
            "years": tr.Val(30, "30")}
    item = {"tools": [loan.schema()],
            "verify": {"kind": "call", "gold": tr.gold_call(loan, vals, loan.templates[0])}}
    args = {"principal": 250000.0, "annual_rate_percent": 4.5, "years": 30}
    assert tr.check(item, _call(loan.name, args))[0]
    recipes = tr.TOOLS_BY_NAME["find_recipes"]
    vals = {"ingredients": tr.Val(["tofu", "broccoli", "ginger"], "tofu, broccoli and ginger")}
    item = {"tools": [recipes.schema()],
            "verify": {"kind": "call",
                       "gold": tr.gold_call(recipes, vals, recipes.templates[0])}}
    assert tr.check(item, _call(recipes.name, {"ingredients": ["ginger", "tofu", "broccoli"]}))[0]
    assert not tr.check(item, _call(recipes.name, {"ingredients": ["tofu", "broccoli"]}))[0]


def test_a_decline_row_passes_only_without_a_call():
    item = {"tools": [], "verify": {"kind": "no_call"}}
    assert tr.check(item, []) == (True, "")
    assert tr.check(item, _call("get_local_time", {"city": "Oslo"})) == (False,
                                                                        "called get_local_time")


def test_verify_needs_a_finished_reply_with_reasoning(tmp_path, generated):
    items, _ = generated
    item = next(i for i in items if i["pool"] == "tool_fit")
    gold = item["verify"]["gold"]
    good = {"id": item["id"], "reasoning": "The user wants ...", "finish_reason": "tool_calls",
            "tool_calls": _call(gold["name"], _gold_args(item))}
    gens = [good, {**good, "reasoning": ""}, {**good, "finish_reason": "length"}]
    # one item per reply: a checker reads one row an item (reasoning_pilot.latest_rows)
    gens = [{**g, "id": f"{item['id']}#{k}"} for k, g in enumerate(gens)]
    (tmp_path / "items.jsonl").write_text("".join(json.dumps({**item, "id": g["id"]}) + "\n"
                                                  for g in gens))
    (tmp_path / "gen.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gens))
    result = tr.verify(tmp_path / "items.jsonl", tmp_path / "gen.jsonl", tmp_path / "out.jsonl")
    assert result == {"passed": {"tool_fit": 1}, "failed": {"tool_fit": 2}}
    whys = [json.loads(line)["check"]["why"] for line in (tmp_path / "out.jsonl").open()]
    assert whys == ["", "no reasoning", "unfinished: length"]
