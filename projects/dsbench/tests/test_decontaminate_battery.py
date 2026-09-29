"""Tests for the training-side battery gate (decontaminate.py rule 4, ADR-004 Revision 2).

The items are synthetic stand-ins shaped like the battery's own: a long MMLU-Pro-style question,
a BFCL request short enough that a 13-gram cannot see it, its function schema, and a request too
short to match at all.
"""

from __future__ import annotations

import json

import pytest
from dsbench.battery.items import Item, save_items
from dsbench.sftgen import decontaminate as dc

WORDS = [f"w{i}" for i in range(40)]  # 40 tokens: 28 distinct 13-grams
SCHEMA = [{"type": "function", "function": {
    "name": "circle_area",
    "description": "Calculate the area of a circle from its radius, in the unit of the radius",
    "parameters": {"type": "dict", "properties": {"radius": {"type": "integer"}},
                   "required": ["radius"]}}}]
ITEMS = [
    Item("mmlu_pro", "1", [{"role": "user", "content":
                            "Question: " + " ".join(WORDS) + "\nAnswer: Let's think"}]),
    Item("bfcl", "simple_9", [{"role": "user", "content":
                               "Calculate the area of a circle with a radius of 5 units."}],
         gen={"tools": SCHEMA}),
    Item("bfcl", "tiny", [{"role": "user", "content": "Who discovered radium?"}]),
]


@pytest.fixture
def items_dir(tmp_path):
    # Its own directory: the gate reads every *.jsonl there as battery items.
    save_items(tmp_path / "items" / "mmlu_pro.jsonl", ITEMS[:1])
    save_items(tmp_path / "items" / "bfcl.jsonl", ITEMS[1:])
    return tmp_path / "items"


@pytest.fixture
def deny(items_dir):
    return dc.build_denylist(battery_items=str(items_dir))


def _chat(user, reasoning=None, tools=None):
    row = {"messages": [{"role": "user", "content": user},
                        {"role": "assistant", "content": "ok", "reasoning_content": reasoning}]}
    if tools:
        row["tools"] = tools
    return row


def test_the_index_matches_short_items_whole_and_counts_the_shortest_unchecked(deny):
    units = {(u.bench, u.id): u.size for u in deny.battery.units}
    assert units == {("mmlu_pro", "1"): 28, ("bfcl", "simple_9"): 1,
                     ("bfcl-schema", "simple_9"): units[("bfcl-schema", "simple_9")]}
    assert deny.battery.unchecked == {"bfcl": 1}  # "Who discovered radium?": 3 tokens


def test_a_fifth_of_an_item_rejects_the_row_and_less_is_only_audited(deny):
    # 18 tokens = 6 of the item's 28 grams (>= a fifth); 13 tokens = 1 gram.
    leak = dc.scan_text(dc._row_text(_chat(" ".join(WORDS[:18]))), deny)
    assert leak == {"rule": "battery", "hit": "mmlu_pro:1", "shared": 6, "of": 28}
    audit: list[dict] = []
    assert dc.scan_text(dc._row_text(_chat(" ".join(WORDS[:13]))), deny, audit=audit) is None
    assert audit == [{"rule": "battery", "hit": "mmlu_pro:1", "shared": 1, "of": 28}]


def test_a_short_item_counts_only_when_found_whole(deny):
    whole = "Please calculate the area of a circle with a radius of 5 units, thanks."
    assert dc.scan_text(dc._row_text(_chat(whole)), deny)["hit"] == "bfcl:simple_9"
    other = "Calculate the area of a circle with a radius of 6 units."
    audit: list[dict] = []
    assert dc.scan_text(dc._row_text(_chat(other)), deny, audit=audit) is None
    assert audit == []


def test_a_row_offering_a_bfcl_schema_is_rejected_whatever_its_key_order(deny):
    reordered = [{"function": dict(reversed(list(SCHEMA[0]["function"].items()))),
                  "type": "function"}]
    hit = dc.scan_text(dc._row_text(_chat("What is the area?", tools=reordered)), deny)
    assert hit["hit"] == "bfcl-schema:simple_9" and hit["shared"] == hit["of"]


def test_the_reasoning_is_scanned_too(deny):
    row = _chat("Explain.", reasoning="Recall: " + " ".join(WORDS))
    assert dc.scan_text(dc._row_text(row), deny)["hit"] == "mmlu_pro:1"


def test_scan_file_reports_rejections_by_unit_and_the_audit(items_dir, tmp_path):
    mix = tmp_path / "mix.jsonl"
    rows = [_chat(" ".join(WORDS[:18])), _chat(" ".join(WORDS[:13])), _chat("hello")]
    for n, row in enumerate(rows):
        row["meta"] = {"id": f"r{n}"}
    mix.write_text("".join(json.dumps(r) + "\n" for r in rows))

    report = dc.scan_file(str(mix), battery_items=str(items_dir))

    assert (report["scanned"], report["clean"], report["rejected"]) == (3, 2, 1)
    assert report["examples"][0]["row"] == "r0"
    battery = report["battery"]
    assert battery["rejected_by_unit"] == {"mmlu_pro:1": 1}
    below = battery["below_line"]
    assert below["rows"] == 1 and below["examples"][0]["row"] == "r1"
    assert battery["unchecked_too_short"] == {"bfcl": 1}


def test_an_empty_items_dir_fails_closed(tmp_path):
    with pytest.raises(SystemExit):
        dc.build_denylist(battery_items=str(tmp_path))

