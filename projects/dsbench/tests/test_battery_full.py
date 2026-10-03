"""Tests for the full battery with thinking on (ADR-004 decision 9): its items, and the report's
rules for thinking-on passes."""

from __future__ import annotations

import json

from dsbench.battery import full, report, stats
from dsbench.battery.items import Item, load_items, save_items, write_jsonl


def _items(bench: str, n: int, **gen) -> list[Item]:
    return [Item(bench, str(i), [{"role": "user", "content": "q"}], gen,
                 meta={"category": "irrelevance"}) for i in range(n)]


def test_every_item_asks_for_thinking_in_a_seeded_order_per_benchmark(tmp_path):
    pinned = tmp_path / "pinned"
    save_items(pinned / "ds1000.jsonl", _items("ds1000", 20, max_tokens=1024, stop=["</code>"]))
    save_items(pinned / "ifeval.jsonl", _items("ifeval", 20, max_tokens=2048))
    (pinned / "manifest.json").write_text(json.dumps({"seed": 20260924}))

    first = full.prepare(pinned, tmp_path / "a", ("ds1000", "ifeval"))
    again = full.prepare(pinned, tmp_path / "b", ("ds1000", "ifeval"))

    ds = [it.id for it in load_items(tmp_path / "a" / "ds1000.jsonl")]
    assert sorted(ds, key=int) == [str(i) for i in range(20)]  # every item, once
    assert ds != sorted(ds, key=int)  # shuffled: the first N are a random sample
    assert ds == [it.id for it in load_items(tmp_path / "b" / "ds1000.jsonl")]  # seeded
    assert ds != [it.id for it in load_items(tmp_path / "a" / "ifeval.jsonl")]  # per benchmark
    assert {json.dumps(it.gen) for it in load_items(tmp_path / "a" / "ds1000.jsonl")} == {
        json.dumps({"max_tokens": 12288, "thinking": True})}  # the stop strings are gone
    assert first == again
    benches = first["benches"]
    assert (benches["ds1000"]["stops_dropped"], benches["ifeval"]["n"]) == (20, 20)
    assert first["source"] == {"seed": 20260924}
    # report.py embeds the run's items/manifest.json
    assert json.loads((tmp_path / "a" / "manifest.json").read_text()) == first


def test_the_report_counts_a_thinking_pass_as_the_mini_battery_does(tmp_path):
    # A reply whose reasoning never closed fails, though BFCL's irrelevance scorer passes it (no
    # call decoded); and a pass the window's time limit cut is compared on the items it answered.
    save_items(tmp_path / "items" / "bfcl.jsonl", _items("bfcl", 6))
    write_jsonl(tmp_path / "scores" / "bfcl.base.jsonl",
                [{"id": str(i), "passed": True, "status": "ok"} for i in range(6)])
    write_jsonl(tmp_path / "gen" / "bfcl.base.jsonl",
                [{"id": str(i), "content": "" if i == 0 else "No tool fits.", "unclosed": i == 0}
                 for i in range(6)])
    # the adapter's pass stopped after 4 items; the scorer failed the rest as no_generation
    write_jsonl(tmp_path / "scores" / "bfcl.adapter.jsonl",
                [{"id": str(i), "passed": i < 4, "status": "ok" if i < 4 else "no_generation"}
                 for i in range(6)])
    write_jsonl(tmp_path / "gen" / "bfcl.adapter.jsonl",
                [{"id": str(i), "content": "" if i == 1 else "No tool fits.", "unclosed": i == 1}
                 for i in range(4)])

    s = report.bench_summary(tmp_path, "bfcl", strong=None)

    p = s["paired"]
    assert (p["n"], p["gains"], p["losses"]) == (4, 1, 1)  # items 0-3: 0 gained, 1 lost
    assert s["unclosed"] == {"base": 1, "adapter": 1}
    assert s["strata"]["irrelevance"]["n"] == 4
    md = report.markdown({"bfcl_irrelevance": s}, stats.gate2_decision({}))
    assert "Replies whose reasoning never closed: bfcl_irrelevance base 1, adapter 1." in md


def test_a_thinking_off_pass_is_reported_as_before(tmp_path):
    # Gate 2's passes answered every item and carry no `unclosed`: same pairs, no loop line
    save_items(tmp_path / "items" / "ifeval.jsonl", _items("ifeval", 3))
    for state, passed in (("base", [1, 1, 0]), ("adapter", [0, 1, 1])):
        write_jsonl(tmp_path / "scores" / f"ifeval.{state}.jsonl",
                    [{"id": str(i), "passed": bool(p), "status": "ok"}
                     for i, p in enumerate(passed)])
        write_jsonl(tmp_path / "gen" / f"ifeval.{state}.jsonl",
                    [{"id": str(i), "content": "x"} for i in range(3)])
    s = report.bench_summary(tmp_path, "ifeval", strong=None)
    assert (s["paired"]["n"], s["paired"]["gains"], s["paired"]["losses"]) == (3, 1, 1)
    assert s["unclosed"] == {"base": 0, "adapter": 0}
    assert "never closed" not in report.markdown({"ifeval": s}, stats.gate2_decision({}))
