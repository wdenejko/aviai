"""Unit tests for the pilot-mixture assembler (pure; uses the real dsbench denylist, no network)."""
from __future__ import annotations

import random

from dsbench.sftgen.assemble import _prov, _select
from dsbench.sftgen.decontaminate import build_denylist


def _clean(i: int) -> dict:
    return {"messages": [{"role": "user", "content": f"question number {i} about widgets"},
                         {"role": "assistant", "content": "here is a neutral helpful answer"}],
            "meta": {"licence": "MIT", "teacher": "DeepSeek", "redistributable": True}}


def test_prov_both_meta_shapes():
    lic, teach, redist = _prov({"meta": {"licence": "MIT", "teacher": "DeepSeek",
                                         "redistributable": True}})
    assert (lic, teach, redist) == ("MIT", "DeepSeek", True)
    # our targeted slice: provenance sub-dict, always treated as redistributable
    lic, teach, redist = _prov({"meta": {"provenance": {"licence": "Apache-2.0",
                                                        "teacher": None}}})
    assert redist is True and lic == "Apache-2.0" and teach == "own-generated"


def test_select_drops_contaminated_and_stamps_bucket():
    deny = build_denylist()
    dirty = {"messages": [{"role": "user", "content": "select from aviation.flights"},
                          {"role": "assistant", "content": "ok"}],
             "meta": {"licence": "MIT", "teacher": "X", "redistributable": True}}
    recs = [_clean(i) for i in range(50)] + [dirty]
    # a big budget forces every record to be evaluated, so the contaminated one is reached + dropped
    kept, stats = _select(recs, target=10_000, deny=deny, rng=random.Random(1),
                          pool="p", bucket="ds_notebooks")
    assert stats["dropped_contam"] == 1
    assert all("aviation.flights" not in m["messages"][0]["content"] for m in kept)
    assert all(m["meta"]["mix_bucket"] == "ds_notebooks" for m in kept)
    assert all(m["meta"]["mix_pool"] == "p" for m in kept)


def test_select_respects_token_budget():
    kept, stats = _select([_clean(i) for i in range(200)], target=60, deny=build_denylist(),
                          rng=random.Random(2), pool="p", bucket="b")
    # stops shortly after crossing the budget, nowhere near all 200 rows
    assert stats["tokens"] >= 60
    assert stats["rows"] < 50
