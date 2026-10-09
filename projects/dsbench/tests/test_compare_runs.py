"""Tests for comparing two pi runs (agentic/compare_runs.py): Fisher's exact test against hand
counts, the per-problem table, the battery's majority pairing, and the refusals."""
from __future__ import annotations

import json

import pytest
from dsbench.agentic import compare_runs as cr


@pytest.mark.parametrize("table, p", [
    ((0, 5, 5, 0), 2 / 252),  # 0 of 5 against 5 of 5: the two most extreme tables
    ((1, 4, 4, 1), 52 / 252),  # 1 of 5 against 4 of 5
    ((5, 0, 5, 0), 1.0),  # no difference
    ((3, 2, 3, 2), 1.0),
    ((5, 0, 0, 5), 2 / 252),
])
def test_fisher_s_exact_test(table, p):
    assert cr.fisher_exact(*table) == pytest.approx(p)


META = {"harness": "pi-probe", "thinking": "high", "repeat": 5, "reply_max_tokens": 12288,
        "provider": "dashi-qwen36", "model": "qwen36"}


def _run(label, passes: dict[str, list[str]]) -> dict:
    """A pi run's payload: per problem, its runs' statuses (`ok` passes)."""
    results = [{"id": pid, "category": "da", "passed": s == "ok", "status": s, "latency_s": 10.0}
               for pid, statuses in passes.items() for s in statuses]
    return {"meta": {**META, "label": label, "timestamp": "t"}, "results": results}


def test_problems_are_compared_by_their_runs_and_by_majority():
    base = _run("rev2-probe-base", {"weekday": ["wrong"] * 5, "utc": ["ok"] * 3 + ["wrong"] * 2,
                                     "share": ["ok"] * 5})
    adapter = _run("rev2-probe-adapter", {"weekday": ["ok"] * 5, "utc": ["ok"] * 5,
                                           "share": ["ok"] * 4 + ["timeout"]})
    result = cr.compare(base, adapter)
    weekday = result["problems"]["weekday"]
    assert (weekday["base"], weekday["adapter"], weekday["runs"]) == (0, 5, [5, 5])
    assert weekday["p_fisher"] == pytest.approx(0.00794, abs=1e-5)
    assert (weekday["base_majority"], weekday["adapter_majority"]) == (False, True)
    assert result["problems"]["share"]["statuses"]["adapter"] == {"ok": 4, "timeout": 1}
    assert result["runs_passed"] == {"base": 8, "adapter": 14, "of": 15}
    assert result["problems_passed"] == {"base": 2, "adapter": 3, "of": 3}
    assert result["majority_flips"] == {"lost": 0, "gained": 1, "p_mcnemar": 1.0}
    assert result["settings"]["reply_max_tokens"] == 12288
    table = cr.markdown(result)
    assert "| weekday | da | 0/5 | 5/5 | 0.00794 | wrong×5 | ok×5 |" in table
    assert "Runs passed: base 8/15, adapter 14/15." in table


def test_runs_with_other_settings_or_other_problems_are_not_compared():
    base = _run("b", {"weekday": ["ok"] * 5})
    other_cap = {**_run("a", {"weekday": ["ok"] * 5})}
    other_cap["meta"] = {**other_cap["meta"], "reply_max_tokens": 16384}
    with pytest.raises(ValueError, match="reply_max_tokens"):
        cr.compare(base, other_cap)
    with pytest.raises(ValueError, match="different problems"):
        cr.compare(base, _run("a", {"utc": ["ok"] * 5}))


def test_the_cli_writes_the_comparison(tmp_path, capsys, monkeypatch):
    for name, statuses in (("base", ["wrong"] * 5), ("adapter", ["ok"] * 5)):
        (tmp_path / f"{name}.json").write_text(json.dumps(_run(name, {"weekday": statuses})))
    monkeypatch.setattr("sys.argv", ["compare_runs", "--base", str(tmp_path / "base.json"),
                                     "--adapter", str(tmp_path / "adapter.json"),
                                     "--out", str(tmp_path / "out.json")])
    cr.main()
    assert json.loads((tmp_path / "out.json").read_text())["problems_passed"]["adapter"] == 1
    assert "| weekday |" in capsys.readouterr().out


def test_several_runs_of_one_state_count_together():
    """Revision 2 against Revision 2.1 in one window: three blocks of k = 5 per adapter."""
    blocks = [_run(f"h2h-rev2-{i}",
                   {"weekday": ["ok"] * 5, "utc": ["ok"] * i + ["wrong"] * (5 - i)})
              for i in (1, 2, 3)]
    merged = cr.merge(blocks)
    assert merged["meta"]["repeat"] == 15
    assert merged["meta"]["label"] == "h2h-rev2-1+h2h-rev2-2+h2h-rev2-3"
    assert len(merged["results"]) == 30
    other = cr.merge([_run(f"h2h-rev21-{i}", {"weekday": ["wrong"] * 5, "utc": ["ok"] * 5})
                      for i in (1, 2, 3)])
    result = cr.compare(merged, other)
    assert result["problems"]["utc"]["runs"] == [15, 15]
    assert (result["problems"]["utc"]["base"], result["problems"]["utc"]["adapter"]) == (6, 15)
    assert result["problems"]["weekday"]["p_fisher"] == pytest.approx(2 / 155117520, rel=1e-3)


def test_runs_with_other_settings_or_problems_are_not_merged():
    with pytest.raises(ValueError, match="different problems"):
        cr.merge([_run("a", {"weekday": ["ok"] * 5}), _run("b", {"utc": ["ok"] * 5})])
    other = _run("b", {"weekday": ["ok"] * 5})
    other["meta"] = {**other["meta"], "thinking": "low"}
    with pytest.raises(ValueError, match="thinking"):
        cr.merge([_run("a", {"weekday": ["ok"] * 5}), other])


def test_the_cli_merges_several_runs_a_side(tmp_path, monkeypatch):
    paths = {}
    for name, statuses in (("b1", ["wrong"] * 5), ("b2", ["ok"] * 5), ("a1", ["ok"] * 5),
                           ("a2", ["ok"] * 5)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(_run(name, {"weekday": statuses})))
    monkeypatch.setattr("sys.argv", ["compare_runs", "--base", str(paths["b1"]), str(paths["b2"]),
                                     "--adapter", str(paths["a1"]), str(paths["a2"]),
                                     "--out", str(tmp_path / "out.json")])
    cr.main()
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["runs_passed"] == {"base": 5, "adapter": 10, "of": 10}
