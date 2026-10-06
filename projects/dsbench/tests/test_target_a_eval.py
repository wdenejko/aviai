"""Tests for Target A's direct test (sftgen/target_a_eval.py): the held-out domains share no name
with the training ones, so every test prompt is new; the items' draws; the paired comparison and
its exact McNemar test. CI has in-process DuckDB only, so the draws run on a stand-in generator.
"""
from __future__ import annotations

import json

import pytest
from dsbench.sftgen import synth
from dsbench.sftgen import target_a_eval as te
from dsbench.sftgen.conventions import _friendly_type
from dsbench.sftgen.dialect_conventions import generate
from dsbench.sftgen.schema import row_to_dict

TRAINING = ["retail_orders", "iot_readings", "support_tickets", "payments", "web_sessions",
            "gym_checkins"]

# --- the held-out domains ------------------------------------------------------------------------


def test_the_training_domains_are_as_they_were():
    # A table's seed follows its domain's position: a new domain among these would change every
    # table a rerun of an earlier draw builds.
    assert synth.domain_names() == TRAINING


def _names(domain: synth.Domain) -> set[str]:
    return {domain.name, domain.label, *domain.df.columns}


def _shape(domain: synth.Domain) -> tuple[str, ...]:
    return tuple(_friendly_type(t) for t in domain.df.dtypes)


def test_no_held_out_name_is_a_training_one_and_the_shapes_are_training_shapes():
    training = [synth.build(name, 1, 20) for name in synth.domain_names()]
    held_out = [synth.build(name, 1, 20) for name in synth.held_out_domain_names()]
    assert len(held_out) == 6
    assert not set().union(*map(_names, training)) & set().union(*map(_names, held_out))
    # an id, a timestamp, a city, a category, a number: five of the six training domains' shape
    assert {_shape(d) for d in held_out} == {("integer", "timestamp", "text", "text", "number")}
    assert sum(_shape(d) == _shape(held_out[0]) for d in training) == 5
    for domain in held_out:
        assert domain.ts_col in domain.df and domain.location_col in domain.df
        assert set(domain.df[domain.location_col]) <= set(synth.US_CITY_UTC_OFFSETS["daylight"])
        assert "-" not in domain.name  # a row id splits on hyphens (parse_row_id)


def test_a_held_out_table_rebuilds_from_its_seed():
    first, again = synth.build("ride_trips", 42, 300), synth.build("ride_trips", 42, 300)
    assert first.df.equals(again.df)
    assert not first.df.equals(synth.build("ride_trips", 43, 300).df)


def test_no_prompt_on_a_held_out_domain_is_a_training_prompt():
    def prompts(domains, reps):
        rows, _ = generate(seed=3, reps=reps, n=200, dialects=["duckdb"], thinking_frac=0.0,
                           domains=domains)
        return {json.dumps([t["content"] for t in row_to_dict(r)["turns"][:2]]) for r in rows}

    held_out = prompts(synth.held_out_domain_names(), 2)
    assert len(held_out) > 40
    assert not held_out & prompts(None, 4)  # the default: the training domains


# --- the items -----------------------------------------------------------------------------------


def _row(family, dialect, domain, seed) -> dict:
    return {"id": f"A-{family}-{dialect}-{domain}-{seed}", "family": family, "dialect": dialect,
            "turns": [{"content": f"You are writing SQL. There is one table `{domain}`."},
                      {"content": f"Count the {family} rows."},
                      {"content": "```sql\nSELECT count(*) FROM t\n```"}],
            "verification": {"truth": 5}}


def _stand_in(calls, table=None):
    """dialect_conventions.generate's rows, with their tables' seeds as it draws them (or one
    `table` for every row), and none of its engines."""
    from dsbench.sftgen.dialect_conventions import table_seed

    def fake_generate(*, seed, reps, n, dialects, families, thinking_frac, domains):
        calls.append((seed, reps, tuple(dialects), tuple(families), tuple(domains)))
        rows = [_row(f, d, domain, table or table_seed(seed, k, r)) for r in range(reps)
                for k, domain in enumerate(domains) for d in dialects for f in families]
        return rows, {"engines": list(dialects), "rejected": 0}

    return fake_generate


@pytest.fixture
def drawn(monkeypatch):
    from dsbench.sftgen import dialect_conventions, schema

    calls = []
    monkeypatch.setattr(dialect_conventions, "generate", _stand_in(calls))
    monkeypatch.setattr(schema, "row_to_dict", lambda row: row)
    return calls


def test_the_items_draw_every_cell_and_clickhouse_s_weekdays_again(drawn):
    items, report = te.build_test_items(seed=100, cell_reps=2,
                                        extra_reps={"weekday-numbering": 4, "weekend-flag": 2})
    held_out = tuple(synth.held_out_domain_names())
    assert drawn == [(100, 2, te.DIALECTS, te.FAMILIES, held_out),
                     (101, 4, ("clickhouse",), ("weekday-numbering",), held_out),
                     (102, 2, ("clickhouse",), ("weekend-flag",), held_out)]
    assert len(items) == 6 * 2 * 16 + 6 * 4 + 6 * 2 == report["items"]
    cells = report["items_by_cell"]
    assert (cells["clickhouse/weekday-numbering"], cells["clickhouse/weekend-flag"]) == (36, 24)
    assert {n for c, n in cells.items() if c not in te.TARGET_CELLS} == {12}
    first = items[0]
    assert first["id"] == "targetA_test:A-weekday-numbering-clickhouse-library_loans-100"
    assert items[-1]["id"] == ("targetA_test:A-weekend-flag-clickhouse-bike_rentals-"
                               f"{102 + 5 * 100003 + 1 * 7919}")  # table_seed's strides
    assert first["pool"] == "targetA_test" and first["max_tokens"] == te.TEST_MAX_TOKENS == 12288
    assert first["meta"]["held_out"] and "hinted" not in first["meta"]
    assert "prefill" not in first and "hint" not in first["verify"]  # the plain prompt, alone
    assert first["verify"]["gold_sql"] == "SELECT count(*) FROM t"


def test_two_draws_that_meet_on_a_table_and_a_cell_are_refused(monkeypatch):
    from dsbench.sftgen import dialect_conventions, schema

    monkeypatch.setattr(dialect_conventions, "generate", _stand_in([], table=7))
    monkeypatch.setattr(schema, "row_to_dict", lambda row: row)
    with pytest.raises(RuntimeError, match="share an id"):
        te.build_test_items(seed=100, cell_reps=1, extra_reps={"weekday-numbering": 1})


def test_the_overlap_check_finds_a_training_prompt_and_counts_names(tmp_path, drawn):
    items, _ = te.build_test_items(seed=100, cell_reps=1, extra_reps={})
    asked = items[0]["messages"]
    mixture = tmp_path / "mixture.jsonl"
    mixture.write_text(json.dumps({"messages": [*asked, {"role": "assistant", "content": "x"}]})
                       + "\n" + json.dumps({"messages": [
                           {"role": "user", "content": "SELECT * FROM ride_trips"}]}) + "\n")
    overlap = te.training_overlap(items, mixture)
    assert overlap["mixture_rows"] == 2
    assert overlap["prompts_in_training"] == 4  # the stand-in's prompt names no dialect
    assert overlap["rows_naming_a_held_out_name"]["ride_trips"] == 1
    assert overlap["rows_naming_a_held_out_name"]["pickup_ts"] == 0


# --- compare -------------------------------------------------------------------------------------


@pytest.mark.parametrize("b, c, p", [(0, 0, 1.0), (0, 10, 2 / 1024), (1, 9, 2 * 11 / 1024),
                                     (5, 5, 1.0), (9, 1, 2 * 11 / 1024)])
def test_the_exact_mcnemar_p(b, c, p):
    assert te.mcnemar_p(b, c) == pytest.approx(p)


def _item(i, dialect, family):
    return {"id": f"t{i}", "meta": {"dialect": dialect, "family": family}}


def _rec(i, status, doubts=0, tokens=100):
    return {"id": f"t{i}", "reasoning_tokens": tokens, "check": {"status": status,
                                                                 "doubts": doubts}}


def test_compare_pairs_each_item_and_counts_who_alone_got_it():
    items = {f"t{i}": _item(i, "clickhouse", "weekday-numbering") for i in range(10)}
    items |= {f"t{i}": _item(i, "mysql", "weekend-flag") for i in range(10, 14)}
    # ClickHouse: the base right on t0 only, the adapter on all but t9; t9 wrong for both
    base = {f"t{i}": _rec(i, "verified" if i == 0 else "wrong", doubts=1, tokens=900)
            for i in range(10)}
    adapter = {f"t{i}": _rec(i, "wrong" if i == 9 else "verified", tokens=300)
               for i in range(10)}
    # MySQL: both right on t10 and t11; the adapter loses t12; t13's table didn't rebuild
    base |= {"t10": _rec(10, "verified"), "t11": _rec(11, "verified"),
             "t12": _rec(12, "verified"), "t13": _rec(13, "rebuild_mismatch")}
    adapter |= {"t10": _rec(10, "verified"), "t11": _rec(11, "verified"),
                "t12": _rec(12, "unfinished"), "t13": _rec(13, "rebuild_mismatch")}
    adapter["t99"] = _rec(99, "verified")  # not an item: ignored
    del base["t11"]  # the base never answered it: unpaired
    result = te.compare(items, base, adapter)
    assert (result["paired"], result["rebuild_mismatch"]) == (12, 1)
    assert result["unpaired"] == {"base": 0, "adapter": 1}
    assert result["groups"]["target"] == {"n": 10, "base": 1, "adapter": 9, "base_only": 0,
                                          "adapter_only": 8, "p_mcnemar": round(2 / 256, 6)}
    other = result["groups"]["other cells"]
    assert (other["n"], other["base"], other["adapter"], other["base_only"]) == (2, 2, 1, 1)
    assert result["groups"]["other cells: mysql"] == other
    assert "other cells: duckdb" not in result["groups"]  # no items, no row
    assert result["cells"]["mysql/weekend-flag"] == other
    assert result["statuses"]["adapter"] == {"verified": 10, "wrong": 1, "unfinished": 1}
    assert result["target_states_sunday_1"] == {"base": 10, "adapter": 0}
    assert result["reasoning_tokens_median"]["adapter"]["target"] == 300
    table = te.markdown(result)
    assert "| target | 10 | 1 | 9 | 0 | 8 | 0.00781 |" in table


def test_a_retried_reply_replaces_its_failed_record(tmp_path):
    path = tmp_path / "verified.jsonl"
    path.write_text(json.dumps(_rec(1, "unfinished")) + "\n" + json.dumps(_rec(1, "verified"))
                    + "\n")
    assert te._records(path)["t1"]["check"]["status"] == "verified"
