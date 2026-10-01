"""Tests for Target A's hint-conditioned generation (ADR-004 Revision 2, action item 2): the hints
against the generator's own conventions, the citation filter, the item pairs, and `verify` on
in-process DuckDB. The live test checks the hints on the sandbox's engines and is skipped without
them, as in CI.
"""
from __future__ import annotations

import json

import pytest
from dsbench.sftgen import target_a_hints as th
from dsbench.sftgen.conventions import _case_utc_add, _dow_expr, _dow_int, _month_expr
from dsbench.sftgen.dialect_conventions import generate
from dsbench.sftgen.engines import DuckDBEngine, available_engines
from dsbench.sftgen.schema import row_to_dict
from dsbench.sftgen.synth import US_CITY_UTC_OFFSETS

# --- the hints -----------------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", th.DIALECTS)
def test_each_hint_states_the_numbering_the_generator_uses(dialect):
    used = th.WEEKDAY_FUNCTIONS[dialect][0]  # the function the gold SQL calls comes first
    assert used.shown == _dow_expr(dialect, "date")
    assert [th._SCHEMES[used.scheme](w) for w in range(7)] == [_dow_int(dialect, w)
                                                                 for w in range(7)]
    assert th.MONTH_FUNCTIONS[dialect].shown == _month_expr(dialect, "date")
    assert used.shown in th.hint("weekend-flag", dialect)
    assert th.hint("weekend-flag", dialect) == th.hint("weekday-numbering", dialect)


def test_the_numbering_is_written_from_its_lowest_number():
    assert th.hint("weekday-numbering", "clickhouse") == (
        "In ClickHouse, toDayOfWeek(date) numbers Monday 1 through Sunday 7.")
    assert "DAYOFWEEK(date) numbers Sunday 1 through Saturday 7" in th.hint(
        "weekday-numbering", "mysql")


def test_the_timezone_hint_is_the_direction_the_truth_uses():
    assert "plus 4 hours" in th.TIMEZONE_HINT and "UTC-4" in th.TIMEZONE_HINT
    # The family's gold SQL adds -offset to local time: 4 for a city at UTC-4.
    for offsets in US_CITY_UTC_OFFSETS.values():
        case = _case_utc_add("city", offsets)
        for city, offset in offsets.items():
            assert f"WHEN '{city}' THEN {-offset}" in case


def test_the_checks_cover_every_day_and_three_months():
    checks = th.hint_checks("clickhouse")
    assert [expected for _, expected in checks] == [1, 2, 3, 4, 5, 6, 7, 1, 7, 12]
    assert checks[6][0] == "SELECT toDayOfWeek(toDate('2024-01-07'))"  # a Sunday
    assert len(th.hint_checks("mysql")) == 7 + 7 + 3


def test_the_hints_hold_on_the_sandboxed_engines():
    engines = {e.name: e for e in available_engines(sandboxed=True)}
    if set(th.DIALECTS) - set(engines):
        pytest.skip("the sandbox's four engines are not all running")
    assert th.check_hints(engines) == {"clickhouse": 10, "duckdb": 17, "postgres": 17,
                                       "mysql": 17}


# --- the citation filter -------------------------------------------------------------------------

CH_HINT = th.hint("weekend-flag", "clickhouse")
CH_TERMS = th.hint_terms("weekend-flag", "clickhouse")
PROMPT = ("You are writing SQL for a ClickHouse database. Answer with a single SQL query in a "
          "```sql code block.\nCount the weekend orders (Saturday or Sunday order_ts).")


@pytest.mark.parametrize("text, cites", [
    ("The hint says the week starts on Monday.", True),
    # From the pilot's first pass, where the hint sat plainly in the system prompt:
    ('Yes, that matches the prompt: "toDayOfWeek(date) numbers Monday 1 through Sunday 7".', True),
    ("The prompt explicitly says `toDayOfWeek(date)` numbers Monday 1.", True),
    ("`IN (6, 7)` is explicit and matches the prompt's hint.", True),
    ("I'll treat this as my own knowledge: toDayOfWeek is ISO.", True),
    ("The weekend is 6 and 7; toDayOfWeek(date) numbers Monday 1 through Sunday 7.", True),  # 8
    # ...and from traces that never saw a hint:
    ('The prompt explicitly says "by order_ts", so filter on order_ts with toDayOfWeek.', False),
    ("The time zone is not mentioned, so toDayOfWeek works on the stored value.", False),
    ("As stated, toDayOfWeek numbers Monday 1, so the weekend is 6 and 7.", True),
    ("The system prompt says toDayOfWeek is ISO.", True),
    ("We’re told that toDayOfWeek returns 7 for Sunday.", True),
    ("In ClickHouse, toDayOfWeek(date) numbers Monday 1 through Sunday 7.", True),  # verbatim
    ("I recall ClickHouse's toDayOfWeek is ISO: Monday is 1 and Sunday is 7.", False),
    ("The system prompt asks for a single query. toDayOfWeek is ISO, so 6 and 7.", False),
    ("Answer with a single SQL query in a sql code block, so no prose.", False),  # the prompt's
])
def test_a_reply_cites_the_hint_only_when_it_reads_as_told(text, cites):
    assert (th.cites_hint(text, CH_HINT, CH_TERMS, PROMPT) is not None) is cites


def test_the_base_quoting_the_question_is_not_a_citation():
    # From the reasoning pilot's unhinted traces: the function is the reply's own SQL.
    said = 'The prompt says "month of April", so `toMonth(order_ts) = 4` is correct.'
    terms = th.hint_terms("month-bucket", "clickhouse")
    assert th.cites_hint(said, th.hint("month-bucket", "clickhouse"), terms, "month of April") \
        is None
    assert th.cites_hint('The prompt says toMonth is 1-based, so April is 4.',
                         th.hint("month-bucket", "clickhouse"), terms, "April") is not None


def test_the_question_s_own_premise_may_be_quoted():
    terms = th.hint_terms("timezone-direction", "duckdb")
    said = "We're told New York is UTC-4, as stated in the question, so add 4 hours."
    assert th.cites_hint(said, th.TIMEZONE_HINT, terms, "New York UTC-4") is None
    assert th.soft_flags(said) == ["We're told", "as stated"]  # kept for review
    borrowed = "As stated, a clock at UTC-4 is 4 hours behind UTC."
    assert th.cites_hint(borrowed, th.TIMEZONE_HINT, terms, "New York UTC-4") is not None


# --- items ---------------------------------------------------------------------------------------

def _row(**over) -> dict:
    row = {"id": "A-weekend-flag-clickhouse-retail_orders-7", "family": "weekend-flag",
           "dialect": "clickhouse",
           "turns": [{"content": "You are writing SQL for a ClickHouse database. There is one "
                                 "table `t`. Answer with a single SQL query in a ```sql code "
                                 "block."},
                     {"content": "Count the weekend rows."},
                     {"content": "```sql\nSELECT count(*) FROM t WHERE toDayOfWeek(ts) IN "
                                 "(6, 7)\n```"}],
           "verification": {"truth": 5}}
    return {**row, **over}


def test_a_row_becomes_a_plain_item_and_its_hinted_twin():
    plain, hinted = th.make_items(_row(), n=200)
    assert plain["id"] == "targetA:A-weekend-flag-clickhouse-retail_orders-7"
    assert hinted["id"] == "targetA_hint:A-weekend-flag-clickhouse-retail_orders-7"
    assert hinted["train_messages"] == plain["messages"]  # the row trains without the hint
    system = hinted["messages"][0]["content"]
    assert system == (f"{plain['messages'][0]['content']}\n\n{th.FRAMING} {CH_HINT}")
    assert plain["verify"]["gold_sql"] == "SELECT count(*) FROM t WHERE toDayOfWeek(ts) IN (6, 7)"
    assert plain["verify"]["n"] == 200 and "hint" not in plain["verify"]
    assert hinted["verify"]["hint"] == CH_HINT
    assert (plain["meta"]["hinted"], hinted["meta"]["hinted"]) == (False, True)
    assert (plain["meta"]["domain"], plain["meta"]["seed"]) == ("retail_orders", 7)
    assert th.make_items(_row(), n=200, hinted=False) == [plain]


# --- verify --------------------------------------------------------------------------------------

@pytest.fixture
def duckdb_pair(tmp_path):
    rows, _ = generate(seed=5, reps=1, n=200, dialects=["duckdb"], families=["weekend-flag"],
                       thinking_frac=0.0)
    plain, hinted = th.make_items(row_to_dict(rows[0]), n=200)
    (tmp_path / "items.jsonl").write_text(json.dumps(plain) + "\n" + json.dumps(hinted) + "\n")
    return plain, hinted


def test_verify_keeps_a_verified_reply_that_does_not_read_as_told(tmp_path, duckdb_pair):
    plain, hinted = duckdb_pair
    gold = plain["verify"]["gold_sql"]
    assert gold.endswith("IN (6, 0)")  # DuckDB's dayofweek: Saturday 6, Sunday 0
    another_engines = gold.replace("IN (6, 0)", "IN (6, 7)")  # ClickHouse's numbers

    def rec(item, sql=gold, reasoning="Sunday is 0 and Saturday is 6.", finish="stop",
            prose=""):
        return {"id": item["id"], "reasoning": reasoning, "finish_reason": finish,
                "answer": f"{prose}```sql\n{sql}\n```", "reasoning_tokens": 40}

    gens = [
        rec(plain),
        rec(plain, finish="length"),
        rec(plain, reasoning=""),
        rec(plain, sql="SELEC count(*) FROM nowhere"),
        rec(hinted),
        rec(hinted, reasoning="We're told dayofweek numbers Sunday 0. So 0 and 6."),
        rec(hinted, sql=another_engines),
        rec(hinted, prose="Here is the query:\n"),
    ]
    (tmp_path / "gen.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gens))
    summary = th.verify(tmp_path / "items.jsonl", tmp_path / "gen.jsonl", tmp_path / "out.jsonl",
                        engines={"duckdb": DuckDBEngine()})
    cell = summary["duckdb/weekend-flag"]
    assert cell["plain"] == {"replies": 4, "verified": 1, "unfinished": 1, "no_reasoning": 1,
                             "error": 1, "cites": 0, "soft_flagged": 0, "kept": 1,
                             "reasoning_tokens_median": 40}
    assert cell["hinted"] == {"replies": 4, "verified": 2, "wrong": 1, "not_sql_only": 1,
                              "cites": 1, "soft_flagged": 1, "kept": 1,
                              "reasoning_tokens_median": 40}
    out = [json.loads(line) for line in (tmp_path / "out.jsonl").open()]
    kept = [r for r in out if r["check"]["kept"]]
    assert all(r["train_messages"] == plain["messages"] for r in kept)  # never the hint
