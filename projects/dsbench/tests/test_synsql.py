"""Tests for Revision 2's SQL pool (ADR-004 Revision 2, action item 3): the SQLite sandbox's runner
and its variants, reading data.json at an offset, the prompt, the build filters and the
execution-match check.

Unit tests run on tiny databases made here; the live test needs the sandbox's `sqlite` container
and is skipped without it, as in CI.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import random
import re
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path

import pytest
from dsbench.sftgen import synsql

RUNNER = Path(__file__).parents[1] / "sandbox" / "sqlite" / "runner.py"


def _runner():
    spec = importlib.util.spec_from_file_location("sqlite_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_db(path: Path, rows=((1, "one"), (2, "two"), (3, "three"))) -> Path:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, blob BLOB)")
    con.executemany("INSERT INTO items VALUES (?, ?, ?)",
                    [(i, name, bytes([i, 2])) for i, name in rows])
    con.commit()
    con.close()
    return path


class InProcessSandbox(synsql.SQLiteSandbox):
    """The runner's own code, run here: for tests, whose queries are written here too."""

    def __init__(self) -> None:
        self.run_request = _runner().run

    def _request(self, request: dict, budget: int) -> dict:
        return json.loads(json.dumps(self.run_request(request)))  # as stdout would carry it


# --- the sandbox runner ----------------------------------------------------------------------

def test_the_sqlite_runner_reads_its_database_and_nothing_else(tmp_path):
    db = _make_db(tmp_path / "t.sqlite")
    queries = ["SELECT id, name FROM items ORDER BY id", "SELECT blob FROM items WHERE id = 1",
               "INSERT INTO items VALUES (9, 'nine', NULL)",
               f"ATTACH DATABASE '{tmp_path}/other.db' AS other", "SELECT 1; SELECT 2",
               "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) "
               "SELECT count(*) FROM r"]
    request = {"db": base64.b64encode(db.read_bytes()).decode(), "queries": queries,
               "timeout_s": 1, "max_rows": 2}
    first, blob, write, attach, two, endless = _runner().run(request)["results"]
    assert first == {"ok": True, "rows": [[1, "one"], [2, "two"]], "more": True}
    assert blob["rows"] == [["0x0102"]]
    assert not write["ok"] and "readonly" in write["error"]
    assert not attach["ok"] and not (tmp_path / "other.db").exists()
    assert not two["ok"]
    assert not endless["ok"] and "interrupt" in endless["error"].lower()


def _league(path: Path) -> Path:
    """Two rows a table, numbered from 0, as SynSQL's databases are."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE teams (team_id INTEGER, name TEXT NOT NULL, is_active INTEGER,
                            PRIMARY KEY (team_id));
        CREATE TABLE players (player_id INTEGER, team_id INTEGER, score REAL, joined TEXT,
                              PRIMARY KEY (player_id),
                              FOREIGN KEY (team_id) REFERENCES teams (team_id));
        CREATE TABLE player_awards (player_id INTEGER, award_id INTEGER,
                                    PRIMARY KEY (player_id, award_id));
        INSERT INTO teams VALUES (0, 'Owls', 1), (1, 'Foxes', 0);
        INSERT INTO players VALUES (0, 0, 7.5, '2023-05-01'), (1, 1, 3.25, '2023-06-15 10:00');
        INSERT INTO player_awards VALUES (0, 0), (1, 1);
    """)
    con.close()
    return path


def _table(path: Path, sql: str) -> list[tuple]:
    con = sqlite3.connect(path)
    rows = con.execute(sql).fetchall()
    con.close()
    return rows


def test_a_variant_keeps_the_rows_and_adds_its_own(tmp_path):
    src = _league(tmp_path / "league.sqlite")
    runner = _runner()
    runner.make_variant(str(src), str(tmp_path / "v.sqlite"), "league:1", 40)
    v = tmp_path / "v.sqlite"
    for table in ("teams", "players", "player_awards"):
        assert _table(v, f"SELECT count(*) FROM {table}") == [(40,)], table
    assert _table(v, "SELECT * FROM teams WHERE team_id < 2") == [(0, "Owls", 1), (1, "Foxes", 0)]
    teams = {r[0] for r in _table(v, "SELECT team_id FROM teams")}
    per_team = Counter(r[0] for r in _table(v, "SELECT team_id FROM players"))
    assert set(per_team) <= teams  # a declared foreign key points at a team
    assert max(per_team.values()) >= 5 and len(per_team) < len(teams)  # big groups, empty teams
    players = {r[0] for r in _table(v, "SELECT player_id FROM players")}
    assert {r[0] for r in _table(v, "SELECT player_id FROM player_awards")} <= players  # by name
    assert {r[0] for r in _table(v, "SELECT is_active FROM teams")} <= {0, 1, None}  # a flag
    assert len({r[0] for r in _table(v, "SELECT award_id FROM player_awards")}) > 2  # not a flag
    joined = [r[0] for r in _table(v, "SELECT joined FROM players WHERE joined IS NOT NULL")]
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}( 10:00)?", d) for d in joined)
    assert len(set(joined)) > 5  # dates move, keeping their time of day

    runner.make_variant(str(src), str(tmp_path / "again.sqlite"), "league:1", 40)
    runner.make_variant(str(src), str(tmp_path / "other.sqlite"), "league:2", 40)
    everything = "SELECT * FROM players ORDER BY player_id"
    assert _table(tmp_path / "again.sqlite", everything) == _table(v, everything)
    assert _table(tmp_path / "other.sqlite", everything) != _table(v, everything)


def test_the_runner_answers_on_each_variant_and_reports_one_it_cannot_build(tmp_path):
    league = _league(tmp_path / "league.sqlite")
    request = {"db": base64.b64encode(league.read_bytes()).decode(),
               "queries": ["SELECT count(*) FROM players"],
               "variants": [{"seed": "a", "rows": 10}, {"seed": "b", "rows": 25}]}
    out = _runner().run(request)
    assert [r[0]["rows"] for r in [out["results"], *out["variants"]]] == [[[2]], [[10]], [[25]]]

    generated = tmp_path / "generated.sqlite"  # a generated column can't be copied row by row
    con = sqlite3.connect(generated)
    con.executescript("CREATE TABLE g (a INTEGER, b INTEGER GENERATED ALWAYS AS (a * 2));"
                      "INSERT INTO g (a) VALUES (1);")
    con.close()
    request = {"db": base64.b64encode(generated.read_bytes()).decode(),
               "queries": ["SELECT a FROM g", "SELECT b FROM g"],
               "variants": [{"seed": "a", "rows": 5}]}
    out = _runner().run(request)
    assert [r["ok"] for r in out["results"]] == [True, True]
    assert all(not r["ok"] and r["error"].startswith("variant:") for r in out["variants"][0])


# --- reading data.json -----------------------------------------------------------------------

def test_first_item_takes_the_next_whole_item():
    items = [{"db_id": "a", "question": "q1", "sql": "SELECT 1"},
             {"db_id": "b", "question": "q2", "sql": "SELECT 2"}]
    text = json.dumps(items, indent=2)
    assert synsql.first_item(text) == items[0]
    assert synsql.first_item(text[5:]) == items[1]  # an offset inside the first item
    assert synsql.first_item(text[5: text.index('"q2"')]) is None  # the next one is cut off


# --- the prompt ------------------------------------------------------------------------------

def test_the_prompt_shows_the_schema_names_sqlite_and_asks_for_the_query_only(tmp_path):
    db = _make_db(tmp_path / "t.sqlite")
    row = {"question": " How many items are there? ", "external_knowledge": "An item is a row."}
    seen = set()
    for seed in range(12):
        prompt, k = synsql.build_prompt(row, synsql._bird_schema(db), random.Random(seed))
        seen.add(k)
        assert "CREATE TABLE items" in prompt and "3 example rows" in prompt
        assert "SQLite" in prompt and "```sql" in prompt
        assert "How many items are there?" in prompt and "Background: An item is a row." in prompt
    assert seen == set(range(len(synsql.INSTRUCTIONS)))
    bare, _ = synsql.build_prompt({**row, "external_knowledge": ""}, "schema", random.Random(0))
    assert "Background" not in bare


# --- building the pool -----------------------------------------------------------------------

@pytest.fixture
def zone(tmp_path):
    """A databases.zip laid out as SynSQL's, with four one-table databases."""
    zip_path = tmp_path / "databases.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for name in ("shop", "zoo", "farm", "club"):
            db = _make_db(tmp_path / f"{name}.sqlite")
            z.write(db, f"databases/{name}/{name}.sqlite")
    return synsql.Databases(zip_path, tmp_path / "zone")


ROWS = [
    {"db_id": "shop", "question": "How many items?", "sql": "SELECT count(*) FROM items",
     "question_style": "Formal", "sql_complexity": "Simple", "external_knowledge": ""},
    {"db_id": "zoo", "question": "Items over 100?", "sql": "SELECT * FROM items WHERE id > 100",
     "question_style": "Formal", "sql_complexity": "Simple", "external_knowledge": ""},
    {"db_id": "farm", "question": "Broken?", "sql": "SELEC name FROM items",
     "question_style": "Vague", "sql_complexity": "Simple", "external_knowledge": ""},
    {"db_id": "club", "question": "**User**: names?", "sql": "SELECT name FROM items",
     "question_style": "Multi-turn Dialogue", "sql_complexity": "Simple",
     "external_knowledge": ""},
    {"db_id": "club", "question": "Which names?", "sql": "SELECT name FROM items",
     "question_style": "Imperative", "sql_complexity": "Moderate", "external_knowledge": ""},
    {"db_id": "zoo", "question": "Count and mean id over 100?",
     "sql": "SELECT count(*), avg(id) FROM items WHERE id > 100", "question_style": "Concise",
     "sql_complexity": "Moderate", "external_knowledge": ""},
]


def _reader(*rows):
    """Candidates in this order, whatever the offset."""
    order = iter(rows)
    return lambda offset: next(order)


def test_build_keeps_questions_whose_gold_sql_checks_something(zone):
    shop, zoo, farm, club_dialogue, club, zoo_blank = ROWS
    items, report = synsql.build(reader=_reader(shop, shop, zoo, zoo_blank, farm, club_dialogue,
                                                club),
                                 size=synsql.CHUNK + 10_000, databases=zone,
                                 sandbox=InProcessSandbox(), gate=None, n=2, seed=3)
    assert sorted(i["meta"]["db_id"] for i in items) == ["club", "shop"]
    assert report["rejected"] == {"database already used": 1, "gold returns no rows": 1,
                                  "gold returns only NULL, 0 or ''": 1, "gold fails": 1,
                                  "style Multi-turn Dialogue": 1}
    shop = next(i for i in items if i["meta"]["db_id"] == "shop")
    assert shop["verify"] == {"kind": "sqlite_ex", "db_id": "shop",
                              "gold_sql": "SELECT count(*) FROM items",
                              "variants": synsql.VARIANTS, "variant_rows": synsql.VARIANT_ROWS,
                              "gold_rows": [1] * (1 + synsql.VARIANTS)}
    assert shop["id"].startswith("synsql:shop:") and shop["meta"]["licence"] == "Apache-2.0"
    assert report["read_by_complexity"] == {"Simple": 5, "Moderate": 2}
    assert report["by_complexity"] == {"Simple": 1, "Moderate": 1}
    assert report["rejected_by_complexity"]["Moderate"] == {"gold returns only NULL, 0 or ''": 1}


def test_the_gold_must_return_something_somewhere():
    ran = {"ok": True, "more": False}
    empty, failed = {**ran, "rows": []}, {"ok": False, "error": "x"}
    blank = synsql.gold_check([{**ran, "rows": [[0, None]]}, {**ran, "rows": [[0.0, ""]]}])
    assert blank == "gold returns only NULL, 0 or ''"
    assert synsql.gold_check([{**ran, "rows": [[0, "0"]]}]) is None  # "0" is a value
    assert synsql.gold_check([empty, empty]) == "gold returns no rows"
    assert synsql.gold_check([empty, {**ran, "rows": [[4]]}]) is None  # a variant can check
    assert synsql.gold_check([{**ran, "rows": [[4]]}, failed]) == "gold fails on a variant"
    assert synsql.gold_check([failed, {**ran, "rows": [[4]]}]) == "gold fails"
    assert synsql.gold_check([{**ran, "rows": [[4]]}, {**ran, "rows": [], "more": True}]) == (
        f"gold returns over {synsql.MAX_ROWS} rows")


def test_a_gate_rejection_is_counted(zone):
    def gate(item):
        return "battery overlap bfcl:1" if "How many" in item["messages"][0]["content"] else None

    items, report = synsql.build(reader=_reader(ROWS[0], ROWS[4]), size=synsql.CHUNK + 10_000,
                                 databases=zone, sandbox=InProcessSandbox(), n=1, seed=3,
                                 gate=gate)
    assert items[0]["meta"]["db_id"] == "club"
    assert report["rejected"] == {"gate: battery": 1}
    assert report["gate_hits"] == [{"id": synsql.make_item(ROWS[0], "", 0)["id"],
                                    "reason": "battery overlap bfcl:1"}]


# --- the check -------------------------------------------------------------------------------

def test_same_rows_compares_sets_as_bird_does():
    assert synsql.same_rows([[2, "b"], [1, "a"]], [[1, "a"], [2, "b"]])
    assert synsql.same_rows([[1, "a"], [1, "a"]], [[1, "a"]])  # BIRD's EX: set against set
    assert synsql.same_rows([[1]], [[1.0]])
    assert not synsql.same_rows([[1, "a"]], [[1, "a"], [2, "b"]])


def test_verify_runs_the_reply_s_sql_against_the_gold(tmp_path, zone):
    item = {"id": "synsql:shop:x", "pool": "synsql",
            "verify": {"kind": "sqlite_ex", "db_id": "shop", "variants": 2, "variant_rows": 30,
                       "gold_sql": "SELECT name FROM items WHERE id < 3"}}
    base = {"id": item["id"], "reasoning": "Two rows.", "finish_reason": "stop"}
    gens = [
        {**base, "answer": "```sql\nSELECT name FROM items WHERE id IN (2, 1) ORDER BY 1\n```"},
        {**base, "answer": "```sql\nSELECT name FROM items\n```"},
        {**base, "answer": "```sql\nSELECT nme FROM items\n```"},
        {**base, "answer": "```sql\nSELECT name FROM items WHERE id < 3\n```", "reasoning": ""},
        {**base, "answer": "", "finish_reason": "length"},
        {**base, "answer": "Here:\n```sql\nSELECT name FROM items WHERE id < 3\n```"},
        {**base, "answer": "```sql\nSELECT 1\n```\nor\n```sql\nSELECT 2\n```"},
        {**base, "answer": "SELECT name FROM items WHERE id < 3"},
        {**base, "answer": "```sql\nSELECT name FROM items WHERE id > 1\n```"},
        {**base, "answer": "```sql\nSELECT name FROM items WHERE id != 3\n```"},  # 3 rows only
    ]
    # one item per reply: a checker reads one row an item (reasoning_pilot.latest_rows)
    gens = [{**g, "id": f"{item['id']}#{k}"} for k, g in enumerate(gens)]
    (tmp_path / "items.jsonl").write_text("".join(json.dumps({**item, "id": g["id"]}) + "\n"
                                                  for g in gens))
    (tmp_path / "gen.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gens))
    result = synsql.verify(tmp_path / "items.jsonl", tmp_path / "gen.jsonl",
                           tmp_path / "out.jsonl", zone, InProcessSandbox())
    assert result == {"passed": {"synsql": 1}, "failed": {"synsql": 9}, "caught_by_variants": 1}
    whys = [json.loads(line)["check"]["why"] for line in (tmp_path / "out.jsonl").open()]
    assert whys[0] == "" and whys[1] == "3 rows, gold 2" and whys[2].startswith("error")
    assert whys[3:] == ["no reasoning", "unfinished: length", "not the SQL only",
                        "not the SQL only", "no SQL block", "other values",
                        "variant 1: 29 rows, gold 2"]


# --- live: the sandbox's sqlite container ----------------------------------------------------

def test_the_sqlite_container_runs_queries_read_only(tmp_path):
    sandbox = synsql.SQLiteSandbox()
    if not sandbox.available():
        pytest.skip("sqlite sandbox service is not running")
    db = _make_db(tmp_path / "t.sqlite")
    ok, write = sandbox.run(db, ["SELECT count(*) FROM items", "DELETE FROM items"])
    assert ok == {"ok": True, "rows": [[3]], "more": False}
    assert not write["ok"]
    suite = sandbox.run_suite(db, "t", ["SELECT count(*) FROM items"], variants=2, rows=10)
    assert [r[0]["rows"] for r in suite] == [[[3]], [[10]], [[10]]]
