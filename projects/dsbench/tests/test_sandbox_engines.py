"""Tests for the engines model-written SQL runs on (ADR-004 Revision 2).

Two kinds: unit tests that need nothing running, and live tests against the dsbench sandbox
(`sandbox/docker-compose.yml`: postgres, mysql, duckdb, clickhouse), skipped when a service is down,
as in CI. Live, every family's gold SQL must return the pandas truth on every engine, and SQL that
reaches outside its table must fail.
"""

from __future__ import annotations

import base64
import importlib.util
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dsbench.sftgen import engines as eng_mod
from dsbench.sftgen import synth
from dsbench.sftgen.conventions import ALL_CONVENTIONS

RUNNER = Path(__file__).parents[1] / "sandbox" / "duckdb" / "runner.py"


def _runner():
    spec = importlib.util.spec_from_file_location("duckdb_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(sql: str, table: str = "t") -> dict:
    buffer = io.BytesIO()
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(buffer, index=False)
    return {"table": table, "parquet": base64.b64encode(buffer.getvalue()).decode(), "sql": sql}


# --- unit: nothing running ----------------------------------------------------------------------


def test_the_duckdb_runner_reads_its_table_and_nothing_else(tmp_path):
    run = _runner().run
    assert run(_request("SELECT sum(a) FROM t")) == {"ok": True, "value": 6}
    for sql in (f"COPY t TO '{tmp_path}/leak.csv'",
                f"SELECT count(*) FROM read_csv('{RUNNER}')",
                "SET enable_external_access = true",
                f"ATTACH '{tmp_path}/other.db'"):
        reply = run(_request(sql))
        assert reply["ok"] is False, sql
    assert not (tmp_path / "leak.csv").exists()
    assert run(_request("SELECT 1", table="t; DROP"))["ok"] is False


def test_sandboxed_engines_never_include_in_process_duckdb(monkeypatch):
    for cls in (eng_mod.ClickHouseEngine, eng_mod.DuckDBEngine, eng_mod.SandboxDuckDBEngine,
                eng_mod.PostgresEngine, eng_mod.MySQLEngine):
        monkeypatch.setattr(cls, "available", lambda self: True)
    sandboxed = eng_mod.available_engines(sandboxed=True)
    assert [type(e) for e in sandboxed] == [eng_mod.ClickHouseEngine, eng_mod.SandboxDuckDBEngine,
                                            eng_mod.PostgresEngine, eng_mod.MySQLEngine]
    assert eng_mod.DuckDBEngine in [type(e) for e in eng_mod.available_engines()]


# --- live: the dsbench sandbox --------------------------------------------------------------------


def _live(cls):
    engine = cls()
    if not engine.available():
        pytest.skip(f"{cls.name} sandbox service is not running")
    return engine


LIVE = [eng_mod.PostgresEngine, eng_mod.MySQLEngine, eng_mod.SandboxDuckDBEngine,
        eng_mod.ClickHouseEngine]


@pytest.mark.parametrize("cls", LIVE, ids=lambda c: c.__name__)
def test_every_familys_gold_sql_returns_the_truth(cls):
    engine = _live(cls)
    domain = synth.build("support_tickets", seed=20260929, n=400)
    rng = np.random.default_rng(3)
    engine.setup()
    try:
        engine.load(domain.name, domain.df)
        for conv in ALL_CONVENTIONS:
            for _ in range(3):
                params = conv.params(rng)
                sql = conv.sql(domain, engine.name, params)
                assert engine.scalar(sql) == conv.truth(domain, params), (conv.family, sql)
    finally:
        engine.teardown()


HOSTILE = {
    "PostgresEngine": ["COPY (SELECT 1) TO '/tmp/leak'", "SELECT pg_read_file('/etc/passwd')",
                       "CREATE TABLE extra (a int)"],
    "MySQLEngine": ["SELECT 1 INTO OUTFILE '/tmp/leak'", "CREATE TABLE extra (a int)"],
    "SandboxDuckDBEngine": ["COPY support_tickets TO '/tmp/leak.csv'",
                            "SELECT count(*) FROM read_csv('/etc/passwd')"],
}


@pytest.mark.parametrize("cls", LIVE[:3], ids=lambda c: c.__name__)
def test_sql_reaching_outside_its_table_fails(cls):
    engine = _live(cls)
    domain = synth.build("support_tickets", seed=1, n=20)
    engine.setup()
    try:
        engine.load(domain.name, domain.df)
        for sql in HOSTILE[cls.__name__]:
            with pytest.raises(Exception):  # noqa: B017 - drivers raise their own error types
                engine.scalar(sql)
        if cls is eng_mod.MySQLEngine:  # without FILE, LOAD_FILE reads nothing
            assert engine.scalar("SELECT LOAD_FILE('/etc/passwd')") is None
    finally:
        engine.teardown()
