"""Dialect execution backends for Target A -- run a query against a REAL engine and read one scalar.

The point of Target A is dialect-AWARENESS, so the data must be verified against each real engine,
not against a mental model of it. Each engine loads a synthetic table and evaluates one scalar
query; the generator keeps a row only if that scalar equals the independent pandas truth (ADR-004).

Availability is graceful: DuckDB is in-process and always present; ClickHouse is used if the
sandbox container is reachable; Postgres/MySQL are used only if their driver AND a DSN env var are
present (`SFTGEN_PG_DSN`, `SFTGEN_MYSQL_DSN`). A dialect with no reachable engine simply produces no
rows this run -- we never emit an unverified row.
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import pandas as pd


def _norm(v: Any) -> Any:
    """Normalise an engine scalar so int/float compare cleanly across drivers."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return round(float(v), 6)
    return v


class Engine:
    """Interface: setup() -> load()* -> scalar()* -> teardown()."""

    name: str = "abstract"

    def available(self) -> bool:  # noqa: D102
        return False

    def setup(self) -> None:  # noqa: D102
        ...

    def load(self, table: str, df: pd.DataFrame) -> None:  # noqa: D102
        raise NotImplementedError

    def scalar(self, sql: str) -> Any:  # noqa: D102
        raise NotImplementedError

    def teardown(self) -> None:  # noqa: D102
        ...


class DuckDBEngine(Engine):
    """In-process; the always-available anchor. DataFrames register as zero-copy views."""

    name = "duckdb"

    def __init__(self) -> None:
        self._con = None

    def available(self) -> bool:
        try:
            import duckdb  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def setup(self) -> None:
        import duckdb

        self._con = duckdb.connect()

    def load(self, table: str, df: pd.DataFrame) -> None:
        # register makes the DataFrame queryable as a view; drop any prior binding first.
        if self._con is not None:
            self._con.unregister(table)
        self._con.register(table, df)

    def scalar(self, sql: str) -> Any:
        row = self._con.execute(sql).fetchone()
        return _norm(row[0]) if row else None

    def teardown(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None


def _ch_type(dtype: Any) -> str:
    """Map a pandas dtype to a ClickHouse column type for the scratch table."""
    s = str(dtype)
    if s.startswith("datetime"):
        return "DateTime"
    if s.startswith("int") or s.startswith("uint"):
        return "Int64"
    if s.startswith("float"):
        return "Float64"
    if s == "bool":
        return "UInt8"
    return "String"


class ClickHouseEngine(Engine):
    """Sandbox ClickHouse (ADR-003). Uses a throwaway scratch database, dropped on teardown."""

    name = "clickhouse"

    def __init__(self) -> None:
        self._client = None
        self._db = f"sftgen_{uuid.uuid4().hex[:10]}"

    def available(self) -> bool:
        try:
            from dsbench.agentic.ch import get_client

            c = get_client(database="default")
            c.command("SELECT 1")
            c.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    def setup(self) -> None:
        from dsbench.agentic.ch import get_client

        admin = get_client(database="default")
        admin.command(f"CREATE DATABASE IF NOT EXISTS {self._db}")
        admin.close()
        self._client = get_client(database=self._db)

    def load(self, table: str, df: pd.DataFrame) -> None:
        cols = ", ".join(f"`{c}` {_ch_type(dt)}" for c, dt in df.dtypes.items())
        self._client.command(f"DROP TABLE IF EXISTS {table}")
        self._client.command(f"CREATE TABLE {table} ({cols}) ENGINE = MergeTree ORDER BY tuple()")
        self._client.insert_df(table, df)

    def scalar(self, sql: str) -> Any:
        res = self._client.query(sql)
        return _norm(res.result_rows[0][0]) if res.result_rows else None

    def teardown(self) -> None:
        if self._client is not None:
            self._client.command(f"DROP DATABASE IF EXISTS {self._db}")
            self._client.close()
            self._client = None


class _DSNEngine(Engine):
    """Shared base for the optional server engines (Postgres, MySQL). Inactive unless configured."""

    name = "abstract-dsn"
    _dsn_env = ""
    _driver = ""

    def __init__(self) -> None:
        self._con = None

    def _dsn(self) -> str | None:
        return os.environ.get(self._dsn_env)

    def available(self) -> bool:
        if not self._dsn():
            return False
        try:
            __import__(self._driver)
            return True
        except Exception:  # noqa: BLE001
            return False


class PostgresEngine(_DSNEngine):
    """Optional: needs `psycopg` and `SFTGEN_PG_DSN`. EXTRACT(DOW) is 0=Sunday."""

    name = "postgres"
    _dsn_env = "SFTGEN_PG_DSN"
    _driver = "psycopg"

    def setup(self) -> None:
        import psycopg

        self._con = psycopg.connect(self._dsn(), autocommit=True)

    def load(self, table: str, df: pd.DataFrame) -> None:
        cur = self._con.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cols = ", ".join(f"{c} {_pg_type(dt)}" for c, dt in df.dtypes.items())
        cur.execute(f"CREATE TABLE {table} ({cols})")
        ph = ", ".join(["%s"] * len(df.columns))
        cur.executemany(
            f"INSERT INTO {table} VALUES ({ph})",
            [tuple(_py(v) for v in rec) for rec in df.itertuples(index=False, name=None)],
        )

    def scalar(self, sql: str) -> Any:
        cur = self._con.cursor()
        cur.execute(sql)
        row = cur.fetchone()
        return _norm(row[0]) if row else None

    def teardown(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None


class MySQLEngine(_DSNEngine):
    """Optional: needs `pymysql` and `SFTGEN_MYSQL_DSN` (host/user/password/db as a JSON dict)."""

    name = "mysql"
    _dsn_env = "SFTGEN_MYSQL_DSN"
    _driver = "pymysql"

    def setup(self) -> None:
        import json

        import pymysql

        cfg = json.loads(self._dsn())
        self._con = pymysql.connect(autocommit=True, **cfg)

    def load(self, table: str, df: pd.DataFrame) -> None:
        cur = self._con.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cols = ", ".join(f"`{c}` {_mysql_type(dt)}" for c, dt in df.dtypes.items())
        cur.execute(f"CREATE TABLE {table} ({cols})")
        ph = ", ".join(["%s"] * len(df.columns))
        cur.executemany(
            f"INSERT INTO {table} VALUES ({ph})",
            [tuple(_py(v) for v in rec) for rec in df.itertuples(index=False, name=None)],
        )

    def scalar(self, sql: str) -> Any:
        cur = self._con.cursor()
        cur.execute(sql)
        row = cur.fetchone()
        return _norm(row[0]) if row else None

    def teardown(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None


def _py(v: Any) -> Any:
    """Coerce a pandas/numpy value to a plain Python type for DB-API binding."""
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:  # noqa: BLE001
            return v
    return v


def _pg_type(dtype: Any) -> str:
    s = str(dtype)
    if s.startswith("datetime"):
        return "timestamp"
    if s.startswith("int") or s.startswith("uint"):
        return "bigint"
    if s.startswith("float"):
        return "double precision"
    if s == "bool":
        return "boolean"
    return "text"


def _mysql_type(dtype: Any) -> str:
    s = str(dtype)
    if s.startswith("datetime"):
        return "datetime"
    if s.startswith("int") or s.startswith("uint"):
        return "bigint"
    if s.startswith("float"):
        return "double"
    if s == "bool":
        return "tinyint"
    return "varchar(255)"


# Registry in the ADR-004 order (ClickHouse first: it is the measured-gap dialect).
_ALL: tuple[type[Engine], ...] = (ClickHouseEngine, DuckDBEngine, PostgresEngine, MySQLEngine)


def available_engines(only: list[str] | None = None) -> list[Engine]:
    """Instantiate the engines that are reachable right now (optionally filtered by name)."""
    out: list[Engine] = []
    for cls in _ALL:
        eng = cls()
        if only and eng.name not in only:
            continue
        if eng.available():
            out.append(eng)
    return out
