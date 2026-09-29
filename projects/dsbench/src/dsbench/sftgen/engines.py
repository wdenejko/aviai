"""Dialect execution backends for Target A -- run a query against a REAL engine and read one scalar.

The point of Target A is dialect-AWARENESS, so the data must be verified against each real engine,
not against a mental model of it. Each engine loads a synthetic table and evaluates one scalar
query; the generator keeps a row only if that scalar equals the independent pandas truth (ADR-004).

Two kinds of SQL run here, and they get different engines (ADR-004 Revision 2):
  * our own gold SQL (`dialect_conventions.py`): any engine, DuckDB in-process included;
  * SQL a model wrote (`reasoning_pilot.verify`, the retrain's Target A verification): only engines
    inside the dsbench sandbox, `available_engines(sandboxed=True)`. In-process DuckDB is left out
    there, because DuckDB SQL can read and write the host's files and fetch URLs; its twin runs in
    a container with no network (`sandbox/duckdb`).

Availability is graceful: an engine is used if it answers now. ClickHouse, Postgres, MySQL and the
DuckDB container are services of the dsbench sandbox (`sandbox/docker-compose.yml`). A dialect with
no reachable engine simply produces no rows this run -- we never emit an unverified row.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
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


class SandboxDuckDBEngine(Engine):
    """DuckDB in the sandbox's `duckdb` container, for SQL a model wrote.

    Each query runs in a fresh process there (`docker exec ... python /runner.py`). The table goes
    in as Parquet on stdin, and the runner switches external access off once it is loaded. The
    process has no network, a read-only filesystem and no host mounts (`sandbox/duckdb`).
    """

    name = "duckdb"
    container = "avbench-duckdb"
    timeout_s = 30

    def __init__(self) -> None:
        self._table: str | None = None
        self._parquet: str | None = None

    def available(self) -> bool:
        try:
            out = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container],
                capture_output=True, text=True, timeout=10,
            )
            return out.stdout.strip() == "true"
        except Exception:  # noqa: BLE001
            return False

    def load(self, table: str, df: pd.DataFrame) -> None:
        frame = df.copy()
        for col, dtype in frame.dtypes.items():
            if str(dtype).startswith("datetime64"):  # TIMESTAMP, as in-process DuckDB reads it
                frame[col] = frame[col].astype("datetime64[us]")
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        self._table, self._parquet = table, base64.b64encode(buffer.getvalue()).decode()

    def scalar(self, sql: str) -> Any:
        request = json.dumps({"table": self._table, "parquet": self._parquet, "sql": sql})
        proc = subprocess.run(
            ["docker", "exec", "-i", self.container,
             "timeout", str(self.timeout_s), "python", "/runner.py"],
            input=request, capture_output=True, text=True, timeout=self.timeout_s + 30,
        )
        try:
            reply = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"duckdb sandbox exit {proc.returncode}: "
                               f"{proc.stderr.strip()[-300:]}") from None
        if not reply["ok"]:
            raise RuntimeError(reply["error"])
        return _norm(reply["value"])


# The login model-written SQL runs as, in the sandbox's Postgres and MySQL (their init/ scripts).
READER = ("sftgen_reader", "sftgen_reader")
# The sandbox's admin logins (sandbox/docker-compose.yml); the env vars override them.
PG_ADMIN_DSN = "postgresql://avbench:avbench@127.0.0.1:55432/sftgen"
MYSQL_ADMIN = {"host": "127.0.0.1", "port": 53306, "user": "root", "password": "avbench-root"}


class PostgresEngine(Engine):
    """Postgres in the sandbox (EXTRACT(DOW) is 0=Sunday).

    Tables load through the admin login into a scratch schema, dropped on teardown. Queries run
    through `sftgen_reader`, which can read that schema and nothing else, and whose statements stop
    after 30 s. `SFTGEN_PG_DSN` overrides the admin DSN.
    """

    name = "postgres"

    def __init__(self) -> None:
        self._admin = None
        self._reader = None
        self._schema = f"sftgen_{uuid.uuid4().hex[:10]}"

    @staticmethod
    def _dsn() -> str:
        return os.environ.get("SFTGEN_PG_DSN", PG_ADMIN_DSN)

    def available(self) -> bool:
        try:
            import psycopg

            with psycopg.connect(self._dsn(), connect_timeout=2):
                return True
        except Exception:  # noqa: BLE001
            return False

    def setup(self) -> None:
        import psycopg
        from psycopg.conninfo import conninfo_to_dict

        self._admin = psycopg.connect(self._dsn(), autocommit=True)
        self._admin.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
        self._admin.execute(f"GRANT USAGE ON SCHEMA {self._schema} TO {READER[0]}")
        reader = conninfo_to_dict(self._dsn()) | {"user": READER[0], "password": READER[1]}
        self._reader = psycopg.connect(**reader, autocommit=True,
                                       options=f"-c search_path={self._schema}")

    def load(self, table: str, df: pd.DataFrame) -> None:
        name = f"{self._schema}.{table}"
        cols = ", ".join(f"{c} {_pg_type(dt)}" for c, dt in df.dtypes.items())
        with self._admin.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {name}")
            cur.execute(f"CREATE TABLE {name} ({cols})")
            with cur.copy(f"COPY {name} FROM STDIN") as copy:
                for rec in df.itertuples(index=False, name=None):
                    copy.write_row([_py(v) for v in rec])
            cur.execute(f"GRANT SELECT ON {name} TO {READER[0]}")

    def scalar(self, sql: str) -> Any:
        with self._reader.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
        return _norm(row[0]) if row else None

    def teardown(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._admin is not None:
            self._admin.execute(f"DROP SCHEMA IF EXISTS {self._schema} CASCADE")
            self._admin.close()
            self._admin = None


class MySQLEngine(Engine):
    """MySQL in the sandbox (DAYOFWEEK() is 1=Sunday).

    The same split as `PostgresEngine`: tables load through root into a scratch database, dropped
    on teardown, and queries run through `sftgen_reader` with SELECT on that database only. The
    server stops every SELECT after 30 s. `SFTGEN_MYSQL_DSN` (a JSON dict of pymysql.connect
    arguments) overrides the admin login.
    """

    name = "mysql"

    def __init__(self) -> None:
        self._admin = None
        self._reader = None
        self._db = f"sftgen_{uuid.uuid4().hex[:10]}"

    @staticmethod
    def _cfg() -> dict:
        env = os.environ.get("SFTGEN_MYSQL_DSN")
        return json.loads(env) if env else dict(MYSQL_ADMIN)

    def available(self) -> bool:
        try:
            import pymysql

            pymysql.connect(connect_timeout=2, **self._cfg()).close()
            return True
        except Exception:  # noqa: BLE001
            return False

    def setup(self) -> None:
        import pymysql

        cfg = self._cfg()
        self._admin = pymysql.connect(autocommit=True, **cfg)
        with self._admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {self._db}")
            cur.execute(f"GRANT SELECT ON {self._db}.* TO '{READER[0]}'@'%'")
        self._reader = pymysql.connect(
            host=cfg["host"], port=cfg.get("port", 3306), user=READER[0], password=READER[1],
            database=self._db, autocommit=True,
        )

    def load(self, table: str, df: pd.DataFrame) -> None:
        name = f"{self._db}.{table}"
        cols = ", ".join(f"`{c}` {_mysql_type(dt)}" for c, dt in df.dtypes.items())
        with self._admin.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {name}")
            cur.execute(f"CREATE TABLE {name} ({cols})")
            ph = ", ".join(["%s"] * len(df.columns))
            cur.executemany(
                f"INSERT INTO {name} VALUES ({ph})",
                [tuple(_py(v) for v in rec) for rec in df.itertuples(index=False, name=None)],
            )

    def scalar(self, sql: str) -> Any:
        with self._reader.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
        return _norm(row[0]) if row else None

    def teardown(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._admin is not None:
            with self._admin.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS {self._db}")
            self._admin.close()
            self._admin = None


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


# Registries in the ADR-004 order (ClickHouse first: it is the measured-gap dialect), one engine
# per dialect in each.
_TRUSTED: tuple[type[Engine], ...] = (ClickHouseEngine, DuckDBEngine, PostgresEngine, MySQLEngine)
_SANDBOXED: tuple[type[Engine], ...] = (
    ClickHouseEngine, SandboxDuckDBEngine, PostgresEngine, MySQLEngine,
)


def available_engines(only: list[str] | None = None, sandboxed: bool = False) -> list[Engine]:
    """Instantiate the engines that are reachable right now (optionally filtered by name).

    `sandboxed=True` is for SQL a model wrote: every engine returned runs inside the dsbench
    sandbox, and in-process DuckDB gives way to its container.
    """
    out: list[Engine] = []
    for cls in _SANDBOXED if sandboxed else _TRUSTED:
        eng = cls()
        if only and eng.name not in only:
            continue
        if eng.available():
            out.append(eng)
    return out
