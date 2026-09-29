"""Run one query on one table in a fresh in-memory DuckDB, with external access switched off.

    docker exec -i avbench-duckdb timeout 30 python /runner.py < request.json

stdin:  {"table": name, "parquet": base64 Parquet bytes, "sql": the query}
stdout: {"ok": true, "value": first column of the first row} or {"ok": false, "error": message}

The table is loaded first. Then external access is switched off (files, URLs, extensions,
ATTACH) and the configuration locked, so the query can read the table and nothing else, and can't
switch access back on. That is the second wall; the first is the container itself (no network, a
read-only root filesystem, no host mounts). Called by sftgen/engines.SandboxDuckDBEngine.
"""

import base64
import datetime
import decimal
import json
import os
import re
import sys
import tempfile

import duckdb


def _plain(value):
    """A JSON-safe scalar: numbers stay numbers, temporal values become ISO strings."""
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, datetime.date | datetime.time | datetime.timedelta):
        return str(value)
    return value


def run(request: dict) -> dict:
    table = request["table"]
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
        return {"ok": False, "error": f"bad table name {table!r}"}
    con = duckdb.connect(":memory:")
    fd, path = tempfile.mkstemp(suffix=".parquet")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(base64.b64decode(request["parquet"]))
        con.execute("SET threads = 2")
        con.execute("SET memory_limit = '1GB'")
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet(?)", [path])
    finally:
        os.remove(path)
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    try:
        row = con.execute(request["sql"]).fetchone()
    except Exception as exc:  # noqa: BLE001 - any failure of the query is the answer
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    finally:
        con.close()
    return {"ok": True, "value": _plain(row[0]) if row else None}


def main() -> None:
    print(json.dumps(run(json.load(sys.stdin))))


if __name__ == "__main__":
    main()
