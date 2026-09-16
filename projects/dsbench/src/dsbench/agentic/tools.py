"""The agent's tools: schemas advertised to the model + the executors that run them.

run_sql runs on ClickHouse (host side). run_python execs INSIDE the `workspace` container, never on
the host (ADR-003 safety). Tool errors are returned as text, not raised, so the agent can see a
failure and recover — that is the whole point of an agent loop.
"""
from __future__ import annotations

import subprocess

WORKSPACE_CONTAINER = "avbench-workspace"
MAX_ROWS = 200  # cap rows returned to the model
MAX_CHARS = 6000  # cap total observation size

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Run ONE ClickHouse SQL statement and return the result. Shared aviation "
                "data is in the `aviation` database (query fully-qualified, e.g. "
                "aviation.flights). Your scratch database is the default, so `CREATE TABLE "
                "t ...` / `INSERT` land there. SELECT/SHOW/DESCRIBE return rows (capped); "
                "DDL/DML return 'OK'. ClickHouse has no correlated subqueries — use JOINs."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "one SQL statement"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python 3 in the analysis container. Available: pandas, numpy, "
                "scipy, scikit-learn, clickhouse_connect. ClickHouse is at host 'clickhouse'; "
                "the CLICKHOUSE_* env (incl. CLICKHOUSE_DB = YOUR scratch database) is set, so a "
                "client built from it reads/writes your scratch DB by default — query shared data "
                "as aviation.* (fully-qualified). Returns stdout+stderr; print what you want."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to run"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call when the task is complete. If the task asks a question, pass the "
                "final value in `answer`. If it asks you to build a table, build it first "
                "(with run_sql), then call finish (answer optional)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"answer": {"description": "final answer: string, number, or JSON"}},
            },
        },
    },
]


def _clip(s: str) -> str:
    return s if len(s) <= MAX_CHARS else s[:MAX_CHARS] + f"\n... [truncated, {len(s)} chars]"


def run_sql(client, query: str) -> str:
    """Run one statement. SELECT-like -> capped text table; else -> 'OK'. Errors -> text."""
    q = (query or "").strip().rstrip(";")
    if not q:
        return "SQL error: empty query"
    head = q.lstrip("( ").split(None, 1)[0].upper() if q else ""
    try:
        if head in {"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"}:
            res = client.query(q)
            cols, rows = res.column_names, res.result_rows
            lines = [" | ".join(cols)]
            for r in rows[:MAX_ROWS]:
                lines.append(" | ".join("" if v is None else str(v) for v in r))
            if len(rows) > MAX_ROWS:
                lines.append(f"... [{len(rows)} rows total, showing {MAX_ROWS}]")
            return _clip("\n".join(lines) if rows else "(0 rows)")
        client.command(q)
        return "OK"
    except Exception as e:  # noqa: BLE001 — surface the error to the agent, don't crash the run
        return f"SQL error: {str(e)[:800]}"


def run_python(code: str, namespace: str | None = None) -> str:
    """Exec Python in the workspace container. Returns stdout+stderr (capped). Errors -> text.

    `namespace` overrides CLICKHOUSE_DB for this exec so the agent's Python defaults to the same
    per-problem scratch DB that run_sql uses (the container's own default is `aviation`).
    """
    if not (code or "").strip():
        return "run_python error: empty code"
    cmd = ["docker", "exec", "-i"]
    if namespace:
        cmd += ["-e", f"CLICKHOUSE_DB={namespace}"]
    cmd += [WORKSPACE_CONTAINER, "python", "-"]
    try:
        p = subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=90)
        out = (p.stdout or "") + (p.stderr or "")
        return _clip(out.strip() or "(no output)")
    except subprocess.TimeoutExpired:
        return "run_python error: timed out (90s)"
    except Exception as e:  # noqa: BLE001
        return f"run_python error: {str(e)[:400]}"
