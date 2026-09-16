"""ClickHouse client for the agentic harness (host side). Local sandbox creds (ADR-003).

Env-overridable so the same code works from the host (localhost:8123) and, if ever needed, from
inside a container (CLICKHOUSE_HOST=clickhouse).
"""
from __future__ import annotations

import os


def get_client(database: str = "aviation"):
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=os.environ.get("CLICKHOUSE_HOST", "localhost"),
        port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        username=os.environ.get("CLICKHOUSE_USER", "avbench"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", "avbench"),
        database=database,
    )
