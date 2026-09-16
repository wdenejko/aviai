"""DE hard: sessionize events (new session when the gap to the previous event exceeds 30 min)."""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import values_equal
from dsbench.schema import Problem

_T = pd.Timestamp


def make_inputs() -> dict:
    events = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2, 2],
        "ts": [
            _T("2024-01-01 09:00:00"),
            _T("2024-01-01 09:20:00"),   # +20m -> same session
            _T("2024-01-01 10:05:00"),   # +45m -> new session
            _T("2024-01-01 10:15:00"),   # +10m -> same
            _T("2024-01-01 08:00:00"),
            _T("2024-01-01 09:00:00"),   # +60m -> new session
            _T("2024-01-01 09:10:00"),   # +10m -> same
        ],
    })
    return {"tables": {"events": events}}


def _reference(events: pd.DataFrame) -> pd.DataFrame:
    e = events.sort_values(["user_id", "ts"]).copy()
    prev = e.groupby("user_id")["ts"].shift()
    is_new = prev.isna() | ((e["ts"] - prev) > pd.Timedelta(minutes=30))
    e["session_id"] = is_new.groupby(e["user_id"]).cumsum().astype(int)
    return e[["user_id", "ts", "session_id"]]


def check(result) -> bool:
    return values_equal(result, _reference(make_inputs()["tables"]["events"]))


PROMPT = """You are given a DuckDB table:
  events(user_id INTEGER, ts TIMESTAMP)

Assign a session_id to every event. Within each user, ordered by ts, a new session starts at the
user's first event and whenever the gap from the previous event is MORE THAN 30 minutes.
session_id starts at 1 for each user and increases by 1 per new session.

Return ONE DuckDB SQL query with columns user_id, ts, session_id. Return only a ```sql code block.
"""

REFERENCE = """
WITH d AS (
  SELECT user_id, ts,
         CASE WHEN LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) IS NULL
                   OR ts - LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) > INTERVAL 30 MINUTE
              THEN 1 ELSE 0 END AS is_new
  FROM events
)
SELECT user_id, ts, SUM(is_new) OVER (PARTITION BY user_id ORDER BY ts) AS session_id FROM d
"""

PROBLEM = Problem(
    id="de_hard_01", category="de", difficulty="hard",
    title="Sessionize by 30-min gap", prompt=PROMPT, mode="sql",
    make_inputs=make_inputs, check=check, reference=REFERENCE,
    tags=("sql", "window", "sessionization"),
)
