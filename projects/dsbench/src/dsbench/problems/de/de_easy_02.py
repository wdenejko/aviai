"""DE easy: keep only the latest record per key (dedup by timestamp)."""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import values_equal
from dsbench.schema import Problem


def make_inputs() -> dict:
    events = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 2, 3],
        "ts": [100, 130, 90, 200, 150, 50],
        "value": ["a", "b", "c", "d", "e", "f"],
    })
    return {"events": events}


def _reference(events: pd.DataFrame) -> pd.DataFrame:
    return events.sort_values("ts").drop_duplicates("user_id", keep="last").reset_index(drop=True)


def check(result) -> bool:
    return values_equal(result, _reference(make_inputs()["events"]))


PROMPT = """You are given a pandas DataFrame `events` with columns:
  user_id (int), ts (int, a unix timestamp), value (str)

Some user_ids appear more than once. Write a function `solve(events)` that returns a DataFrame
with exactly one row per user_id: the row that has the largest ts for that user. Keep the columns
user_id, ts, value.

Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
def solve(events):
    return events.sort_values("ts").drop_duplicates("user_id", keep="last").reset_index(drop=True)
'''

PROBLEM = Problem(
    id="de_easy_02", category="de", difficulty="easy",
    title="Latest record per user", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("pandas", "dedup"),
)
