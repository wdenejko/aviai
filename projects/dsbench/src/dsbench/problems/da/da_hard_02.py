"""DA hard: month-1 cohort retention."""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import values_equal
from dsbench.schema import Problem


def make_inputs() -> dict:
    # signup_month/active_month are integer month indices.
    users = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 6],
        "signup_month": [0, 0, 0, 1, 1, 2],
    })
    events = pd.DataFrame({
        "user_id": [1, 2, 4, 6, 1, 6],
        "active_month": [1, 1, 2, 3, 2, 4],
    })
    return {"users": users, "events": events}


def _reference(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    active = set(zip(events["user_id"], events["active_month"], strict=False))
    rows = []
    for m, grp in users.groupby("signup_month"):
        uids = grp["user_id"].tolist()
        n = len(uids)
        act = sum(1 for u in uids if (u, m + 1) in active)
        rows.append((m, act / n))
    return pd.DataFrame(rows, columns=["signup_month", "retention_m1"])


def check(result) -> bool:
    ins = make_inputs()
    return values_equal(result, _reference(ins["users"], ins["events"]))


PROMPT = """You are given two pandas DataFrames:
  users(user_id int, signup_month int)     -- each user's signup month index
  events(user_id int, active_month int)     -- (user, month) rows when a user was active

Write `solve(users, events)` that computes month-1 retention per signup cohort: for each
signup_month m, the fraction of that cohort's users who were active in month m+1. Return a
DataFrame with columns signup_month and retention_m1 (one row per signup_month).
Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
import pandas as pd
def solve(users, events):
    active = set(zip(events["user_id"], events["active_month"], strict=False))
    rows = []
    for m, grp in users.groupby("signup_month"):
        uids = grp["user_id"].tolist()
        act = sum(1 for u in uids if (u, m + 1) in active)
        rows.append((m, act / len(uids)))
    return pd.DataFrame(rows, columns=["signup_month", "retention_m1"])
'''

PROBLEM = Problem(
    id="da_hard_02", category="da", difficulty="hard",
    title="Cohort month-1 retention", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("pandas", "cohort"),
)
