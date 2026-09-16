"""DA medium: conversion rate per cohort."""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import values_equal
from dsbench.schema import Problem


def make_inputs() -> dict:
    users = pd.DataFrame({
        "cohort": ["jan", "jan", "jan", "feb", "feb", "mar"],
        "converted": [1, 0, 1, 0, 0, 1],
    })
    return {"users": users}


def _reference(users: pd.DataFrame) -> pd.DataFrame:
    r = users.groupby("cohort", as_index=False)["converted"].mean()
    return r.rename(columns={"converted": "conversion_rate"})


def check(result) -> bool:
    return values_equal(result, _reference(make_inputs()["users"]))


PROMPT = """You are given a pandas DataFrame `users` with columns cohort (str) and
converted (int, 0 or 1). Write `solve(users)` that returns a DataFrame with one row per cohort
and columns cohort, conversion_rate, where conversion_rate is the fraction of that cohort that
converted (the mean of converted). Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
def solve(users):
    r = users.groupby("cohort", as_index=False)["converted"].mean()
    return r.rename(columns={"converted": "conversion_rate"})
'''

PROBLEM = Problem(
    id="da_medium_01", category="da", difficulty="medium",
    title="Conversion rate per cohort", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("pandas", "groupby"),
)
