"""DA hard: weekly resample + 4-week rolling mean of a daily series."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsbench.harness.checks import values_equal
from dsbench.schema import Problem


def make_inputs() -> dict:
    # 70 daily points -> 10-11 weekly buckets, enough for several full 4-week windows.
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    rng = np.random.default_rng(0)
    value = rng.integers(1, 100, size=70).astype(float)
    df = pd.DataFrame({"date": dates, "value": value})
    return {"df": df}


def _reference(df: pd.DataFrame) -> pd.DataFrame:
    s = df.set_index("date")["value"].resample("W").sum()
    r = s.rolling(4, min_periods=4).mean().dropna()
    return r.reset_index().rename(columns={"value": "rolling_mean", "date": "week"})


def check(result) -> bool:
    return values_equal(result, _reference(make_inputs()["df"]))


PROMPT = """You are given a pandas DataFrame `df` with columns date (datetime, daily) and
value (float). Write `solve(df)`:
  1. resample to weekly buckets with pandas frequency "W" (weeks ending Sunday), summing value,
  2. compute the 4-week rolling mean of those weekly sums (min_periods=4),
  3. drop weeks that do not have a full 4-week window,
and return a DataFrame with columns week (the weekly timestamp) and rolling_mean.
Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
def solve(df):
    s = df.set_index("date")["value"].resample("W").sum()
    r = s.rolling(4, min_periods=4).mean().dropna()
    return r.reset_index().rename(columns={"value": "rolling_mean", "date": "week"})
'''

PROBLEM = Problem(
    id="da_hard_01", category="da", difficulty="hard",
    title="Weekly rolling mean", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("pandas", "timeseries"),
)
