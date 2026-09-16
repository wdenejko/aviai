"""DA easy: count rows matching a compound condition."""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import approx
from dsbench.schema import Problem


def make_inputs() -> dict:
    df = pd.DataFrame({
        "amount": [50, 150, 200, 90, 300, 120],
        "status": ["active", "active", "inactive", "active", "active", "inactive"],
    })
    return {"df": df}


def check(result) -> bool:
    df = make_inputs()["df"]
    exp = int(((df["amount"] > 100) & (df["status"] == "active")).sum())
    return approx(result, exp, rtol=0, atol=0)


PROMPT = """You are given a pandas DataFrame `df` with columns amount (int) and status (str).
Write `solve(df)` that returns the integer number of rows where amount > 100 AND status == "active".
Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
def solve(df):
    return int(((df["amount"] > 100) & (df["status"] == "active")).sum())
'''

PROBLEM = Problem(
    id="da_easy_02", category="da", difficulty="easy",
    title="Count with a filter", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("pandas", "filter"),
)
