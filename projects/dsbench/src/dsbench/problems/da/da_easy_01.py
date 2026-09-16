"""DA easy: mean of a column grouped by another, returned as a dict."""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import dicts_approx
from dsbench.schema import Problem


def make_inputs() -> dict:
    df = pd.DataFrame({
        "group": ["a", "a", "b", "b", "b", "c"],
        "score": [10.0, 20.0, 5.0, 7.0, 9.0, 100.0],
    })
    return {"df": df}


def check(result) -> bool:
    df = make_inputs()["df"]
    exp = df.groupby("group")["score"].mean().to_dict()
    return dicts_approx(result, exp)


PROMPT = """You are given a pandas DataFrame `df` with columns group (str) and score (float).
Write `solve(df)` that returns a dict mapping each group to the MEAN score of that group.
Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
def solve(df):
    return df.groupby("group")["score"].mean().to_dict()
'''

PROBLEM = Problem(
    id="da_easy_01", category="da", difficulty="easy",
    title="Mean score per group", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("pandas", "groupby"),
)
