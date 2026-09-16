"""DS easy: Pearson correlation between two columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsbench.harness.checks import approx
from dsbench.schema import Problem


def make_inputs() -> dict:
    rng = np.random.default_rng(1)
    x = np.arange(50, dtype=float)
    y = 2.0 * x + rng.normal(0, 5, size=50)
    return {"df": pd.DataFrame({"x": x, "y": y})}


def check(result) -> bool:
    df = make_inputs()["df"]
    exp = float(np.corrcoef(df["x"], df["y"])[0, 1])
    return approx(result, exp, rtol=1e-3)


PROMPT = """You are given a pandas DataFrame `df` with numeric columns x and y.
Write `solve(df)` that returns the Pearson correlation coefficient between x and y as a float.
Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
import numpy as np
def solve(df):
    return float(np.corrcoef(df["x"], df["y"])[0, 1])
'''

PROBLEM = Problem(
    id="ds_easy_01", category="ds", difficulty="easy",
    title="Pearson correlation", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("numpy", "stats"),
)
