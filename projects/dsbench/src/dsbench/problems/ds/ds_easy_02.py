"""DS easy: OLS linear regression slope via scikit-learn."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsbench.harness.checks import approx
from dsbench.schema import Problem


def make_inputs() -> dict:
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 60)
    y = 3.0 * x + 1.0 + rng.normal(0, 0.5, size=60)
    return {"df": pd.DataFrame({"x": x, "y": y})}


def check(result) -> bool:
    from sklearn.linear_model import LinearRegression

    df = make_inputs()["df"]
    m = LinearRegression().fit(df[["x"]], df["y"])
    return approx(result, float(m.coef_[0]), rtol=2e-2)


PROMPT = """You are given a pandas DataFrame `df` with numeric columns x and y.
Write `solve(df)` that fits an ordinary least-squares linear regression of y on x and returns the
slope (the coefficient of x) as a float. Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
from sklearn.linear_model import LinearRegression
def solve(df):
    m = LinearRegression().fit(df[["x"]], df["y"])
    return float(m.coef_[0])
'''

PROBLEM = Problem(
    id="ds_easy_02", category="ds", difficulty="easy",
    title="Linear regression slope", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("sklearn", "regression"),
)
