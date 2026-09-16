"""DS hard: leakage-free CV with an imputing + scaling + model Pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsbench.schema import Problem


def make_inputs() -> dict:
    rng = np.random.default_rng(4)
    n = 120
    f1 = np.concatenate([rng.normal(-1.5, 1.0, n), rng.normal(1.5, 1.0, n)])
    f2 = np.concatenate([rng.normal(-1.5, 1.0, n), rng.normal(1.5, 1.0, n)])
    label = np.array([0] * n + [1] * n)
    df = pd.DataFrame({"f1": f1, "f2": f2, "label": label})
    df = df.sample(frac=1.0, random_state=9).reset_index(drop=True)
    # Punch a few holes so imputation is actually needed.
    holes = rng.choice(len(df), size=12, replace=False)
    df.loc[holes, "f1"] = np.nan
    return {"df": df}


def check(result) -> tuple[bool, str]:
    try:
        score = float(result)
    except (TypeError, ValueError):
        return False, "did not return a float"
    # Moderately separable -> a correct leakage-free pipeline scores high; property check keeps
    # this stable across sklearn versions.
    return (0.8 <= score <= 1.0), f"mean CV accuracy {score:.3f} outside [0.80, 1.0]"


PROMPT = """You are given a pandas DataFrame `df` with feature columns f1, f2 (f1 has some missing
values) and a binary `label`. Write `solve(df)` that:
  - builds an sklearn Pipeline: SimpleImputer(strategy="mean") -> StandardScaler ->
    LogisticRegression(max_iter=1000),
  - runs 5-fold cross-validation (cv=5, accuracy) on the whole dataset,
  - returns the MEAN cross-validation accuracy as a float.
Use a Pipeline so imputation and scaling are fit inside each fold (no leakage).
Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
def solve(df):
    X = df[["f1", "f2"]]; y = df["label"]
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    return float(cross_val_score(pipe, X, y, cv=5).mean())
'''

PROBLEM = Problem(
    id="ds_hard_01", category="ds", difficulty="hard",
    title="Leakage-free CV pipeline", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("sklearn", "pipeline", "cv"),
)
