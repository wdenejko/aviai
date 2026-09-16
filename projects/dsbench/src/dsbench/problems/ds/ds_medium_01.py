"""DS medium: train/test classification accuracy (procedure test on separable data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsbench.schema import Problem


def make_inputs() -> dict:
    rng = np.random.default_rng(3)
    n = 100
    f1 = np.concatenate([rng.normal(-2, 0.5, n), rng.normal(2, 0.5, n)])
    f2 = np.concatenate([rng.normal(-2, 0.5, n), rng.normal(2, 0.5, n)])
    label = np.array([0] * n + [1] * n)
    df = pd.DataFrame({"f1": f1, "f2": f2, "label": label})
    return {"df": df.sample(frac=1.0, random_state=7).reset_index(drop=True)}


def check(result) -> tuple[bool, str]:
    try:
        acc = float(result)
    except (TypeError, ValueError):
        return False, "did not return a float"
    # Data is linearly separable, so any correct train/fit/score reaches ~1.0. Reject leakage-free
    # runs that still land low (broken pipeline) without over-fitting to a sklearn version.
    return (0.9 <= acc <= 1.0), f"accuracy {acc:.3f} outside [0.90, 1.0]"


PROMPT = """You are given a pandas DataFrame `df` with feature columns f1, f2 and a binary label
column `label`. Write `solve(df)`:
  - split into train/test with test_size=0.25 and random_state=42,
  - train sklearn LogisticRegression(max_iter=1000) on the train split,
  - return the accuracy on the test split as a float.
Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def solve(df):
    X = df[["f1", "f2"]]; y = df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return float(accuracy_score(yte, clf.predict(Xte)))
'''

PROBLEM = Problem(
    id="ds_medium_01", category="ds", difficulty="medium",
    title="Train/test classification accuracy", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("sklearn", "classification"),
)
