"""DS hard: compute ROC AUC for a scored binary classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsbench.harness.checks import approx
from dsbench.schema import Problem


def make_inputs() -> dict:
    rng = np.random.default_rng(5)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7, 0.9, 0.6, 0.55, 0.95])
    _ = rng  # fixed data; rng kept for future variants
    return {"df": pd.DataFrame({"y_true": y_true, "score": score})}


def check(result) -> bool:
    from sklearn.metrics import roc_auc_score

    df = make_inputs()["df"]
    exp = float(roc_auc_score(df["y_true"], df["score"]))
    return approx(result, exp, rtol=1e-3)


PROMPT = """You are given a pandas DataFrame `df` with columns y_true (0/1) and score (a float
predicted score, higher = more likely positive). Write `solve(df)` that returns the ROC AUC as a
float. Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
from sklearn.metrics import roc_auc_score
def solve(df):
    return float(roc_auc_score(df["y_true"], df["score"]))
'''

PROBLEM = Problem(
    id="ds_hard_02", category="ds", difficulty="hard",
    title="ROC AUC", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("sklearn", "metrics", "auc"),
)
