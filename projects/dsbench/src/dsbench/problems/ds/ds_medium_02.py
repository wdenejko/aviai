"""DS medium: precision / recall / F1 for the positive class."""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import dicts_approx
from dsbench.schema import Problem


def make_inputs() -> dict:
    df = pd.DataFrame({
        "y_true": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        "y_pred": [1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    })
    return {"df": df}


def check(result) -> bool:
    from sklearn.metrics import f1_score, precision_score, recall_score

    df = make_inputs()["df"]
    exp = {
        "precision": float(precision_score(df["y_true"], df["y_pred"])),
        "recall": float(recall_score(df["y_true"], df["y_pred"])),
        "f1": float(f1_score(df["y_true"], df["y_pred"])),
    }
    return dicts_approx(result, exp, rtol=1e-3)


PROMPT = """You are given a pandas DataFrame `df` with columns y_true and y_pred (each 0/1).
Write `solve(df)` that returns a dict with keys "precision", "recall", "f1" for the positive
class (label 1). Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
from sklearn.metrics import precision_score, recall_score, f1_score
def solve(df):
    yt, yp = df["y_true"], df["y_pred"]
    return {
        "precision": float(precision_score(yt, yp)),
        "recall": float(recall_score(yt, yp)),
        "f1": float(f1_score(yt, yp)),
    }
'''

PROBLEM = Problem(
    id="ds_medium_02", category="ds", difficulty="medium",
    title="Precision / recall / F1", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("sklearn", "metrics"),
)
