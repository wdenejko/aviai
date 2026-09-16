"""DE medium: clean a messy extract (mixed date formats, dirty numerics, sloppy categories)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsbench.harness.checks import values_equal
from dsbench.schema import Problem


def make_inputs() -> dict:
    raw = pd.DataFrame({
        "date": ["2024-01-05", "01/06/2024", "2024-01-07", "01/08/2024", "2024-01-09"],
        "amount": ["$1,000.50", "200", "NA", "$3,250.00", "42.10"],
        "category": [" Books ", "TOYS", "food ", "  Books", "Food"],
    })
    return {"raw": raw}


def _reference(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["amount"] = (
        df["amount"].str.replace(r"[$,]", "", regex=True).replace("NA", np.nan).astype(float)
    )
    df["category"] = df["category"].str.strip().str.lower()
    return df


def check(result) -> bool:
    return values_equal(result, _reference(make_inputs()["raw"]))


PROMPT = """You are given a pandas DataFrame `raw` with columns:
  date (str, mixed formats: some "YYYY-MM-DD", some "MM/DD/YYYY")
  amount (str, e.g. "$1,000.50", "200", or "NA" for missing)
  category (str, inconsistent casing and surrounding whitespace)

Write a function `solve(raw)` returning a cleaned DataFrame with the SAME column names where:
  - date is parsed to a pandas datetime,
  - amount is a float with "$" and "," removed and "NA" becoming NaN,
  - category is lowercased and stripped of surrounding whitespace.

Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
import numpy as np
import pandas as pd
def solve(raw):
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    _amt = df["amount"].str.replace(r"[$,]", "", regex=True)
    df["amount"] = _amt.replace("NA", np.nan).astype(float)
    df["category"] = df["category"].str.strip().str.lower()
    return df
'''

PROBLEM = Problem(
    id="de_medium_02", category="de", difficulty="medium",
    title="Clean a messy extract", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("pandas", "cleaning"),
)
