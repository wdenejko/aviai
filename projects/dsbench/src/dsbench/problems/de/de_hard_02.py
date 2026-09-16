"""DE hard: upsert / SCD merge -- newest record per id wins, new ids appended."""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import values_equal
from dsbench.schema import Problem


def make_inputs() -> dict:
    current = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "value": [10, 20, 30],
        "updated_at": [100, 100, 300],
    })
    updates = pd.DataFrame({
        "id": [2, 3, 4],
        "name": ["b2", "c2", "d"],
        "value": [21, 31, 40],
        "updated_at": [200, 150, 200],  # id 2 update is newer; id 3 update is OLDER than current
    })
    return {"current": current, "updates": updates}


def _reference(current: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([current, updates], ignore_index=True).sort_values("updated_at")
    return both.drop_duplicates("id", keep="last").sort_values("id").reset_index(drop=True)


def check(result) -> bool:
    ins = make_inputs()
    return values_equal(result, _reference(ins["current"], ins["updates"]))


PROMPT = """You are given two pandas DataFrames with the same columns:
  current(id, name, value, updated_at)   -- the existing table
  updates(id, name, value, updated_at)   -- incoming changes

Write `solve(current, updates)` that merges them so that, for each id, the row with the LARGEST
updated_at wins (an update only replaces the current row if it is newer). ids present only in one
frame are kept. Return the merged DataFrame sorted by id, columns id, name, value, updated_at.

Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
import pandas as pd
def solve(current, updates):
    both = pd.concat([current, updates], ignore_index=True).sort_values("updated_at")
    return both.drop_duplicates("id", keep="last").sort_values("id").reset_index(drop=True)
'''

PROBLEM = Problem(
    id="de_hard_02", category="de", difficulty="hard",
    title="Upsert / SCD merge", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("pandas", "upsert"),
)
