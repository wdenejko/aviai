"""DE medium: top-N per group with a window function."""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import values_equal
from dsbench.schema import Problem


def make_inputs() -> dict:
    sales = pd.DataFrame({
        "category": ["a", "a", "a", "b", "b", "b", "c"],
        "product": ["p1", "p2", "p3", "q1", "q2", "q3", "r1"],
        "revenue": [30.0, 20.0, 10.0, 5.0, 25.0, 15.0, 7.0],
    })
    return {"tables": {"sales": sales}}


def _reference(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.sort_values(["category", "revenue", "product"], ascending=[True, False, True])
    return s.groupby("category").head(2)[["category", "product", "revenue"]]


def check(result) -> bool:
    return values_equal(result, _reference(make_inputs()["tables"]["sales"]))


PROMPT = """You are given a DuckDB table:
  sales(category TEXT, product TEXT, revenue DOUBLE)

Write ONE DuckDB SQL query that returns the top 2 products by revenue WITHIN each category,
with columns category, product, revenue. If two products in a category tie on revenue, prefer
the smaller product name (ascending).

Return only a single ```sql code block.
"""

REFERENCE = """
SELECT category, product, revenue FROM (
  SELECT category, product, revenue,
         ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC, product ASC) AS rn
  FROM sales
) t WHERE rn <= 2
"""

PROBLEM = Problem(
    id="de_medium_01", category="de", difficulty="medium",
    title="Top 2 products per category", prompt=PROMPT, mode="sql",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("sql", "window"),
)
