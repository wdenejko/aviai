"""DE easy: aggregate revenue per category (GROUP BY + ORDER BY)."""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import values_equal
from dsbench.schema import Problem


def make_inputs() -> dict:
    orders = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 6, 7],
        "category": ["books", "books", "toys", "toys", "food", "food", "food"],
        "amount": [10.0, 5.0, 20.0, 7.5, 3.0, 3.0, 4.0],
    })
    return {"tables": {"orders": orders}}


def check(result) -> bool:
    orders = make_inputs()["tables"]["orders"]
    exp = orders.groupby("category", as_index=False)["amount"].sum()
    return values_equal(result, exp[["category", "amount"]])


PROMPT = """You are given a DuckDB table:
  orders(order_id INTEGER, category TEXT, amount DOUBLE)

Write ONE DuckDB SQL SELECT that returns two columns: the category and its total amount,
one row per category, ordered by total amount descending.

Return only a single ```sql code block.
"""

REFERENCE = (
    "SELECT category, SUM(amount) AS revenue FROM orders "
    "GROUP BY category ORDER BY revenue DESC"
)

PROBLEM = Problem(
    id="de_easy_01", category="de", difficulty="easy",
    title="Revenue per category", prompt=PROMPT, mode="sql",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("sql", "groupby"),
)
