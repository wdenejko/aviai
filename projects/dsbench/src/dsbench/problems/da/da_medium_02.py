"""DA medium: join two frames, aggregate a metric per key from the joined result."""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import values_equal
from dsbench.schema import Problem


def make_inputs() -> dict:
    orders = pd.DataFrame({
        "customer_id": [1, 1, 2, 3, 3, 3],
        "amount": [10.0, 5.0, 20.0, 1.0, 2.0, 3.0],
    })
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],           # customer 4 has no orders
        "country": ["PL", "DE", "PL", "FR"],
    })
    return {"orders": orders, "customers": customers}


def _reference(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    j = orders.merge(customers, on="customer_id")
    r = j.groupby("country", as_index=False)["amount"].sum()
    return r.rename(columns={"amount": "total"})


def check(result) -> bool:
    ins = make_inputs()
    return values_equal(result, _reference(ins["orders"], ins["customers"]))


PROMPT = """You are given two pandas DataFrames:
  orders(customer_id int, amount float)
  customers(customer_id int, country str)

Write `solve(orders, customers)` that returns the total order amount per country, with columns
country, total. Only include countries that have at least one order. Return only a single
```python code block defining `solve`.
"""

REFERENCE = '''
def solve(orders, customers):
    j = orders.merge(customers, on="customer_id")
    r = j.groupby("country", as_index=False)["amount"].sum()
    return r.rename(columns={"amount": "total"})
'''

PROBLEM = Problem(
    id="da_medium_02", category="da", difficulty="medium",
    title="Revenue per country (join)", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE, tags=("pandas", "join"),
)
