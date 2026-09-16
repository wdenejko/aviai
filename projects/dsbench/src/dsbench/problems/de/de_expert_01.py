"""DE expert: as-of effective-price join.

Trap hypothesis: each sale must be priced with the MOST RECENT price whose effective_from is
<= the sale timestamp (per product), and sales that happen before a product's first price have
no applicable price and must be dropped. The tempting wrong answers:
  - equi-join sale.ts = price.effective_from  -> matches almost nothing;
  - join on effective_from <= ts without "take the latest" -> counts every earlier price, so
    revenue is inflated;
  - LEFT join that keeps pre-price sales -> extra rows / NULL revenue.
The reference uses DuckDB's ASOF JOIN; the checker recomputes the truth independently with
pandas.merge_asof (a different engine and code path), so agreeing is real evidence, not a mirror.
"""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import values_equal
from dsbench.schema import Problem


def make_inputs() -> dict:
    # Two products. Prices step up over time. Some sales land before the first price (must drop),
    # some exactly on an effective_from boundary (inclusive), some between changes (take latest).
    sales = pd.DataFrame({
        "sale_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "product": ["A", "A", "A", "A", "B", "B", "B", "B", "A"],
        "ts": pd.to_datetime([
            "2026-01-01 09:00",  # A: before first A price (2026-01-02) -> DROP
            "2026-01-02 00:00",  # A: exactly on boundary -> price 10.0
            "2026-01-05 12:00",  # A: between -> still 10.0
            "2026-01-10 08:00",  # A: after change -> 12.5
            "2026-01-03 10:00",  # B: exactly on boundary -> 4.0
            "2026-01-03 09:59",  # B: one minute before first B price -> DROP
            "2026-01-08 00:00",  # B: after change -> 5.0
            "2026-01-20 00:00",  # B: after 2nd change -> 6.0
            "2026-01-02 06:00",  # A: after first A price same day -> 10.0
        ]),
        "qty": [3, 2, 1, 5, 10, 7, 4, 2, 1],
    })
    price_history = pd.DataFrame({
        "product": ["A", "A", "B", "B", "B"],
        "effective_from": pd.to_datetime([
            "2026-01-02 00:00", "2026-01-09 00:00",
            "2026-01-03 10:00", "2026-01-07 00:00", "2026-01-15 00:00",
        ]),
        "price": [10.0, 12.5, 4.0, 5.0, 6.0],
    })
    return {"tables": {"sales": sales, "price_history": price_history}}


def _expected() -> pd.DataFrame:
    t = make_inputs()["tables"]
    sales = t["sales"].sort_values("ts")
    ph = t["price_history"].sort_values("effective_from")
    merged = pd.merge_asof(
        sales, ph, left_on="ts", right_on="effective_from", by="product", direction="backward"
    )
    merged = merged.dropna(subset=["price"])  # sales before the product's first price: excluded
    merged["revenue"] = merged["qty"] * merged["price"]
    return merged.groupby("product", as_index=False)["revenue"].sum()[["product", "revenue"]]


def check(result) -> tuple[bool, str]:
    exp = _expected()
    ok = values_equal(result, exp)
    return ok, "" if ok else f"expected rows {exp.values.tolist()}"


PROMPT = """You are given two DuckDB tables:
  sales(sale_id INTEGER, product TEXT, ts TIMESTAMP, qty INTEGER)
  price_history(product TEXT, effective_from TIMESTAMP, price DOUBLE)

For each sale, the applicable unit price is the price for that product whose effective_from is
the LATEST one that is <= the sale's ts (a price takes effect at its effective_from, inclusive).
A sale whose ts is before that product's earliest effective_from has NO applicable price and must
be excluded entirely.

Write ONE DuckDB SQL SELECT returning two columns: the product and its total revenue, where
revenue = SUM(qty * applicable unit price) over that product's priced sales. One row per product
that has at least one priced sale.

Return only a single ```sql code block.
"""

REFERENCE = (
    "SELECT s.product, SUM(s.qty * p.price) AS revenue "
    "FROM sales s ASOF JOIN price_history p "
    "ON s.product = p.product AND s.ts >= p.effective_from "
    "GROUP BY s.product ORDER BY s.product"
)

PROBLEM = Problem(
    id="de_expert_01", category="de", difficulty="expert",
    title="As-of effective-price join", prompt=PROMPT, mode="sql",
    make_inputs=make_inputs, check=check, reference=REFERENCE,
    tags=("sql", "asof", "temporal-join"),
)
