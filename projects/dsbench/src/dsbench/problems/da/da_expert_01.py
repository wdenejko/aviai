"""DA expert: ordered-funnel conversion with conditional denominators.

Trap hypothesis: "conversion" is ORDERED and CONDITIONAL, not a set-membership count.
  - A user reaches `cart` only via a cart event strictly AFTER their first view (a cart that
    happens before the user's first view does not count).
  - A user reaches `purchase` only via a purchase strictly AFTER that qualifying cart.
  - The cart->purchase rate is over users who REACHED cart, not over all viewers or all users.
  - Users with no view at all are not in the funnel (n_view excludes them).
The tempting wrong answer -- (# users with any cart)/(# users with any view), etc. -- is a
different number here, because one user carts before viewing and one carts+buys without ever
viewing. The checker recomputes each stage by brute force over per-user sorted events.
"""
from __future__ import annotations

import pandas as pd

from dsbench.harness.checks import dicts_approx
from dsbench.schema import Problem


def make_inputs() -> dict:
    # ts is an integer event time; larger = later. Hand-built to exercise every ordering trap.
    rows = [
        # u1: clean full funnel
        ("u1", 1, "view"), ("u1", 2, "cart"), ("u1", 3, "purchase"),
        # u2: reaches cart (after a repeat view), never buys
        ("u2", 1, "view"), ("u2", 2, "view"), ("u2", 3, "cart"),
        # u3: cart happens BEFORE the first view -> does NOT reach cart
        ("u3", 1, "cart"), ("u3", 2, "view"),
        # u4: views then buys but never carts -> not in cart denominator
        ("u4", 5, "view"), ("u4", 6, "purchase"),
        # u5: view only
        ("u5", 1, "view"),
        # u6: cart+purchase but NO view -> excluded from the funnel entirely
        ("u6", 1, "cart"), ("u6", 2, "purchase"),
        # u7: an early pre-view cart is ignored; the qualifying cart is the first one after view
        ("u7", 10, "view"), ("u7", 5, "cart"), ("u7", 12, "cart"), ("u7", 15, "purchase"),
        # u8: purchase happens before the qualifying cart -> reaches cart, does NOT reach purchase
        ("u8", 1, "view"), ("u8", 2, "cart"), ("u8", 1, "purchase"),
    ]
    df = pd.DataFrame(rows, columns=["user_id", "ts", "step"])
    return {"df": df}


def _expected() -> dict:
    df = make_inputs()["df"]
    viewers = reached_cart = reached_purchase = 0
    for _, g in df.groupby("user_id"):
        g = g.sort_values("ts", kind="stable")
        views = g.loc[g["step"] == "view", "ts"]
        if views.empty:
            continue
        viewers += 1
        first_view = views.min()
        carts_after = g.loc[(g["step"] == "cart") & (g["ts"] > first_view), "ts"]
        if carts_after.empty:
            continue
        reached_cart += 1
        first_cart = carts_after.min()
        buys_after = g.loc[(g["step"] == "purchase") & (g["ts"] > first_cart), "ts"]
        if not buys_after.empty:
            reached_purchase += 1
    return {
        "n_view": viewers,
        "view_to_cart": reached_cart / viewers,
        "cart_to_purchase": reached_purchase / reached_cart,
    }


def check(result) -> tuple[bool, str]:
    exp = _expected()
    if not isinstance(result, dict):
        return False, "expected a dict"
    ok = dicts_approx(result, exp)
    return ok, "" if ok else f"expected {exp}, got {result}"


PROMPT = """You are given a pandas DataFrame `df` of funnel events with columns:
  user_id (str), ts (int event time; larger = later), step (one of 'view', 'cart', 'purchase').

Compute an ORDERED funnel. Rules:
  - Only users with at least one 'view' are in the funnel.
  - A user "reaches cart" iff they have a 'cart' event strictly after their FIRST 'view'.
  - A user "reaches purchase" iff they have a 'purchase' event strictly after the FIRST cart
    that itself qualified above (i.e. after that first post-view cart).
  - Events out of order or before the first view do not count.

Write `solve(df)` that returns a dict with exactly these keys:
  'n_view'          : int   -- number of users with at least one view
  'view_to_cart'    : float -- (users who reached cart) / n_view
  'cart_to_purchase': float -- (users who reached purchase) / (users who reached cart)

Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
def solve(df):
    n_view = reached_cart = reached_purchase = 0
    for _, g in df.groupby("user_id"):
        g = g.sort_values("ts", kind="stable")
        views = g.loc[g["step"] == "view", "ts"]
        if views.empty:
            continue
        n_view += 1
        first_view = views.min()
        carts = g.loc[(g["step"] == "cart") & (g["ts"] > first_view), "ts"]
        if carts.empty:
            continue
        reached_cart += 1
        first_cart = carts.min()
        buys = g.loc[(g["step"] == "purchase") & (g["ts"] > first_cart), "ts"]
        if not buys.empty:
            reached_purchase += 1
    return {
        "n_view": n_view,
        "view_to_cart": reached_cart / n_view,
        "cart_to_purchase": reached_purchase / reached_cart,
    }
'''

PROBLEM = Problem(
    id="da_expert_01", category="da", difficulty="expert",
    title="Ordered funnel conversion", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE,
    tags=("funnel", "ordering", "conditional-rate"),
)
