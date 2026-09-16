"""Comparison helpers for checkers. Import these in problem `check()` functions.

The recurring hazard in DS/DE grading is being too strict (float noise, column aliasing, row
order) or too loose (accepting a df with the right shape but wrong values). These helpers pick
sensible middles: floats compared with tolerance, DataFrames compared as value multisets so a
model's column names and row order do not matter, NaN treated as equal to NaN.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def approx(a: Any, b: Any, rtol: float = 1e-3, atol: float = 1e-6) -> bool:
    """Scalar closeness with NaN==NaN. Use for correlations, coefficients, metrics."""
    try:
        if a is None or b is None:
            return a is None and b is None
        af, bf = float(a), float(b)
        if math.isnan(af) and math.isnan(bf):
            return True
        return bool(np.isclose(af, bf, rtol=rtol, atol=atol))
    except (TypeError, ValueError):
        return False


def _norm(x):
    # Collapse every flavour of missing (NaN, None, pd.NA, NaT) to one sentinel so that
    # tuple equality treats missing == missing (Python's NaN != NaN would otherwise break it).
    try:
        if bool(pd.isna(x)):
            return "__NULL__"
    except (TypeError, ValueError):
        pass
    return x


def _rows(df: pd.DataFrame, round_to: int) -> list[tuple]:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(round_to)
    rows = [tuple(_norm(v) for v in r) for r in d.itertuples(index=False, name=None)]
    return sorted(rows, key=lambda t: [str(x) for x in t])


def values_equal(got: Any, expected: pd.DataFrame, round_to: int = 6) -> bool:
    """True if `got` holds the same rows as `expected`, ignoring column names and row order.

    Robust to SQL aliasing and to a model returning columns in a different order. Requires the
    same number of columns and the same value multiset. Use for SQL results and for pandas
    results where only the values matter.
    """
    if not isinstance(got, pd.DataFrame):
        return False
    if got.shape[1] != expected.shape[1]:
        return False
    try:
        return _rows(got, round_to) == _rows(expected, round_to)
    except Exception:
        return False


def frame_equal(got: Any, expected: pd.DataFrame, round_to: int = 6, sort_cols=None) -> bool:
    """Stricter: same columns (case-insensitive), compared after sorting rows and rounding floats.

    Use when the prompt fixes the output column names and they are part of correctness.
    """
    if not isinstance(got, pd.DataFrame):
        return False
    try:
        a = got.copy()
        b = expected.copy()
        a.columns = [str(c).lower() for c in a.columns]
        b.columns = [str(c).lower() for c in b.columns]
        if sorted(a.columns) != sorted(b.columns):
            return False
        a = a[sorted(a.columns)]
        b = b[sorted(b.columns)]
        sc = [c.lower() for c in sort_cols] if sort_cols else list(a.columns)
        a = a.sort_values(sc).reset_index(drop=True)
        b = b.sort_values(sc).reset_index(drop=True)
        for c in a.columns:
            if pd.api.types.is_float_dtype(a[c]) or pd.api.types.is_float_dtype(b[c]):
                if not np.allclose(
                    pd.to_numeric(a[c], errors="coerce"),
                    pd.to_numeric(b[c], errors="coerce"),
                    rtol=10 ** (-round_to),
                    atol=10 ** (-round_to),
                    equal_nan=True,
                ):
                    return False
            else:
                if not a[c].reset_index(drop=True).equals(b[c].reset_index(drop=True)):
                    return False
        return True
    except Exception:
        return False


def dicts_approx(got: Any, expected: dict, rtol: float = 1e-3) -> bool:
    """Same keys, numeric values close. For 'return a dict of group -> metric' problems."""
    if not isinstance(got, dict):
        return False
    if set(got.keys()) != set(expected.keys()):
        return False
    return all(approx(got[k], expected[k], rtol=rtol) for k in expected)
