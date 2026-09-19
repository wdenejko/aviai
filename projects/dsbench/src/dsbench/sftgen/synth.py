"""Seeded synthetic domains for Target A -- deliberately NOT aviation (ADR-004 transfer rule).

Each domain is a small table with (i) a timestamp column, so the weekday-numbering and month/quarter
conventions have something to bite on, and (ii) a US-location column with a known summer UTC offset,
so the timezone-direction trap has real cross-zone data (no single offset can fake the answer -- the
same reason dsbench's `da_utc_peak_hour` uses five hubs across four zones).

Data is generated with a seeded numpy Generator, so a row's `(domain, seed)` reproduces it exactly
(the `provenance.seed` contract). No faker dependency: the columns only need realistic *shape*, not
realistic *values*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# US cities -> June/DST offset to add to reach UTC (i.e. -tz_offset; all US offsets are negative).
# Spread across four zones on purpose. e.g. New York = UTC-4 => add 4.
US_CITY_UTC_ADD: dict[str, int] = {
    "New York": 4, "Atlanta": 4, "Chicago": 5, "Dallas": 5, "Denver": 6, "Los Angeles": 7,
}
_CITIES = list(US_CITY_UTC_ADD)


@dataclass(frozen=True)
class Domain:
    name: str  # table name, e.g. "retail_orders"
    df: pd.DataFrame
    ts_col: str  # the primary timestamp column
    location_col: str  # the US-city column (for the timezone family)
    label: str  # human phrase for the row noun, e.g. "orders"


def _timestamps(rng: np.random.Generator, n: int) -> pd.Series:
    """Random datetimes spread across a year, with a random time-of-day (whole seconds)."""
    days = rng.integers(0, 365, size=n)
    secs = rng.integers(0, 24 * 3600, size=n)
    base = np.datetime64("2025-01-01T00:00:00")
    ts = base + days.astype("timedelta64[D]") + secs.astype("timedelta64[s]")
    return pd.Series(pd.to_datetime(ts))


def _cities(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.choice(_CITIES, size=n)


def retail_orders(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1, dtype="int64"),
        "order_ts": _timestamps(rng, n),
        "store_city": _cities(rng, n),
        "category": rng.choice(["grocery", "apparel", "electronics", "home"], size=n),
        "amount": np.round(rng.gamma(3.0, 20.0, size=n), 2),
    })
    return Domain("retail_orders", df, "order_ts", "store_city", "orders")


def iot_readings(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "reading_id": np.arange(1, n + 1, dtype="int64"),
        "reading_ts": _timestamps(rng, n),
        "site_city": _cities(rng, n),
        "sensor": rng.choice(["temp", "humidity", "pressure", "co2"], size=n),
        "value": np.round(rng.normal(50.0, 12.0, size=n), 3),
    })
    return Domain("iot_readings", df, "reading_ts", "site_city", "readings")


def support_tickets(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ticket_id": np.arange(1, n + 1, dtype="int64"),
        "opened_ts": _timestamps(rng, n),
        "office_city": _cities(rng, n),
        "priority": rng.choice(["low", "medium", "high", "urgent"], size=n),
        "channel": rng.choice(["email", "phone", "chat"], size=n),
    })
    return Domain("support_tickets", df, "opened_ts", "office_city", "tickets")


def payments(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "payment_id": np.arange(1, n + 1, dtype="int64"),
        "paid_ts": _timestamps(rng, n),
        "branch_city": _cities(rng, n),
        "method": rng.choice(["card", "ach", "wire", "cash"], size=n),
        "amount": np.round(rng.gamma(2.5, 40.0, size=n), 2),
    })
    return Domain("payments", df, "paid_ts", "branch_city", "payments")


def web_sessions(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "session_id": np.arange(1, n + 1, dtype="int64"),
        "started_ts": _timestamps(rng, n),
        "edge_city": _cities(rng, n),
        "device": rng.choice(["mobile", "desktop", "tablet"], size=n),
        "duration_s": np.round(rng.exponential(180.0, size=n), 1),
    })
    return Domain("web_sessions", df, "started_ts", "edge_city", "sessions")


def gym_checkins(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "checkin_id": np.arange(1, n + 1, dtype="int64"),
        "checkin_ts": _timestamps(rng, n),
        "club_city": _cities(rng, n),
        "plan": rng.choice(["basic", "plus", "premium"], size=n),
        "minutes": np.round(rng.normal(65.0, 20.0, size=n), 1),
    })
    return Domain("gym_checkins", df, "checkin_ts", "club_city", "check-ins")


_BUILDERS = {b.__name__: b for b in (
    retail_orders, iot_readings, support_tickets, payments, web_sessions, gym_checkins,
)}


def build(domain: str, seed: int, n: int = 4000) -> Domain:
    return _BUILDERS[domain](seed, n)


def domain_names() -> list[str]:
    return list(_BUILDERS)
