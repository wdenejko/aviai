"""Seeded synthetic domains for Target A -- deliberately NOT aviation (ADR-004 transfer rule).

Each domain is a small table with (i) a timestamp column, so the weekday-numbering and month/quarter
conventions have something to bite on, and (ii) a US-location column whose UTC offsets the
timezone-direction family states in its question, so that trap has real cross-zone data (no single
offset can fake the answer -- the same reason dsbench's `da_utc_peak_hour` uses five hubs across
four zones).

Data is generated with a seeded numpy Generator, so a row's `(domain, seed)` reproduces it exactly
(the `provenance.seed` contract). No faker dependency: the columns only need realistic *shape*, not
realistic *values*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# US cities -> UTC offset, signed as written ("UTC-4" = -4), in two fixed sets: daylight saving
# time and standard time. The timezone family states one set in its question and holds it for every
# row (conventions.TimezoneDirection). Local -> UTC is UTC = local - offset, so New York in daylight
# time (UTC-4) gives UTC = local + 4. Spread across four zones on purpose.
US_CITY_UTC_OFFSETS: dict[str, dict[str, int]] = {
    "daylight": {"New York": -4, "Atlanta": -4, "Chicago": -5, "Dallas": -5, "Denver": -6,
                 "Los Angeles": -7},
    "standard": {"New York": -5, "Atlanta": -5, "Chicago": -6, "Dallas": -6, "Denver": -7,
                 "Los Angeles": -8},
}
# The order the data draws cities from. Keep it: rows are re-verified by rebuilding their table
# from (domain, seed, n), and a new order would draw different cities.
_CITIES = ["New York", "Atlanta", "Chicago", "Dallas", "Denver", "Los Angeles"]


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


# --- held out: the Target A test's domains (sftgen/target_a_test.py) ------------------------------
# The model sees a table's schema and the question, never its rows. So a new table in a training
# domain asks a training prompt again: Revision 2's training drew 124 of ClickHouse's 126 weekday
# prompts and all 18 weekend ones. These six domains' table names, timestamp columns and row nouns
# appear in no training row, so every prompt built on them is new. Their columns have the shape
# five of the training domains have (an id, a 2025 timestamp, a US city, a category, a number), so
# only the names change. They are not in domain_names(): a generator run over the training domains
# must draw what it drew before (a table's seed follows its domain's position,
# dialect_conventions.table_seed).


def library_loans(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "loan_id": np.arange(1, n + 1, dtype="int64"),
        "loaned_ts": _timestamps(rng, n),
        "library_city": _cities(rng, n),
        "genre": rng.choice(["fiction", "history", "science", "children"], size=n),
        "days_out": np.round(rng.gamma(2.0, 7.0, size=n), 1),
    })
    return Domain("library_loans", df, "loaned_ts", "library_city", "loans")


def ride_trips(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "trip_id": np.arange(1, n + 1, dtype="int64"),
        "pickup_ts": _timestamps(rng, n),
        "depot_city": _cities(rng, n),
        "vehicle": rng.choice(["sedan", "suv", "van"], size=n),
        "fare": np.round(rng.gamma(2.2, 9.0, size=n), 2),
    })
    return Domain("ride_trips", df, "pickup_ts", "depot_city", "trips")


def hotel_stays(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "stay_id": np.arange(1, n + 1, dtype="int64"),
        "arrival_ts": _timestamps(rng, n),
        "hotel_city": _cities(rng, n),
        "room_type": rng.choice(["single", "double", "suite"], size=n),
        "nightly_rate": np.round(rng.gamma(4.0, 35.0, size=n), 2),
    })
    return Domain("hotel_stays", df, "arrival_ts", "hotel_city", "stays")


def parcel_scans(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "scan_id": np.arange(1, n + 1, dtype="int64"),
        "scanned_ts": _timestamps(rng, n),
        "hub_city": _cities(rng, n),
        "service": rng.choice(["ground", "express", "overnight"], size=n),
        "weight_kg": np.round(rng.gamma(1.5, 2.0, size=n), 2),
    })
    return Domain("parcel_scans", df, "scanned_ts", "hub_city", "scans")


def clinic_visits(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "visit_id": np.arange(1, n + 1, dtype="int64"),
        "visit_ts": _timestamps(rng, n),
        "clinic_city": _cities(rng, n),
        "department": rng.choice(["cardiology", "dermatology", "pediatrics", "radiology"], size=n),
        "wait_min": np.round(rng.exponential(25.0, size=n), 1),
    })
    return Domain("clinic_visits", df, "visit_ts", "clinic_city", "visits")


def bike_rentals(seed: int, n: int = 4000) -> Domain:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "rental_id": np.arange(1, n + 1, dtype="int64"),
        "rented_ts": _timestamps(rng, n),
        "dock_city": _cities(rng, n),
        "bike_type": rng.choice(["classic", "electric"], size=n),
        "ride_min": np.round(rng.exponential(22.0, size=n), 1),
    })
    return Domain("bike_rentals", df, "rented_ts", "dock_city", "rentals")


_HELD_OUT = {b.__name__: b for b in (
    library_loans, ride_trips, hotel_stays, parcel_scans, clinic_visits, bike_rentals,
)}


def build(domain: str, seed: int, n: int = 4000) -> Domain:
    return (_BUILDERS | _HELD_OUT)[domain](seed, n)


def domain_names() -> list[str]:
    """The training domains, in the order the generator draws their tables."""
    return list(_BUILDERS)


def held_out_domain_names() -> list[str]:
    """The Target A test's domains: no training row names them."""
    return list(_HELD_OUT)
