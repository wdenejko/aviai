"""The Target A convention traps -- the transferable skills Gate 0 measured as gaps (ADR-004).

Two families, both execution-verifiable:

  * `weekday-numbering` -- the DIALECT trap. Every engine numbers weekdays differently
    (ClickHouse ISO 1=Mon; Postgres/DuckDB 0=Sun; MySQL 1=Sun), so the SAME question needs a
    DIFFERENT integer per dialect. The pandas truth is dialect-independent; the per-dialect SQL must
    reproduce it, which is exactly the awareness the model lacks.

  * `timezone-direction` -- the SIGN/direction trap. US local time is behind UTC, so UTC = local +
    |offset| (New York UTC-4 => add 4); the model adds the signed offset the wrong way. The
    arithmetic is identical across dialects, so this trains the REASONING, not a function name.

A convention exposes: params(rng) [the random draw that fixes the instance], then truth(domain,
params) [pandas, independent], sql(domain, dialect, params) [the label], question / thinking /
system [the prose]. The generator executes sql against a real engine and keeps the row only if it
equals truth.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsbench.sftgen.synth import US_CITY_UTC_ADD, Domain

DIALECT_DISPLAY = {
    "clickhouse": "ClickHouse", "duckdb": "DuckDB", "postgres": "PostgreSQL", "mysql": "MySQL",
}
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _friendly_type(dtype: Any) -> str:
    s = str(dtype)
    if s.startswith("datetime"):
        return "timestamp"
    if s.startswith("int") or s.startswith("uint"):
        return "integer"
    if s.startswith("float"):
        return "number"
    return "text"


def _dow_int(dialect: str, py_weekday: int) -> int:
    """Map a Python weekday (Mon=0..Sun=6) to the dialect's own weekday integer."""
    sun0 = (py_weekday + 1) % 7  # Sunday=0..Saturday=6 (Postgres DOW / DuckDB dayofweek)
    return {
        "clickhouse": py_weekday + 1,  # ISO: Monday=1..Sunday=7
        "duckdb": sun0,
        "postgres": sun0,
        "mysql": sun0 + 1,  # Sunday=1..Saturday=7
    }[dialect]


def _dow_expr(dialect: str, col: str) -> str:
    return {
        "clickhouse": f"toDayOfWeek({col})",
        "duckdb": f"dayofweek({col})",
        "postgres": f"EXTRACT(DOW FROM {col})",
        "mysql": f"DAYOFWEEK({col})",
    }[dialect]


def _hour_expr(dialect: str, col: str) -> str:
    return {
        "clickhouse": f"toHour({col})",
        "duckdb": f"hour({col})",
        "postgres": f"EXTRACT(HOUR FROM {col})",
        "mysql": f"HOUR({col})",
    }[dialect]


def _case_utc_add(loc_col: str) -> str:
    """A standard-SQL CASE mapping each city to the hours to ADD to reach UTC (works everywhere)."""
    whens = " ".join(f"WHEN '{city}' THEN {add}" for city, add in US_CITY_UTC_ADD.items())
    return f"CASE {loc_col} {whens} ELSE 0 END"


class Convention:
    family = "abstract"
    tags: tuple[str, ...] = ()

    def params(self, rng: np.random.Generator) -> dict:  # noqa: D102
        return {}

    def truth(self, domain: Domain, params: dict) -> Any:  # noqa: D102
        raise NotImplementedError

    def sql(self, domain: Domain, dialect: str, params: dict) -> str:  # noqa: D102
        raise NotImplementedError

    def question(self, domain: Domain, params: dict) -> str:  # noqa: D102
        raise NotImplementedError

    def thinking(self, dialect: str, params: dict) -> str:  # noqa: D102
        raise NotImplementedError

    def system(self, domain: Domain, dialect: str) -> str:
        """Engine + schema context (dialect-only). Names the engine so the row is dialect-aware."""
        cols = ", ".join(f"{c} ({_friendly_type(dt)})" for c, dt in domain.df.dtypes.items())
        return (
            f"You are writing SQL for a {DIALECT_DISPLAY[dialect]} database. "
            f"There is one table `{domain.name}` with columns: {cols}. "
            f"Answer with a single SQL query in a ```sql code block, then the numeric result."
        )


class WeekdayNumbering(Convention):
    family = "weekday-numbering"
    tags = ("date", "weekday", "dialect")

    def params(self, rng: np.random.Generator) -> dict:
        return {"weekday": int(rng.integers(0, 7)), "variant": int(rng.integers(0, 3))}

    def truth(self, domain: Domain, params: dict) -> int:
        s = domain.df[domain.ts_col].dt.dayofweek  # Monday=0..Sunday=6
        return int((s == params["weekday"]).sum())

    def question(self, domain: Domain, params: dict) -> str:
        day, col, lab = _WEEKDAYS[params["weekday"]], domain.ts_col, domain.label
        return [
            f"How many {lab} have a {col} that falls on a {day}? Reply with the count.",
            f"Count the {lab} whose {col} is a {day}.",
            f"On {day}s specifically, how many {lab} were there (by {col})? Give the number.",
        ][params["variant"]]

    def sql(self, domain: Domain, dialect: str, params: dict) -> str:
        expr = _dow_expr(dialect, domain.ts_col)
        n = _dow_int(dialect, params["weekday"])
        return f"SELECT count(*) FROM {domain.name} WHERE {expr} = {n}"

    def thinking(self, dialect: str, params: dict) -> str:
        day = _WEEKDAYS[params["weekday"]]
        n = _dow_int(dialect, params["weekday"])
        scheme = {
            "clickhouse": "toDayOfWeek() is ISO: Monday=1 ... Sunday=7",
            "duckdb": "dayofweek() counts Sunday=0 ... Saturday=6",
            "postgres": "EXTRACT(DOW) counts Sunday=0 ... Saturday=6",
            "mysql": "DAYOFWEEK() counts Sunday=1 ... Saturday=7",
        }[dialect]
        return (
            f"{DIALECT_DISPLAY[dialect]} {scheme}, so {day} = {n}. Filtering on the wrong integer "
            f"(another engine's numbering) is the usual mistake here."
        )


class TimezoneDirection(Convention):
    family = "timezone-direction"
    tags = ("time", "timezone", "reasoning")

    def params(self, rng: np.random.Generator) -> dict:
        # A UTC window to count in; the answer depends on getting the conversion direction right.
        windows = [(18, 23), (0, 5), (12, 17), (6, 11)]
        lo, hi = windows[int(rng.integers(0, len(windows)))]
        return {"lo": lo, "hi": hi, "variant": int(rng.integers(0, 2))}

    def _utc_hour(self, domain: Domain) -> pd.Series:
        local_h = domain.df[domain.ts_col].dt.hour
        add = domain.df[domain.location_col].map(US_CITY_UTC_ADD).fillna(0).astype(int)
        return (local_h + add) % 24

    def truth(self, domain: Domain, params: dict) -> int:
        uh = self._utc_hour(domain)
        return int(((uh >= params["lo"]) & (uh <= params["hi"])).sum())

    def question(self, domain: Domain, params: dict) -> str:
        col, loc, lab, lo, hi = (
            domain.ts_col, domain.location_col, domain.label, params["lo"], params["hi"]
        )
        return [
            (f"Each {col} is a LOCAL time in its {loc} (all US cities). Convert each to UTC and "
             f"count how many {lab} fall in UTC hours {lo} through {hi} inclusive. "
             f"Reply with the count."),
            (f"{col} is local time for the {loc}. After converting to UTC, how many {lab} land in "
             f"the UTC hour range {lo}-{hi} (inclusive)? Give the count."),
        ][params["variant"]]

    def sql(self, domain: Domain, dialect: str, params: dict) -> str:
        h = _hour_expr(dialect, domain.ts_col)
        add = _case_utc_add(domain.location_col)
        utc = f"(({h} + {add}) % 24)"
        return (
            f"SELECT count(*) FROM {domain.name} "
            f"WHERE {utc} BETWEEN {params['lo']} AND {params['hi']}"
        )

    def thinking(self, dialect: str, params: dict) -> str:
        return (
            "US local time is behind UTC, so UTC = local + |offset| (e.g. New York is UTC-4, so "
            "ADD 4). Adding the signed offset (local + (-4)) shifts the wrong way and miscounts; "
            f"the CASE maps each city to the hours to add before the {params['lo']}-{params['hi']} "
            "UTC filter."
        )


_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def _month_expr(dialect: str, col: str) -> str:
    return {
        "clickhouse": f"toMonth({col})",
        "duckdb": f"month({col})",
        "postgres": f"EXTRACT(MONTH FROM {col})",
        "mysql": f"MONTH({col})",
    }[dialect]


class WeekendFlag(Convention):
    """The `da_weekend_delay` gap: 'weekend' is Sat+Sun, whose integers differ by dialect."""

    family = "weekend-flag"
    tags = ("date", "weekend", "dialect")

    def params(self, rng: np.random.Generator) -> dict:
        return {"variant": int(rng.integers(0, 3))}

    def truth(self, domain: Domain, params: dict) -> int:
        return int((domain.df[domain.ts_col].dt.dayofweek >= 5).sum())  # Sat=5, Sun=6

    def question(self, domain: Domain, params: dict) -> str:
        col, lab = domain.ts_col, domain.label
        return [
            f"How many {lab} fall on a weekend (Sat or Sun), by {col}? Reply with the count.",
            f"Count the weekend {lab} (Saturday or Sunday {col}).",
            f"Of all {lab}, how many have a {col} on Sat or Sun? Give the number.",
        ][params["variant"]]

    def sql(self, domain: Domain, dialect: str, params: dict) -> str:
        expr = _dow_expr(dialect, domain.ts_col)
        sat, sun = _dow_int(dialect, 5), _dow_int(dialect, 6)
        return f"SELECT count(*) FROM {domain.name} WHERE {expr} IN ({sat}, {sun})"

    def thinking(self, dialect: str, params: dict) -> str:
        sat, sun = _dow_int(dialect, 5), _dow_int(dialect, 6)
        return (
            f"In {DIALECT_DISPLAY[dialect]}, Saturday = {sat}, Sunday = {sun}, so weekend = "
            f"IN ({sat}, {sun}). The mistake is another engine's numbering (e.g. treating Sunday "
            "as 0 when this engine calls it 7, or the reverse)."
        )


class MonthBucket(Convention):
    """Function-name breadth: month numbering is consistent (1-12); the extractor differs."""

    family = "month-bucket"
    tags = ("date", "month", "dialect")

    def params(self, rng: np.random.Generator) -> dict:
        return {"month": int(rng.integers(1, 13)), "variant": int(rng.integers(0, 2))}

    def truth(self, domain: Domain, params: dict) -> int:
        return int((domain.df[domain.ts_col].dt.month == params["month"]).sum())

    def question(self, domain: Domain, params: dict) -> str:
        name, col, lab = _MONTHS[params["month"] - 1], domain.ts_col, domain.label
        return [
            f"How many {lab} have a {col} in {name}? Reply with the count.",
            f"Count the {lab} whose {col} falls in the month of {name}.",
        ][params["variant"]]

    def sql(self, domain: Domain, dialect: str, params: dict) -> str:
        return (
            f"SELECT count(*) FROM {domain.name} "
            f"WHERE {_month_expr(dialect, domain.ts_col)} = {params['month']}"
        )

    def thinking(self, dialect: str, params: dict) -> str:
        name = _MONTHS[params["month"] - 1]
        return (
            f"Month numbering is consistent (January=1 ... December=12), so {name} = "
            f"{params['month']}; the dialect difference is only the extractor "
            f"({_month_expr(dialect, 'ts')})."
        )


ALL_CONVENTIONS: tuple[Convention, ...] = (
    WeekdayNumbering(), WeekendFlag(), TimezoneDirection(), MonthBucket(),
)


def conventions_by_family() -> dict[str, Convention]:
    return {c.family: c for c in ALL_CONVENTIONS}
