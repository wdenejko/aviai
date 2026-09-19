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
        return {"weekday": int(rng.integers(0, 7))}

    def truth(self, domain: Domain, params: dict) -> int:
        s = domain.df[domain.ts_col].dt.dayofweek  # Monday=0..Sunday=6
        return int((s == params["weekday"]).sum())

    def question(self, domain: Domain, params: dict) -> str:
        day = _WEEKDAYS[params["weekday"]]
        return (
            f"How many {domain.label} have a {domain.ts_col} that falls on a {day}? "
            f"Reply with the count."
        )

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
        return {"lo": 18, "hi": 23}

    def _utc_hour(self, domain: Domain) -> pd.Series:
        local_h = domain.df[domain.ts_col].dt.hour
        add = domain.df[domain.location_col].map(US_CITY_UTC_ADD).fillna(0).astype(int)
        return (local_h + add) % 24

    def truth(self, domain: Domain, params: dict) -> int:
        uh = self._utc_hour(domain)
        return int(((uh >= params["lo"]) & (uh <= params["hi"])).sum())

    def question(self, domain: Domain, params: dict) -> str:
        return (
            f"Each {domain.ts_col} is a LOCAL time in its {domain.location_col} (all US cities). "
            f"Convert each to UTC and count how many {domain.label} fall in UTC hours "
            f"{params['lo']} through {params['hi']} inclusive. Reply with the count."
        )

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


ALL_CONVENTIONS: tuple[Convention, ...] = (WeekdayNumbering(), TimezoneDirection())


def conventions_by_family() -> dict[str, Convention]:
    return {c.family: c for c in ALL_CONVENTIONS}
