"""The Target A convention traps -- the transferable skills Gate 0 measured as gaps (ADR-004).

Two families, both execution-verifiable:

  * `weekday-numbering` -- the DIALECT trap. Every engine numbers weekdays differently
    (ClickHouse ISO 1=Mon; Postgres/DuckDB 0=Sun; MySQL 1=Sun), so the SAME question needs a
    DIFFERENT integer per dialect. The pandas truth is dialect-independent; the per-dialect SQL must
    reproduce it, which is exactly the awareness the model lacks.

  * `timezone-direction` -- the SIGN/direction trap. US local time is behind UTC, so UTC = local +
    |offset| (New York UTC-4 => add 4); the model adds the signed offset the wrong way. The
    arithmetic is identical across dialects, so this trains the REASONING, not a function name.
    The question states the offsets it holds for every row (ADR-004 Revision 2).

A convention exposes: params(rng) [the random draw that fixes the instance], then truth(domain,
params) [pandas, independent], sql(domain, dialect, params) [the label], question / thinking /
system [the prose]. The generator executes sql against a real engine and keeps the row only if it
equals truth.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsbench.sftgen.synth import US_CITY_UTC_OFFSETS, Domain

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


def _case_utc_add(loc_col: str, offsets: dict[str, int]) -> str:
    """A standard-SQL CASE mapping each city to the hours to ADD to reach UTC, i.e. -offset."""
    whens = " ".join(f"WHEN '{city}' THEN {-offset}" for city, offset in offsets.items())
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
        """Engine + schema context (dialect-only). Names the engine so the row is dialect-aware.

        The answer is the SQL only (ADR-004 Revision 2). Revision 1 also asked for "the numeric
        result", which the model cannot know without the data, and the Gate-2 adapter learned to
        invent one.
        """
        cols = ", ".join(f"{c} ({_friendly_type(dt)})" for c, dt in domain.df.dtypes.items())
        return (
            f"You are writing SQL for a {DIALECT_DISPLAY[dialect]} database. "
            f"There is one table `{domain.name}` with columns: {cols}. "
            f"Answer with a single SQL query in a ```sql code block."
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
            f"How many {lab} have a {col} that falls on a {day}?",
            f"Count the {lab} whose {col} is a {day}.",
            f"On {day}s specifically, how many {lab} were there (by {col})?",
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
    """The `da_utc_peak_hour` gap: convert local time to UTC in the right direction.

    ADR-004 Revision 2: the question states each city's UTC offset, and holds it for every row
    whatever the date. Revision 1 converted with June (daylight-time) offsets for timestamps
    across all of 2025 without saying so, so an answer aware of daylight saving disagreed with the
    truth for about five months of every table (reports/gate-evals/20260928-reasoning-pilot.md).
    With the offsets as the premise, the truth is right by construction, and what is left to learn
    is the direction: UTC = local - offset, and US offsets are negative. The set (daylight or
    standard time) is drawn per instance, so the answer comes from reading the question, not from
    a table the model has memorised.
    """

    family = "timezone-direction"
    tags = ("time", "timezone", "reasoning")

    def params(self, rng: np.random.Generator) -> dict:
        # A UTC window to count in; the answer depends on getting the conversion direction right.
        windows = [(18, 23), (0, 5), (12, 17), (6, 11)]
        lo, hi = windows[int(rng.integers(0, len(windows)))]
        return {"lo": lo, "hi": hi, "variant": int(rng.integers(0, 2)),
                "offsets": ("daylight", "standard")[int(rng.integers(0, 2))]}

    def _utc_hour(self, domain: Domain, params: dict) -> pd.Series:
        local_h = domain.df[domain.ts_col].dt.hour
        offset = domain.df[domain.location_col].map(US_CITY_UTC_OFFSETS[params["offsets"]])
        return (local_h - offset.fillna(0).astype(int)) % 24  # UTC = local - offset

    def truth(self, domain: Domain, params: dict) -> int:
        uh = self._utc_hour(domain, params)
        return int(((uh >= params["lo"]) & (uh <= params["hi"])).sum())

    def question(self, domain: Domain, params: dict) -> str:
        col, loc, lab, lo, hi = (
            domain.ts_col, domain.location_col, domain.label, params["lo"], params["hi"]
        )
        kind = {"daylight": "daylight-saving", "standard": "standard"}[params["offsets"]]
        offsets = ", ".join(f"{city} UTC{offset:+d}"
                            for city, offset in US_CITY_UTC_OFFSETS[params["offsets"]].items())
        premise = (f"Take each {loc} at its US {kind} time offset, for every row regardless of "
                   f"date: {offsets}.")
        return [
            (f"Every {col} is a local time in the row's {loc}. {premise} How many {lab} have a "
             f"UTC hour between {lo} and {hi}, inclusive?"),
            (f"{premise} The {col} values are local clock times. Counted in UTC, how many {lab} "
             f"fall in hours {lo} to {hi}, both included?"),
        ][params["variant"]]

    def sql(self, domain: Domain, dialect: str, params: dict) -> str:
        h = _hour_expr(dialect, domain.ts_col)
        add = _case_utc_add(domain.location_col, US_CITY_UTC_OFFSETS[params["offsets"]])
        utc = f"(({h} + {add}) % 24)"
        return (
            f"SELECT count(*) FROM {domain.name} "
            f"WHERE {utc} BETWEEN {params['lo']} AND {params['hi']}"
        )

    def thinking(self, dialect: str, params: dict) -> str:
        city, offset = next(iter(US_CITY_UTC_OFFSETS[params["offsets"]].items()))
        return (
            f"UTC = local time - UTC offset. US offsets are negative ({city} is UTC{offset:+d}), "
            f"so UTC = local + {-offset}: the CASE adds each city's hours before the "
            f"{params['lo']}-{params['hi']} UTC filter. Adding the signed offset "
            f"(local + ({offset})) shifts the wrong way and miscounts."
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
            f"How many {lab} fall on a weekend (Sat or Sun), by {col}?",
            f"Count the weekend {lab} (Saturday or Sunday {col}).",
            f"Of all {lab}, how many have a {col} on Sat or Sun?",
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
            f"How many {lab} have a {col} in {name}?",
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
