"""DE (state-graded, conditional denominator + rank): per-hub carrier delay-recovery leaderboard.

"Recovery" is a CONDITIONAL rate: among departures that left more than 15 minutes late (and actually
arrived), the share that still arrived on time (ArrDelayMinutes <= 15). The denominator is late
departures, not all flights -- the trap is getting that base right and excluding cancelled / no-show
arrivals. reference builds it with a ClickHouse window; check recomputes the rates from raw and
validates the per-hub ranking is consistent with them.
"""
from __future__ import annotations

import pandas as pd

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = "('ATL','ORD','DFW','DEN','LAX')"
_BASE = (
    f"FROM aviation.flights WHERE Origin IN {_HUBS} AND Cancelled = 0 "
    f"AND ArrDelayMinutes IS NOT NULL AND DepDelayMinutes > 15"
)


def _truth(ctx: GradeContext) -> pd.DataFrame:
    return ctx.client.query_df(
        f"SELECT Origin AS iata, Reporting_Airline AS carrier, count() AS n_late, "
        f"countIf(ArrDelayMinutes <= 15) AS n_recovered, "
        f"round(countIf(ArrDelayMinutes <= 15) / count(), 4) AS recovery_rate "
        f"{_BASE} GROUP BY iata, carrier HAVING n_late >= 100"
    )


def check(ctx: GradeContext) -> tuple[bool, str]:
    truth = _truth(ctx)
    try:
        got = ctx.client.query_df(
            f"SELECT iata, carrier, n_late, n_recovered, recovery_rate, rank "
            f"FROM {ctx.namespace}.recovery"
        )
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.recovery: {str(e)[:150]}"
    if len(got) != len(truth):
        return False, f"expected {len(truth)} rows, got {len(got)}"
    m = truth.merge(got, on=["iata", "carrier"], how="left", suffixes=("_t", ""))
    if m["recovery_rate"].isna().any():
        return False, "some (hub, carrier) rows missing from recovery"
    if (abs(m["recovery_rate"] - m["recovery_rate_t"]) > 0.001).any():
        return False, "recovery_rate values do not match the data"
    if (m["n_late"] != m["n_late_t"]).any():
        return False, "n_late values do not match the data"
    for iata, g in got.groupby("iata"):
        g = g.sort_values("rank")
        if list(g["rank"]) != list(range(1, len(g) + 1)):
            return False, f"{iata}: rank is not 1..{len(g)} contiguous"
        if (g["recovery_rate"].values[:-1] < g["recovery_rate"].values[1:]).any():
            return False, f"{iata}: rank order disagrees with recovery_rate (want DESC)"
    return True, ""


def reference(ctx: GradeContext):
    ctx.client.command(
        f"CREATE TABLE {ctx.namespace}.recovery ENGINE = MergeTree ORDER BY (iata, rank) AS "
        f"SELECT iata, carrier, n_late, n_recovered, recovery_rate, "
        f"row_number() OVER (PARTITION BY iata ORDER BY recovery_rate DESC, carrier ASC) AS rank "
        f"FROM (SELECT Origin AS iata, Reporting_Airline AS carrier, count() AS n_late, "
        f"  countIf(ArrDelayMinutes <= 15) AS n_recovered, "
        f"  round(countIf(ArrDelayMinutes <= 15) / count(), 4) AS recovery_rate "
        f"  {_BASE} GROUP BY iata, carrier HAVING n_late >= 100)"
    )
    return None


PROMPT = """In your scratch database, create a table named exactly `recovery` with columns:
  - iata            (String)   hub IATA code (the origin)
  - carrier         (String)   reporting airline code
  - n_late          (UInt32)   that carrier's late departures from the hub that also arrived
  - n_recovered     (UInt32)   of those, how many still arrived on time (ArrDelayMinutes <= 15)
  - recovery_rate   (Float64)  n_recovered / n_late, rounded to 4 dp
  - rank            (UInt32)   1-based rank within the hub by recovery_rate DESC (ties: carrier asc)

A "late departure" is a non-cancelled flight with DepDelayMinutes > 15 that has a known arrival
(non-null ArrDelayMinutes). Consider the five hubs (ATL, ORD, DFW, DEN, LAX) as origins. Include a
(hub, carrier) row only if it has at least 100 late departures. One row per (hub, carrier). That
table is the deliverable.
"""

PROBLEM = AgentProblem(
    id="de_recovery_leaderboard", category="de", difficulty="hard",
    title="Per-hub carrier delay-recovery leaderboard", prompt=PROMPT,
    check=check, reference=reference, max_steps=18, tags=("rate", "denominator", "rank"),
)
