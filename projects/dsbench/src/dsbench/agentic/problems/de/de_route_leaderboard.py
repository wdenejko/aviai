"""DE (state-graded, grain + rank + threshold): per-hub worst-destination leaderboard.

The grain is the ROUTE (origin hub -> destination), not the flight: aggregate arrival delay per
(hub, dest), keep routes with enough volume, and rank destinations within each hub by average
arrival delay. reference builds it with a ClickHouse window (`row_number() OVER (PARTITION ...)`);
check recomputes the per-route averages independently and validates the ranking is consistent with
them (ordering, not an exact tie-break), so a defensible tie choice still passes.
"""
from __future__ import annotations

import pandas as pd

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = "('ATL','ORD','DFW','DEN','LAX')"


def _truth(ctx: GradeContext) -> pd.DataFrame:
    return ctx.client.query_df(
        f"SELECT Origin AS iata, Dest AS dest, count() AS n, "
        f"round(avg(ArrDelayMinutes), 2) AS avg_arr_delay "
        f"FROM aviation.flights WHERE Origin IN {_HUBS} AND Cancelled = 0 "
        f"AND ArrDelayMinutes IS NOT NULL GROUP BY iata, dest HAVING n >= 100"
    )


def check(ctx: GradeContext) -> tuple[bool, str]:
    truth = _truth(ctx)
    try:
        got = ctx.client.query_df(
            f"SELECT iata, dest, n, avg_arr_delay, rank FROM {ctx.namespace}.route_delay"
        )
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.route_delay: {str(e)[:150]}"
    if len(got) != len(truth):
        return False, f"expected {len(truth)} rows, got {len(got)}"
    m = truth.merge(got, on=["iata", "dest"], how="left", suffixes=("_t", ""))
    if m["avg_arr_delay"].isna().any():
        return False, "some (hub, dest) routes missing from route_delay"
    if (abs(m["avg_arr_delay"] - m["avg_arr_delay_t"]) > 0.01).any():
        return False, "avg_arr_delay values do not match the data"
    for iata, g in got.groupby("iata"):
        g = g.sort_values("rank")
        if list(g["rank"]) != list(range(1, len(g) + 1)):
            return False, f"{iata}: rank is not 1..{len(g)} contiguous"
        if (g["avg_arr_delay"].values[:-1] < g["avg_arr_delay"].values[1:]).any():
            return False, f"{iata}: rank order disagrees with avg_arr_delay (want DESC)"
    return True, ""


def reference(ctx: GradeContext):
    ctx.client.command(
        f"CREATE TABLE {ctx.namespace}.route_delay ENGINE = MergeTree ORDER BY (iata, rank) AS "
        f"SELECT iata, dest, n, avg_arr_delay, "
        f"row_number() OVER (PARTITION BY iata ORDER BY avg_arr_delay DESC, dest ASC) AS rank "
        f"FROM ("
        f"  SELECT Origin AS iata, Dest AS dest, count() AS n, "
        f"  round(avg(ArrDelayMinutes), 2) AS avg_arr_delay "
        f"  FROM aviation.flights WHERE Origin IN {_HUBS} AND Cancelled = 0 "
        f"  AND ArrDelayMinutes IS NOT NULL GROUP BY iata, dest HAVING n >= 100)"
    )
    return None


PROMPT = """In your scratch database, create a table named exactly `route_delay` with columns:
  - iata            (String)   hub IATA code (the origin)
  - dest            (String)   destination IATA code
  - n               (UInt32)   non-cancelled flights on that route with a known arrival delay
  - avg_arr_delay   (Float64)  average ArrDelayMinutes on the route, rounded to 2 dp
  - rank            (UInt32)   1-based rank within the hub by avg_arr_delay DESC (ties: dest asc)

Consider the five hubs (ATL, ORD, DFW, DEN, LAX) as origins, only non-cancelled flights that have a
non-null ArrDelayMinutes. Include a (hub, dest) route only if it has at least 100 such flights. One
row per (hub, dest). That table is the deliverable.
"""

PROBLEM = AgentProblem(
    id="de_route_leaderboard", category="de", difficulty="hard",
    title="Per-hub worst-destination delay leaderboard", prompt=PROMPT,
    check=check, reference=reference, max_steps=18, tags=("grain", "rank", "window"),
)
