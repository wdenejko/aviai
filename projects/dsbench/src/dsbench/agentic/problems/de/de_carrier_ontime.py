"""DE (state-graded, group+rank): per-hub on-time leaderboard of carriers.

Multi-step: filter to non-cancelled, aggregate a rate per (hub, carrier), threshold on volume, and
rank within each hub. reference builds it with a ClickHouse window (`row_number() OVER (PARTITION
BY ...)`); check recomputes the rates in pandas and validates the ranking is consistent with them —
grading the ordering, not an exact tie-break, so a defensible tie choice still passes.
"""
from __future__ import annotations

import pandas as pd

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = "('ATL','ORD','DFW','DEN','LAX')"


def _truth(ctx: GradeContext) -> pd.DataFrame:
    return ctx.client.query_df(
        f"SELECT Origin AS iata, Reporting_Airline AS carrier, count() AS n, "
        f"round(countIf(DepDelayMinutes<=15)/count(), 4) AS ontime_rate "
        f"FROM aviation.flights WHERE Origin IN {_HUBS} AND Cancelled=0 "
        f"GROUP BY iata, carrier HAVING n>=100"
    )


def check(ctx: GradeContext) -> tuple[bool, str]:
    truth = _truth(ctx)
    try:
        got = ctx.client.query_df(
            f"SELECT iata, carrier, n, ontime_rate, rank FROM {ctx.namespace}.carrier_ontime"
        )
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.carrier_ontime: {str(e)[:150]}"
    if len(got) != len(truth):
        return False, f"expected {len(truth)} rows, got {len(got)}"
    m = truth.merge(got, on=["iata", "carrier"], how="left", suffixes=("_t", ""))
    if m["ontime_rate"].isna().any():
        return False, "some (hub, carrier) rows missing from carrier_ontime"
    if (abs(m["ontime_rate"] - m["ontime_rate_t"]) > 0.001).any():
        return False, "ontime_rate values do not match the data"
    # rank must be 1..k contiguous per hub and ordered by ontime_rate desc
    for iata, g in got.groupby("iata"):
        g = g.sort_values("rank")
        if list(g["rank"]) != list(range(1, len(g) + 1)):
            return False, f"{iata}: rank is not 1..{len(g)} contiguous"
        if (g["ontime_rate"].values[:-1] < g["ontime_rate"].values[1:]).any():
            return False, f"{iata}: rank order disagrees with ontime_rate"
    return True, ""


def reference(ctx: GradeContext):
    ctx.client.command(
        f"CREATE TABLE {ctx.namespace}.carrier_ontime ENGINE = MergeTree ORDER BY (iata, rank) AS "
        f"SELECT iata, carrier, n, ontime_rate, "
        f"row_number() OVER (PARTITION BY iata ORDER BY ontime_rate DESC, carrier ASC) "
        f"AS rank FROM ("
        f"  SELECT Origin AS iata, Reporting_Airline AS carrier, count() AS n, "
        f"  round(countIf(DepDelayMinutes<=15)/count(), 4) AS ontime_rate "
        f"  FROM aviation.flights WHERE Origin IN {_HUBS} AND Cancelled=0 "
        f"  GROUP BY iata, carrier HAVING n>=100)"
    )
    return None


PROMPT = """In your scratch database, create a table named exactly `carrier_ontime` with columns:
  - iata          (String)   hub IATA code
  - carrier       (String)   reporting airline code
  - n             (UInt32)   that carrier's non-cancelled departures from that hub in June 2026
  - ontime_rate   (Float64)  fraction with DepDelayMinutes <= 15, rounded to 4 dp
  - rank          (UInt32)   1-based rank in the hub by ontime_rate DESC (ties: carrier asc)

Consider the five hubs (ATL, ORD, DFW, DEN, LAX) and only non-cancelled departures (Cancelled = 0).
Include a (hub, carrier) row only if that carrier has at least 100 such departures from that hub.
One row per (hub, carrier). That table is the deliverable.
"""

PROBLEM = AgentProblem(
    id="de_carrier_ontime", category="de", difficulty="hard",
    title="Per-hub carrier on-time leaderboard", prompt=PROMPT,
    check=check, reference=reference, max_steps=18, tags=("group", "rank", "window"),
)
