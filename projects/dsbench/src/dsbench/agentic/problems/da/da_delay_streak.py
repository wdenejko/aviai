"""DA (answer-graded, gaps-and-islands): the longest consecutive-day delay streak at any hub.

For each hub, build its daily average departure delay, then find the longest run of CONSECUTIVE
calendar days on which that average exceeded 20 minutes. Report the longest such run at any hub.
This is a classic gaps-and-islands problem: counting the maximal consecutive stretch, not just how
many days qualify. The oracle scans each hub's series explicitly with a calendar-adjacency guard;
the reference uses a different route (a run-id via cumulative breaks) -- both must agree.
"""
from __future__ import annotations

import re

import pandas as pd

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = "('ATL','ORD','DFW','DEN','LAX')"


def _daily(ctx: GradeContext) -> pd.DataFrame:
    df = ctx.client.query_df(
        f"SELECT Origin AS o, FlightDate AS day, avg(DepDelayMinutes) AS d FROM aviation.flights "
        f"WHERE Origin IN {_HUBS} AND Cancelled = 0 GROUP BY o, day"
    )
    df["day"] = pd.to_datetime(df["day"])
    return df


def check(ctx: GradeContext) -> tuple[bool, str]:
    df = _daily(ctx)
    best = 0
    for _, g in df.groupby("o"):  # explicit scan with a calendar-adjacency guard (route A)
        g = g.sort_values("day")
        run = 0
        prev = None
        for day, over in zip(g["day"], g["d"] > 20, strict=True):
            if over and prev is not None and (day - prev).days == 1:
                run += 1
            elif over:
                run = 1
            else:
                run = 0
            prev = day
            best = max(best, run)
    ans = "" if ctx.answer is None else str(ctx.answer)
    m = re.search(r"\d+", ans)
    got = int(m.group(0)) if m else None
    ok = got == best
    return ok, "" if ok else f"expected longest streak {best} days; got {got!r} (raw {ans!r})"


def reference(ctx: GradeContext):
    df = _daily(ctx)
    best = 0
    for _, g in df.groupby("o"):  # run-id via cumulative breaks (route B), full-June reindex
        g = g.set_index("day").sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        over = (g["d"].reindex(idx) > 20).fillna(False)
        run_id = (~over).cumsum()
        runs = over.groupby(run_id).sum()
        best = max(best, int(runs.max()) if len(runs) else 0)
    return best


PROMPT = """For each of the five hub airports (ATL, ORD, DFW, DEN, LAX), using only departures that
actually departed (Cancelled = 0), compute that hub's average DepDelayMinutes for each calendar day
in June 2026.

For each hub, find the longest run of CONSECUTIVE calendar days on which its daily average exceeded
20 minutes. Across all five hubs, what is the longest such run (in days)? Reply with ONLY the
integer, nothing else.
"""

PROBLEM = AgentProblem(
    id="da_delay_streak", category="da", difficulty="hard",
    title="Longest consecutive-day delay streak at any hub", prompt=PROMPT,
    check=check, reference=reference, max_steps=16, tags=("window", "gaps-islands", "timeseries"),
)
