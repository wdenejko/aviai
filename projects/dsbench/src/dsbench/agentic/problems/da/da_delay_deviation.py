"""DA (answer-graded, window): the hub-day whose avg dep delay most exceeds its trailing-7-day mean.

Harder than the single-aggregate DA problems: it needs a per-hub daily series, a *previous-7-days*
window (a frame that excludes the current row), and an argmax across hubs. The oracle recomputes the
deviation independently in pandas (rolling mean over the 7 strictly-prior daily averages) and
compares only the winning (hub, day) — a discrete answer, so no float-tolerance disputes.
`dayOfWeek`/frame conventions are where this model slipped at baseline, so this grades that.
"""
from __future__ import annotations

import re

import pandas as pd

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = "('ATL','ORD','DFW','DEN','LAX')"


def _truth(ctx: GradeContext) -> tuple[str, str, float]:
    df = ctx.client.query_df(
        f"SELECT Origin AS iata, FlightDate AS day, avg(DepDelayMinutes) AS d "
        f"FROM aviation.flights WHERE Origin IN {_HUBS} AND Cancelled=0 GROUP BY iata, day"
    )
    df["day"] = pd.to_datetime(df["day"])
    best: tuple[str, str, float] | None = None
    for hub, g in df.groupby("iata"):
        g = g.sort_values("day")
        dev = g["d"] - g["d"].shift(1).rolling(7).mean()  # mean of the 7 strictly-prior days
        g = g.assign(dev=dev).dropna(subset=["dev"])
        if len(g):
            row = g.loc[g["dev"].idxmax()]
            if best is None or float(row["dev"]) > best[2]:
                best = (str(hub), str(row["day"].date()), float(row["dev"]))
    return best  # type: ignore[return-value]


def check(ctx: GradeContext) -> tuple[bool, str]:
    iata, day, _ = _truth(ctx)
    ans = "" if ctx.answer is None else str(ctx.answer)
    got_iata = next((h for h in ("ATL", "ORD", "DFW", "DEN", "LAX") if h in ans.upper()), None)
    m = re.search(r"\d{4}-\d{2}-\d{2}", ans)
    got_day = m.group(0) if m else None
    ok = got_iata == iata and got_day == day
    return ok, "" if ok else f"expected {iata} {day}; got iata={got_iata} day={got_day}"


def reference(ctx: GradeContext):
    iata, day, _ = _truth(ctx)
    return f"{iata} {day}"


PROMPT = """For the five hub airports (ATL, ORD, DFW, DEN, LAX), consider only departures that
actually departed (Cancelled = 0).

For each hub, compute the average DepDelayMinutes for each calendar day in June 2026. Then, for each
(hub, day) that has a full window of the 7 immediately-preceding calendar days, compute:

    deviation = (that day's average) - (mean of that hub's daily averages over the 7 preceding days)

Across all hubs and days, which (hub, day) has the LARGEST positive deviation?

Your final message must be ONLY: <IATA> <YYYY-MM-DD> (e.g. `ORD 2026-06-15`), nothing else.
"""

PROBLEM = AgentProblem(
    id="da_delay_deviation", category="da", difficulty="hard",
    title="Hub-day delay spike vs trailing 7-day mean", prompt=PROMPT,
    check=check, reference=reference, max_steps=20, tags=("window", "timeseries"),
)
