"""DA (answer-graded, Simpson / weighted-average trap): the overall on-time rate across hubs.

"Overall rate across the hubs" is a single pooled rate over all hub departures, which weights each
hub by its volume. The tempting shortcut -- average the five per-hub rates -- gives a different
number (here 0.7238 vs the correct 0.7212), because the hubs carry very different volumes. The
tolerance is tighter than that gap, so an average-of-averages answer fails. The oracle pools in
pandas; the reference pools in ClickHouse -- different routes, same pooled value.
"""
from __future__ import annotations

import re

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = "('ATL','ORD','DFW','DEN','LAX')"


def check(ctx: GradeContext) -> tuple[bool, str]:
    df = ctx.client.query_df(
        f"SELECT DepDelayMinutes AS d FROM aviation.flights "
        f"WHERE Origin IN {_HUBS} AND Cancelled = 0 AND DepDelayMinutes IS NOT NULL"
    )
    truth = round(float((df["d"] <= 15).mean()), 4)
    ans = "" if ctx.answer is None else str(ctx.answer)
    m = re.search(r"\d+(?:\.\d+)?", ans)
    got = float(m.group(0)) if m else None
    ok = got is not None and abs(got - truth) <= 0.0005
    return ok, "" if ok else f"expected pooled rate ~{truth}; got {got!r} (raw {ans!r})"


def reference(ctx: GradeContext):
    res = ctx.client.query(
        f"SELECT countIf(DepDelayMinutes <= 15) / count() FROM aviation.flights "
        f"WHERE Origin IN {_HUBS} AND Cancelled = 0 AND DepDelayMinutes IS NOT NULL"
    )
    return round(float(res.result_rows[0][0]), 4)


PROMPT = """Across all five hub airports (ATL, ORD, DFW, DEN, LAX), using only departures that
actually departed (Cancelled = 0), what is the OVERALL on-time rate across the hubs -- the share of
all those departures with DepDelayMinutes <= 15?

Reply with ONLY the rate as a decimal rounded to 4 places (e.g. 0.7345), and nothing else.
"""

PROBLEM = AgentProblem(
    id="da_weighted_ontime", category="da", difficulty="medium",
    title="Overall (pooled) on-time rate across hubs", prompt=PROMPT,
    check=check, reference=reference, max_steps=10, tags=("weighted", "simpson", "rate"),
)
