"""DA (answer-graded, NULL/denominator trap): average delay counting cancellations as zero.

Cancelled flights have a NULL DepDelayMinutes. SQL `avg(DepDelayMinutes)` silently skips NULLs, so
its denominator is only the non-cancelled flights. The task defines a cancelled flight as 0 minutes
of delay, so the correct denominator is ALL scheduled flights. A model that just calls
`avg(DepDelayMinutes)` over-states the mean. The oracle computes sum/all-count in pandas; the
reference does it in ClickHouse (`sumIf(...) / count()`), a different route.
"""
from __future__ import annotations

import re

from dsbench.agentic.schema import AgentProblem, GradeContext


def check(ctx: GradeContext) -> tuple[bool, str]:
    df = ctx.client.query_df(
        "SELECT Cancelled AS c, DepDelayMinutes AS d FROM aviation.flights WHERE Origin = 'ORD'"
    )
    # cancelled -> 0 (non-cancelled ORD flights are never null); denominator = all flights.
    s = float(df.loc[df["c"] == 0, "d"].sum())
    truth = round(s / len(df), 2)
    ans = "" if ctx.answer is None else str(ctx.answer)
    m = re.search(r"-?\d+(?:\.\d+)?", ans)
    got = float(m.group(0)) if m else None
    ok = got is not None and round(abs(got - truth), 2) <= 0.05
    return ok, "" if ok else f"expected ~{truth} min; got {got!r} (raw {ans!r})"


def reference(ctx: GradeContext):
    res = ctx.client.query(
        "SELECT sumIf(DepDelayMinutes, Cancelled = 0) / count() FROM aviation.flights "
        "WHERE Origin = 'ORD'"
    )
    return round(float(res.result_rows[0][0]), 2)


PROMPT = """For ALL flights scheduled to depart ORD in June 2026 (Origin = 'ORD'), compute the
average departure delay in minutes, where every CANCELLED flight counts as 0 minutes of delay (it
never departed late). Non-cancelled flights contribute their DepDelayMinutes. The denominator is
every scheduled ORD flight, cancelled or not.

Reply with ONLY the number, rounded to 2 decimals, and nothing else.
"""

PROBLEM = AgentProblem(
    id="da_all_flights_avg_delay", category="da", difficulty="medium",
    title="Avg ORD departure delay counting cancellations as zero", prompt=PROMPT,
    check=check, reference=reference, max_steps=10, tags=("null", "denominator", "dialect"),
)
