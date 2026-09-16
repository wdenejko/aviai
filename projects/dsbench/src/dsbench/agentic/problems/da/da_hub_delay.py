"""DA (answer-graded): which hub had the worst average departure delay in June 2026?

The simplest end-to-end agentic task — one aggregation over the real warehouse — so it exercises
the whole loop (tool call -> observation -> finish -> grade). check() derives the truth itself and
compares to the agent's finished answer; reference() returns what a correct agent produces.
"""
from __future__ import annotations

from dsbench.agentic.schema import GradeContext

HUBS = ("ATL", "ORD", "DFW", "DEN", "LAX")
_IN = "(" + ", ".join(f"'{h}'" for h in HUBS) + ")"


def check(ctx: GradeContext) -> tuple[bool, str]:
    if ctx.answer is None or not str(ctx.answer).strip():
        return False, "no answer provided"
    ans = str(ctx.answer).strip().upper()
    res = ctx.client.query(
        f"SELECT Origin, avg(DepDelayMinutes) FROM aviation.flights "
        f"WHERE Origin IN {_IN} GROUP BY Origin"
    )
    avgs = {row[0]: float(row[1]) for row in res.result_rows}
    truth = max(avgs, key=avgs.get)
    ok = ans == truth or truth in ans  # tolerate "DFW" or "The answer is DFW."
    detail = ", ".join(f"{k}={v:.1f}" for k, v in sorted(avgs.items(), key=lambda kv: -kv[1]))
    return ok, "" if ok else f"expected {truth} (avg dep delay by hub: {detail}); got {ans!r}"


def reference(ctx: GradeContext):
    res = ctx.client.query(
        f"SELECT Origin FROM aviation.flights WHERE Origin IN {_IN} "
        f"GROUP BY Origin ORDER BY avg(DepDelayMinutes) DESC LIMIT 1"
    )
    return res.result_rows[0][0]


PROMPT = """Among the five hub airports (ATL, ORD, DFW, DEN, LAX), which one had the HIGHEST average
departure delay in June 2026? Use avg(DepDelayMinutes) over flights DEPARTING that airport
(Origin). Query the warehouse to find out, then call finish with the airport's 3-letter IATA code
as `answer`.
"""

from dsbench.agentic.schema import AgentProblem  # noqa: E402

PROBLEM = AgentProblem(
    id="da_hub_delay", category="da", difficulty="medium",
    title="Worst-delay hub (June 2026)", prompt=PROMPT,
    check=check, reference=reference, max_steps=8, tags=("sql", "aggregation"),
)
