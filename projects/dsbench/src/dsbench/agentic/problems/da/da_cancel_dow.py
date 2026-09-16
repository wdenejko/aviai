"""DA (answer-graded): which weekday has the worst cancellation RATE at ORD?

Trap: "worst" is a RATE (cancelled / total that weekday), not a count. The weekday with the most
cancellations need not be the one with the highest rate. check() derives the truth by rate.
"""
from __future__ import annotations

from dsbench.agentic.schema import AgentProblem, GradeContext

DOW = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
       5: "Friday", 6: "Saturday", 7: "Sunday"}


def check(ctx: GradeContext) -> tuple[bool, str]:
    if ctx.answer is None or not str(ctx.answer).strip():
        return False, "no answer provided"
    ans = str(ctx.answer).strip().lower()
    res = ctx.client.query(
        "SELECT toDayOfWeek(FlightDate) d, sum(Cancelled) / count() r "
        "FROM aviation.flights WHERE Origin = 'ORD' GROUP BY d"
    )
    rates = {int(row[0]): float(row[1]) for row in res.result_rows}
    truth_d = max(rates, key=rates.get)
    name = DOW[truth_d]
    ok = ans == name.lower() or ans == str(truth_d) or name.lower() in ans
    order = sorted(rates, key=lambda k: -rates[k])
    detail = ", ".join(f"{DOW[d]}={rates[d] * 100:.2f}%" for d in order)
    return ok, "" if ok else f"expected {name} (cancel rate by weekday: {detail}); got {ans!r}"


def reference(ctx: GradeContext):
    res = ctx.client.query(
        "SELECT toDayOfWeek(FlightDate) d FROM aviation.flights WHERE Origin = 'ORD' "
        "GROUP BY d ORDER BY sum(Cancelled) / count() DESC LIMIT 1"
    )
    return DOW[int(res.result_rows[0][0])]


PROMPT = """For flights DEPARTING ORD (Origin = 'ORD') in June 2026, which day of the week had the
highest CANCELLATION RATE — i.e. cancelled flights divided by total flights for that weekday
(Cancelled is 0/1)? Query the warehouse, then call finish with the weekday NAME
(Monday, Tuesday, ... Sunday) as `answer`.
"""

PROBLEM = AgentProblem(
    id="da_cancel_dow", category="da", difficulty="medium",
    title="Worst cancellation-rate weekday at ORD", prompt=PROMPT,
    check=check, reference=reference, max_steps=8, tags=("sql", "rate", "flights"),
)
