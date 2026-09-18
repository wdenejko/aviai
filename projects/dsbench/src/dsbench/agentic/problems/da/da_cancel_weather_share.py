"""DA (answer-graded, rate-within-group trap): the hub most weather-driven in its cancellations.

CancellationCode is only meaningful for cancelled flights (A=carrier, B=weather, C=NAS, D=security).
The metric is a share WITHIN each hub's cancellations (weather cancels / that hub's total cancels),
not a raw count and not a share of all flights -- a hub can have the most weather cancellations yet
not the highest weather SHARE. The oracle computes the per-hub share in pandas; the reference ranks
the hubs in ClickHouse -- different routes, same winner.
"""
from __future__ import annotations

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = ("ATL", "ORD", "DFW", "DEN", "LAX")
_HUBS_SQL = "('ATL','ORD','DFW','DEN','LAX')"


def _truth(ctx: GradeContext) -> tuple[str, dict[str, float]]:
    df = ctx.client.query_df(
        f"SELECT Origin AS o, CancellationCode AS code FROM aviation.flights "
        f"WHERE Origin IN {_HUBS_SQL} AND Cancelled = 1"
    )
    share = df.groupby("o").apply(lambda g: float((g["code"] == "B").mean()))
    return str(share.idxmax()), {k: round(v, 4) for k, v in share.items()}


def check(ctx: GradeContext) -> tuple[bool, str]:
    truth, shares = _truth(ctx)
    ans = "" if ctx.answer is None else str(ctx.answer).upper()
    got = next((h for h in _HUBS if h in ans), None)
    ok = got == truth
    order = sorted(shares.items(), key=lambda x: -x[1])
    detail = ", ".join(f"{k}={v:.3f}" for k, v in order)
    return ok, "" if ok else f"expected {truth} (weather share by hub: {detail}); got {got!r}"


def reference(ctx: GradeContext):
    res = ctx.client.query(
        f"SELECT Origin FROM aviation.flights WHERE Origin IN {_HUBS_SQL} AND Cancelled = 1 "
        f"GROUP BY Origin ORDER BY countIf(CancellationCode = 'B') / count() DESC LIMIT 1"
    )
    return str(res.result_rows[0][0])


PROMPT = """Among CANCELLED flights (Cancelled = 1) at the five hub airports (ATL, ORD, DFW, DEN,
LAX), the CancellationCode gives the reason: A = carrier, B = weather, C = national air system,
D = security.

For each hub, compute the share of ITS cancellations that were weather-driven (code B). Which hub
has the HIGHEST weather share of its cancellations? Reply with ONLY that hub's 3-letter IATA code.
"""

PROBLEM = AgentProblem(
    id="da_cancel_weather_share", category="da", difficulty="medium",
    title="Hub with the highest weather share of cancellations", prompt=PROMPT,
    check=check, reference=reference, max_steps=10, tags=("rate", "group", "cancellation"),
)
