"""DA (answer-graded, domain/denominator trap): share of delay-minutes from late-arriving aircraft.

The five BTS contributory-cause fields (Carrier/Weather/NAS/Security/LateAircraft) are populated
ONLY for arrivals delayed 15+ minutes; otherwise they are null. The denominator that makes a "share"
meaningful is the sum of those five causes -- NOT total ArrDelayMinutes, NOT a per-flight average.
A model that divides by the wrong base gets a different number. The oracle sums in pandas (nulls ->
0); the reference sums in ClickHouse (`sum` skips nulls) -- different routes, same share.
"""
from __future__ import annotations

import re

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = "('ATL','ORD','DFW','DEN','LAX')"
_CAUSES = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]


def check(ctx: GradeContext) -> tuple[bool, str]:
    cols = ", ".join(f"{c} AS {c.lower()}" for c in _CAUSES)
    df = ctx.client.query_df(f"SELECT {cols} FROM aviation.flights WHERE Origin IN {_HUBS}")
    df = df.fillna(0.0)
    total = float(df.sum().sum())
    late = float(df["lateaircraftdelay"].sum())
    truth = round(100.0 * late / total, 1)
    ans = "" if ctx.answer is None else str(ctx.answer)
    m = re.search(r"\d+(?:\.\d+)?", ans)
    got = float(m.group(0)) if m else None
    ok = got is not None and abs(got - truth) <= 0.2
    return ok, "" if ok else f"expected ~{truth}%; got {got!r} (raw {ans!r})"


def reference(ctx: GradeContext):
    num = "sum(LateAircraftDelay)"
    den = " + ".join(f"sum({c})" for c in _CAUSES)
    res = ctx.client.query(
        f"SELECT 100 * {num} / ({den}) FROM aviation.flights WHERE Origin IN {_HUBS}"
    )
    return round(float(res.result_rows[0][0]), 1)


PROMPT = """Across the five hub airports (ATL, ORD, DFW, DEN, LAX), consider the five contributory
cause fields: CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay. These are
populated only for arrivals that were delayed 15 or more minutes (otherwise null).

Of the total delay-minutes summed across ALL FIVE causes, what percentage is attributable to
LateAircraftDelay? Reply with ONLY the number rounded to 1 decimal (e.g. 41.7), and nothing else.
"""

PROBLEM = AgentProblem(
    id="da_delay_attribution", category="da", difficulty="hard",
    title="Share of hub delay-minutes from late-arriving aircraft", prompt=PROMPT,
    check=check, reference=reference, max_steps=12, tags=("attribution", "null", "share"),
)
