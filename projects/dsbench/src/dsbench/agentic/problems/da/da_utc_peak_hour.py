"""DA (answer-graded, timezone trap): the busiest UTC clock-hour for hub departures.

Flight times are LOCAL clock times; the five hubs sit in four different US zones (June/DST). To
histogram departures by UTC hour you must add each hub's offset-to-UTC, and get the SIGN right:
UTC = local - tz_offset, and the offsets are negative, so e.g. ATL (UTC-4) => UTC = local + 4. A
model that adds the signed offset (local + (-4)) shifts the wrong way and picks the wrong peak.
Because the hubs span four zones, no single offset can fake it. The oracle converts in pandas; the
reference converts in ClickHouse (a per-hub `multiIf` delta) — different routes, same answer.
"""
from __future__ import annotations

import re

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = "('ATL','ORD','DFW','DEN','LAX')"
# hours to ADD to local time to get UTC in June (DST): UTC = local + DELTA
_DELTA = {"ATL": 4, "ORD": 5, "DFW": 5, "DEN": 6, "LAX": 7}


def check(ctx: GradeContext) -> tuple[bool, str]:
    df = ctx.client.query_df(
        f"SELECT Origin AS o, CRSDepTime AS t FROM aviation.flights "
        f"WHERE Origin IN {_HUBS} AND Cancelled = 0"
    )
    local_h = df["t"].astype(str).str.zfill(4).str[:2].astype(int)
    utc_h = (local_h + df["o"].map(_DELTA)) % 24
    truth = int(utc_h.value_counts().idxmax())
    ans = "" if ctx.answer is None else str(ctx.answer)
    m = re.search(r"\d{1,2}", ans)
    got = int(m.group(0)) if m else None
    ok = got == truth
    return ok, "" if ok else f"expected UTC hour {truth}; got {got!r} (raw {ans!r})"


def reference(ctx: GradeContext):
    res = ctx.client.query(
        f"SELECT (toInt16(substring(CRSDepTime, 1, 2)) "
        f"  + multiIf(Origin = 'ATL', 4, Origin IN ('ORD','DFW'), 5, Origin = 'DEN', 6, 7)) % 24 "
        f"  AS uh FROM aviation.flights WHERE Origin IN {_HUBS} AND Cancelled = 0 "
        f"GROUP BY uh ORDER BY count() DESC LIMIT 1"
    )
    return int(res.result_rows[0][0])


PROMPT = """Across the five hub airports, using only departures that actually departed
(Cancelled = 0), convert each flight's scheduled departure to UTC and find the busiest UTC hour.

CRSDepTime is a LOCAL clock time stored as a string like "1830" (hour 18). Use these June (DST) UTC
offsets: ATL = UTC-4, ORD = UTC-5, DFW = UTC-5, DEN = UTC-6, LAX = UTC-7.

Which UTC clock-hour (0-23) has the greatest number of hub departures? Reply with ONLY the integer
hour, nothing else.
"""

PROBLEM = AgentProblem(
    id="da_utc_peak_hour", category="da", difficulty="hard",
    title="Busiest UTC departure hour across hubs", prompt=PROMPT,
    check=check, reference=reference, max_steps=12, tags=("timezone", "time", "convert"),
)
