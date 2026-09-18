"""DA (answer-graded, string-time trap): the local hour-of-day with the worst avg departure delay.

`CRSDepTime` is stored as a 4-char zero-padded STRING ("1830"), not a time or an int. The hour is
its first two characters; a model that treats it as a number (e.g. `CRSDepTime / 100`, or `toHour`
on a string) buckets wrong. The oracle parses the hour in pandas from the raw string; the reference
parses it a different way in ClickHouse (`substring`), so the gate cross-validates the two routes.
"""
from __future__ import annotations

import re

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = "('ATL','ORD','DFW','DEN','LAX')"


def check(ctx: GradeContext) -> tuple[bool, str]:
    df = ctx.client.query_df(
        f"SELECT CRSDepTime AS t, DepDelayMinutes AS d FROM aviation.flights "
        f"WHERE Origin IN {_HUBS} AND Cancelled = 0 AND DepDelayMinutes IS NOT NULL"
    )
    # hour = first two chars of the zero-padded "hhmm" string (independent of any SQL expression)
    df["hour"] = df["t"].astype(str).str.zfill(4).str[:2].astype(int)
    g = df.groupby("hour")["d"].agg(["mean", "size"])
    g = g[g["size"] >= 1000]  # ignore sparse hours (e.g. a handful of 1am red-eyes)
    truth = int(g["mean"].idxmax())
    ans = "" if ctx.answer is None else str(ctx.answer)
    m = re.search(r"\d{1,2}", ans)
    got = int(m.group(0)) if m else None
    ok = got == truth
    return ok, "" if ok else f"expected hour {truth}; got {got!r} (raw {ans!r})"


def reference(ctx: GradeContext):
    res = ctx.client.query(
        f"SELECT toInt16(substring(CRSDepTime, 1, 2)) AS h FROM aviation.flights "
        f"WHERE Origin IN {_HUBS} AND Cancelled = 0 AND DepDelayMinutes IS NOT NULL "
        f"GROUP BY h HAVING count() >= 1000 ORDER BY avg(DepDelayMinutes) DESC LIMIT 1"
    )
    return int(res.result_rows[0][0])


PROMPT = """Across the five hub airports (ATL, ORD, DFW, DEN, LAX), using only departures that
actually departed (Cancelled = 0), bucket each flight by the HOUR of its scheduled departure time.
CRSDepTime is a local 24-hour clock time stored as a string like "1830" (meaning 18:30, hour 18) or
"0705" (hour 7). Consider only hours that have at least 1,000 departures across the hubs (ignore
sparse hours). Which hour of the day (0-23) has the highest average DepDelayMinutes, pooled across
all five hubs? Reply with ONLY the integer hour, nothing else.
"""

PROBLEM = AgentProblem(
    id="da_worst_dep_hour", category="da", difficulty="medium",
    title="Worst local departure hour by delay (hubs)", prompt=PROMPT,
    check=check, reference=reference, max_steps=10, tags=("time", "string", "parse"),
)
