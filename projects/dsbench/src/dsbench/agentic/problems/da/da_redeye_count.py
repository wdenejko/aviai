"""DA (answer-graded, midnight-wraparound trap): count scheduled red-eye departures.

A red-eye is scheduled to LAND at an earlier clock time than it departs -- i.e. it crosses midnight
(CRSArrTime < CRSDepTime). Both are local "hhmm" strings. The catch is recognising the wraparound
rather than reaching for AirTime or a naive (arr - dep) subtraction that goes negative. The oracle
parses both times to integers in pandas and counts the wraparound; the reference does the comparison
in ClickHouse (`toInt32`), a different route.
"""
from __future__ import annotations

import re

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = "('ATL','ORD','DFW','DEN','LAX')"


def check(ctx: GradeContext) -> tuple[bool, str]:
    df = ctx.client.query_df(
        f"SELECT CRSDepTime AS dep, CRSArrTime AS arr FROM aviation.flights "
        f"WHERE Origin IN {_HUBS} AND Cancelled = 0"
    )
    dep = df["dep"].astype(str).str.zfill(4).astype(int)
    arr = df["arr"].astype(str).str.zfill(4).astype(int)
    truth = int((arr < dep).sum())
    ans = "" if ctx.answer is None else str(ctx.answer)
    m = re.search(r"\d[\d,]*", ans)
    got = int(m.group(0).replace(",", "")) if m else None
    ok = got == truth
    return ok, "" if ok else f"expected {truth} red-eye departures; got {got!r} (raw {ans!r})"


def reference(ctx: GradeContext):
    res = ctx.client.query(
        f"SELECT countIf(toInt32(CRSArrTime) < toInt32(CRSDepTime)) FROM aviation.flights "
        f"WHERE Origin IN {_HUBS} AND Cancelled = 0"
    )
    return int(res.result_rows[0][0])


PROMPT = """Across the five hub airports (ATL, ORD, DFW, DEN, LAX), using only departures that
actually departed (Cancelled = 0), count the SCHEDULED red-eye flights: those scheduled to arrive at
an earlier local clock time than they are scheduled to depart (they cross midnight).

CRSDepTime and CRSArrTime are local clock times stored as "hhmm" strings (e.g. "2330", "0515").
Reply with ONLY the integer count, nothing else.
"""

PROBLEM = AgentProblem(
    id="da_redeye_count", category="da", difficulty="medium",
    title="Scheduled red-eye departures across hubs", prompt=PROMPT,
    check=check, reference=reference, max_steps=10, tags=("time", "midnight", "count"),
)
