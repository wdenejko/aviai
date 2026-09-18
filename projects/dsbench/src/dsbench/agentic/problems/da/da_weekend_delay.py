"""DA (answer-graded, dialect trap): weekend vs weekday average departure delay across the hubs.

Deliberately hinges on the weekday convention the model got wrong at baseline: ClickHouse
`toDayOfWeek` is ISO (1=Mon … 7=Sun), so weekend = 6,7. A model that assumes 1=Sunday buckets the
wrong days and misses. The oracle derives the truth in pandas (`dt.dayofweek`, Mon=0 … Sun=6 →
weekend = 5,6) — a different route from any ClickHouse expression the agent uses.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = "('ATL','ORD','DFW','DEN','LAX')"


def _truth(ctx: GradeContext) -> tuple[float, float]:
    df = ctx.client.query_df(
        f"SELECT FlightDate AS day, DepDelayMinutes AS m FROM aviation.flights "
        f"WHERE Origin IN {_HUBS} AND Cancelled=0 AND DepDelayMinutes IS NOT NULL"
    )
    dow = pd.to_datetime(df["day"]).dt.dayofweek  # Mon=0 … Sun=6
    we = round(float(df.loc[dow >= 5, "m"].mean()), 1)
    wd = round(float(df.loc[dow < 5, "m"].mean()), 1)
    return we, wd


def check(ctx: GradeContext) -> tuple[bool, str]:
    we, wd = _truth(ctx)
    ans = ctx.answer
    if not isinstance(ans, dict):
        if not isinstance(ans, str):
            return False, "expected a JSON object with weekend_avg and weekday_avg"
        m = re.search(r"\{.*\}", ans, re.DOTALL)
        try:
            ans = json.loads(m.group(0) if m else ans)
        except (json.JSONDecodeError, AttributeError):
            return False, "answer is not a JSON object"
    if not isinstance(ans, dict) or "weekend_avg" not in ans or "weekday_avg" not in ans:
        return False, "expected keys weekend_avg and weekday_avg"
    try:
        gwe, gwd = float(ans["weekend_avg"]), float(ans["weekday_avg"])
    except (TypeError, ValueError):
        return False, "values must be numbers"
    ok = round(abs(gwe - we), 1) <= 0.1 and round(abs(gwd - wd), 1) <= 0.1
    return ok, "" if ok else f"expected weekend/weekday ~{we}/{wd}; got {gwe}/{gwd}"


def reference(ctx: GradeContext):
    we, wd = _truth(ctx)
    return {"weekend_avg": we, "weekday_avg": wd}


PROMPT = """Across all five hub airports (ATL, ORD, DFW, DEN, LAX), using only departures that
actually departed (Cancelled = 0), compare the average DepDelayMinutes on WEEKEND days (Sat, Sun)
against WEEKDAYS (Monday-Friday). Bucket each flight by the day of week of its FlightDate.

Your final message must be ONLY a JSON object and nothing else:
{"weekend_avg": <number>, "weekday_avg": <number>}   each rounded to 1 decimal.
"""

PROBLEM = AgentProblem(
    id="da_weekend_delay", category="da", difficulty="medium",
    title="Weekend vs weekday departure delay (hubs)", prompt=PROMPT,
    check=check, reference=reference, max_steps=14, tags=("dialect", "dayofweek"),
)
