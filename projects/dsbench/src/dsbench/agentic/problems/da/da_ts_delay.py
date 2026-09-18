"""DA (cross-source, answer-graded): do thunderstorm days delay ORD departures more?

Joins flights to METAR (via the day) — the agent must derive the set of thunderstorm days from raw
METAR text, then partition departure delays by day membership. Trap: it's a conditional average over
a day-set defined in another table, not a single-table aggregate.
"""
from __future__ import annotations

import json

from dsbench.agentic.schema import AgentProblem, GradeContext

# Cancelled = 0: a cancelled flight never departed, so it has no departure delay. Including
# cancelled-but-delayed rows would count their pre-cancellation delay as a "departure delay" and
# shifts the thunderstorm-day average by ~0.2 — the ambiguity that made this problem flip on a
# defensible reading. Pin it to the correct definition and say so in the prompt.
_TRUTH_SQL = """
SELECT round(avgIf(DepDelayMinutes, has(ts, FlightDate)), 1),
       round(avgIf(DepDelayMinutes, NOT has(ts, FlightDate)), 1)
FROM aviation.flights
WHERE Origin = 'ORD' AND Cancelled = 0
""".strip()
_WITH = ("WITH (SELECT groupUniqArray(toDate(valid_utc)) FROM aviation.metar "
         "WHERE station = 'KORD' AND position(raw, 'TS') > 0) AS ts ")


def _truth(ctx: GradeContext) -> tuple[float, float]:
    row = ctx.client.query(_WITH + _TRUTH_SQL).result_rows[0]
    return float(row[0]), float(row[1])


def check(ctx: GradeContext) -> tuple[bool, str]:
    ans = ctx.answer
    if isinstance(ans, str):
        try:
            ans = json.loads(ans)
        except json.JSONDecodeError:
            return False, "answer is not a JSON object"
    if not isinstance(ans, dict) or "ts_day_avg" not in ans or "non_ts_day_avg" not in ans:
        return False, "expected a dict with keys ts_day_avg and non_ts_day_avg"
    ts, non_ts = _truth(ctx)
    try:
        got_ts, got_non = float(ans["ts_day_avg"]), float(ans["non_ts_day_avg"])
    except (TypeError, ValueError):
        return False, "values must be numbers"
    # round the gap to 1 dp before comparing: both sides are already rounded to 1 dp, so a
    # legitimate 0.2 gap must not fail on float noise (0.2 stored as 0.2000000000000028).
    ok = round(abs(got_ts - ts), 1) <= 0.2 and round(abs(got_non - non_ts), 1) <= 0.2
    return ok, "" if ok else f"expected ts/non ~{ts}/{non_ts}; got {got_ts}/{got_non}"


def reference(ctx: GradeContext):
    ts, non_ts = _truth(ctx)
    return {"ts_day_avg": ts, "non_ts_day_avg": non_ts}


PROMPT = """At ORD in June 2026, compare average departure delay on thunderstorm days vs other days.

Definitions:
  - A "thunderstorm day" is a UTC calendar date on which at least one KORD METAR
    (aviation.metar, station 'KORD') has raw text CONTAINING the substring 'TS'.
  - Consider ORD departures that actually departed (aviation.flights, Origin = 'ORD',
    Cancelled = 0); a flight's day is its FlightDate.

Compute avg(DepDelayMinutes) for ORD departures whose FlightDate is a thunderstorm day, and for
those whose FlightDate is NOT, each rounded to 1 decimal. Your final message must be ONLY a JSON
object and nothing else: {"ts_day_avg": <number>, "non_ts_day_avg": <number>}.
"""

PROBLEM = AgentProblem(
    id="da_ts_delay", category="da", difficulty="hard",
    title="Thunderstorm-day vs clear-day delays (ORD)", prompt=PROMPT,
    check=check, reference=reference, max_steps=12, tags=("cross-source", "metar", "conditional"),
)
