"""Field-level majority vote — tier-1 consensus v1 (Phase 2, Skill B).

Each field is decided independently by the voices that HAVE an opinion (non-None):
the plurality value wins. The hard-case trigger is *no clear majority* (a tie for
the top), NOT mere dissent — a field where 2 of 3 agree yields a confident consensus
even though one voice differs. That distinction matters: mivek is ~1 hPa low on inHg
altimeters, so it dissents on ~22% of US reports, but python-metar+avwx outvote it
every time — resolved, not a hard case. Genuine hard cases (all voices differ) are
rare, and those are the ones worth a human/gold look.

Flattening to comparable canonical values (rounded numerics, cloud tuples) happens
here so voting is unit-agnostic. (v2, later: Dawid-Skene weights each voice by its
estimated reliability instead of one-vote-each.)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from avtext.schema import MetarObservation

# Fields voted on, flattened to hashable canonical values.
FIELDS = (
    "report_type",
    "automated",
    "wind_dir",
    "wind_speed",
    "wind_gust",
    "visibility_m",
    "temperature_c",
    "dewpoint_c",
    "altimeter_hpa",
    "clouds",
    "cavok",
)


def flatten(obs: MetarObservation) -> dict[str, Any]:
    """Canonical, hashable field values for voting (None = the voice abstains)."""
    w = obs.wind
    return {
        "report_type": obs.report_type.value,
        "automated": obs.automated,
        "wind_dir": w.direction if w else None,
        "wind_speed": w.speed_kt if w else None,
        "wind_gust": w.gust_kt if w else None,
        "visibility_m": round(obs.visibility_m / 100) * 100
        if obs.visibility_m is not None
        else None,
        "temperature_c": round(obs.temperature_c) if obs.temperature_c is not None else None,
        "dewpoint_c": round(obs.dewpoint_c) if obs.dewpoint_c is not None else None,
        "altimeter_hpa": round(obs.altimeter_hpa) if obs.altimeter_hpa is not None else None,
        "clouds": tuple((c.cover.value, c.base_ft) for c in obs.clouds),
        "cavok": obs.cavok,
    }


@dataclass
class ConsensusResult:
    values: dict[str, Any]  # consensus value per field (None if no majority / no opinion)
    agreement: dict[str, float | None]  # top-count / #voices-with-opinion, per field
    disagreements: list[str]  # fields with no clear majority -> hard case


def consensus(observations: list[MetarObservation]) -> ConsensusResult:
    flats = [flatten(o) for o in observations]
    values: dict[str, Any] = {}
    agreement: dict[str, float | None] = {}
    disagreements: list[str] = []

    for f in FIELDS:
        opinions = [fl[f] for fl in flats if fl[f] is not None]
        if not opinions:
            values[f], agreement[f] = None, None
            continue
        ranked = Counter(opinions).most_common()
        top_value, top_n = ranked[0]
        agreement[f] = top_n / len(opinions)
        if len(ranked) > 1 and ranked[1][1] == top_n:  # tie for the top -> undecided
            values[f] = None
            disagreements.append(f)
        else:
            values[f] = top_value

    return ConsensusResult(values, agreement, disagreements)
