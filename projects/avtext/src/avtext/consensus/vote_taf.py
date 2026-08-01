"""TAF field-level consensus — period-aligned majority vote (Phase 6).

The TAF sibling of consensus/vote.py. A TAF is a *sequence of change groups*, so voting happens
per (period, field). The one structural choice (agreed 2026-08-01): periods are aligned **by
sequence** — period i of each voice — NOT by timestamp, because the voices use different timing
conventions (avwx resolves a change-group window; mivek states it). Consequently:

  • TIMING IS NOT VOTED. from/to are taken from the canonical STATED-timing voice (mivek, which
    matches the raw); this keeps the "decode what's stated" rule that governs the schema.
  • EVERYTHING ELSE IS A PLURALITY VOTE among the voices that have an opinion (non-None), exactly
    like the METAR consensus — including the same nearest-100 m visibility flatten, which quietly
    reconciles avwx's 9999 vs mivek's 10000 for "≥10 km".

A record is a STRUCTURAL hard case (the TAF analogue of METAR's 'no clear majority') when the
voices disagree on the number of periods, or on a period's change_type — those escalate to
gold/human rather than silently voting a shape into the reference.

Two-voice caveat: with only avwx + mivek, any field disagreement is a 1–1 tie → undecided (None)
+ flagged. A third voice (AWC's bundled decode, preserved in the raw XML) breaks those ties and
is the natural next addition before an eval freeze.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from avtext.schema import ForecastPeriod, TafForecast

# Votable per-period fields. Timing is excluded (taken from the canonical voice); weather and
# wind_shear are deferred (parity with the adapters, which don't decode them yet).
PERIOD_FIELDS = (
    "change_type",
    "probability",
    "wind_dir",
    "wind_speed",
    "wind_gust",
    "visibility_m",
    "visibility_plus",
    "cavok",
    "clouds",
    "sky_clear",
    "vertical_visibility_ft",
)
TIMING_FIELDS = ("from_day", "from_hour", "from_minute", "to_day", "to_hour")


def flatten_period(p: ForecastPeriod) -> dict[str, Any]:
    """Canonical, hashable per-period field values for voting (None = the voice abstains).
    Visibility is rounded to the nearest 100 m so avwx's 9999 and mivek's 10000 vote alike."""
    w = p.wind
    return {
        "change_type": p.change_type.value,
        "probability": p.probability,
        "wind_dir": w.direction if w else None,
        "wind_speed": w.speed_kt if w else None,
        "wind_gust": w.gust_kt if w else None,
        "visibility_m": round(p.visibility_m / 100) * 100 if p.visibility_m is not None else None,
        # 'plus' is meaningful only alongside a visibility value — abstain when vis is absent,
        # else a voice that reports 9999+ vs one that omits vis reads as a spurious disagreement.
        "visibility_plus": p.visibility_plus if p.visibility_m is not None else None,
        "cavok": p.cavok,
        "clouds": tuple((c.cover.value, c.base_ft) for c in p.clouds),
        "sky_clear": p.sky_clear.value if p.sky_clear else None,
        "vertical_visibility_ft": p.vertical_visibility_ft,
    }


def _vote(opinions: list[Any]) -> tuple[Any, float | None, bool]:
    """(value, agreement, tie). Only voices with an opinion (non-None) vote; a tie for the top
    is undecided → value None. agreement = top-count / #voices-with-opinion."""
    ops = [o for o in opinions if o is not None]
    if not ops:
        return None, None, False
    ranked = Counter(ops).most_common()
    top, n = ranked[0]
    tie = len(ranked) > 1 and ranked[1][1] == n
    return (None if tie else top), n / len(ops), tie


@dataclass
class TafConsensus:
    header: dict  # station + issue + validity + modifiers (identical across voices by construction)
    periods: list[dict]  # per aligned period: voted PERIOD_FIELDS + canonical TIMING_FIELDS
    agreement: dict[tuple[int, str], float | None]  # (period_index, field) -> agreement
    disagreements: list[str]  # "p{i}.{field}" with no clear majority
    structural_dissent: bool  # voices disagree on #periods or a change_type
    n_voices: int


def consensus_taf(by_oracle: dict[str, TafForecast]) -> TafConsensus:
    """Vote a set of decoded TAFs (one per voice) into a consensus reference.

    `by_oracle` maps oracle name -> TafForecast (e.g. {'avwx': …, 'mivek': …}). The header is
    taken from any voice (all share the lexical parse). Periods are aligned over the common
    prefix; a differing period count is itself structural dissent."""
    if not by_oracle:
        raise ValueError("consensus needs at least one voice")
    voices = list(by_oracle.values())
    header = voices[0].model_dump(exclude={"periods"})
    # Prefer mivek's stated timing; fall back to avwx (or whatever's present) if mivek is absent.
    timing_fc = by_oracle.get("mivek") or by_oracle.get("avwx") or voices[0]

    counts = {name: len(fc.periods) for name, fc in by_oracle.items()}
    n = min(counts.values())
    structural = len(set(counts.values())) > 1  # voices disagree on how many change groups

    flats = {name: [flatten_period(p) for p in fc.periods] for name, fc in by_oracle.items()}
    periods: list[dict] = []
    agreement: dict[tuple[int, str], float | None] = {}
    disagreements: list[str] = []

    for i in range(n):
        voted: dict[str, Any] = {}
        for f in PERIOD_FIELDS:
            val, agr, tie = _vote([flats[name][i][f] for name in by_oracle])
            voted[f] = val
            agreement[(i, f)] = agr
            if tie:
                disagreements.append(f"p{i}.{f}")
                if f == "change_type":
                    structural = True  # a type mismatch means the alignment itself is suspect
        # Canonical timing from the stated-timing voice (not voted).
        tp = timing_fc.periods[i]
        for tf in TIMING_FIELDS:
            voted[tf] = getattr(tp, tf)
        periods.append(voted)

    return TafConsensus(
        header=header,
        periods=periods,
        agreement=agreement,
        disagreements=disagreements,
        structural_dissent=structural,
        n_voices=len(by_oracle),
    )
