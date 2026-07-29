"""Tier-0 invariants (Phase 2, Skill B).

Cheap, objective, cross-field sanity checks on a decoded MetarObservation — the
physical/structural rules the schema (structure-only) deliberately leaves out.
Each rule appends a human-readable violation; `check_invariants` returns them all.
A non-empty result routes the record to the hard-case queue — never dropped, because
those records are exactly the finetune's target material.

These encode spec rules (FMH-1 / WMO), so they hold regardless of which parser
produced the record. That's the point: if every oracle AGREES on a record that
still violates an invariant, it's a strong "the sources share a bug / the data is
genuinely odd" signal — the thing consensus-without-calibration would miss.
"""

from __future__ import annotations

from avtext.schema import MetarObservation

# Generous physical bounds — flag only the absurd, not the merely extreme.
_TEMP_MIN, _TEMP_MAX = -90.0, 60.0  # °C  (record extremes ~ -89 / +57)
_ALT_MIN, _ALT_MAX = 850.0, 1100.0  # hPa (record extremes ~ 870 / 1084)


def check_invariants(obs: MetarObservation) -> list[str]:
    """Return every invariant violation for `obs` (empty list = passes tier-0)."""
    v: list[str] = []

    # Dewpoint can never exceed temperature (saturation is the ceiling).
    if obs.temperature_c is not None and obs.dewpoint_c is not None:
        if obs.dewpoint_c > obs.temperature_c:
            v.append(f"dewpoint {obs.dewpoint_c} > temperature {obs.temperature_c}")

    if obs.wind is not None:
        w = obs.wind
        # A gust is only meaningful if it exceeds the sustained wind.
        if w.gust_kt is not None and w.gust_kt <= w.speed_kt:
            v.append(f"gust {w.gust_kt} <= sustained {w.speed_kt}")
        # A variable range needs both endpoints.
        if (w.var_from is None) != (w.var_to is None):
            v.append("variable wind range has only one endpoint")

    # Cloud layers are reported bottom-up: bases must be non-decreasing. CB/TCU are
    # exempt — they're reported at their actual height and may be appended out of
    # order (a real convention, e.g. Australian "...BKN100 FEW054CB").
    bases = [c.base_ft for c in obs.clouds if c.base_ft is not None and c.cloud_type is None]
    if bases != sorted(bases):
        v.append(f"cloud bases not ascending: {bases}")

    # Plausible physical ranges.
    if obs.temperature_c is not None and not (_TEMP_MIN <= obs.temperature_c <= _TEMP_MAX):
        v.append(f"temperature {obs.temperature_c} out of plausible range")
    if obs.altimeter_hpa is not None and not (_ALT_MIN <= obs.altimeter_hpa <= _ALT_MAX):
        v.append(f"altimeter {obs.altimeter_hpa} out of plausible range")

    # CAVOK asserts vis ≥10 km + no cloud <5000 ft + no significant weather.
    if obs.cavok and (obs.clouds or obs.visibility_m is not None or obs.weather):
        v.append("CAVOK but clouds/visibility/weather present")

    return v
