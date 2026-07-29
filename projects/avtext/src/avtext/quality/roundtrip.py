"""Tier-0 round-trip encoder (Phase 2, Skill B).

`encode_metar(obs)` renders the canonical record back to a METAR string. Paired
with the oracles it gives the round-trip property — `decode(encode(obs))` must
reproduce `obs` on the encoded fields — which is a *self-checking* oracle: it needs
no gold labels, because the input IS the label (Hypothesis generates obs, we assert
they survive the trip).

v1 encodes the unambiguous numeric core: header, wind, temperature/dewpoint, and a
canonical Q-altimeter (hPa). A fixed `9999` visibility is emitted so the string is a
structurally valid METAR the parsers will accept; it isn't asserted on. Encoding
visibility/clouds/weather faithfully grows with the schema — tracked as TODO.
"""

from __future__ import annotations

from avtext.schema import MetarObservation


def _temp(t: float) -> str:
    """16.0 -> '16', -5.0 -> 'M05' (METAR sign convention)."""
    i = int(round(t))
    return f"M{abs(i):02d}" if i < 0 else f"{i:02d}"


def encode_metar(obs: MetarObservation) -> str:
    parts = [obs.report_type.value, obs.station, f"{obs.day:02d}{obs.hour:02d}{obs.minute:02d}Z"]
    if obs.automated:
        parts.append("AUTO")

    if obs.wind is not None:
        w = obs.wind
        head = "VRB" if w.variable else (f"{w.direction:03d}" if w.direction is not None else "000")
        group = f"{head}{w.speed_kt:02d}"
        if w.gust_kt is not None:
            group += f"G{w.gust_kt:02d}"
        parts.append(group + "KT")
        if w.var_from is not None and w.var_to is not None:
            parts.append(f"{w.var_from:03d}V{w.var_to:03d}")

    parts.append("9999")  # filler so the report is structurally valid; not asserted

    if obs.temperature_c is not None and obs.dewpoint_c is not None:
        parts.append(f"{_temp(obs.temperature_c)}/{_temp(obs.dewpoint_c)}")

    if obs.altimeter_hpa is not None:
        parts.append(f"Q{int(round(obs.altimeter_hpa)):04d}")

    return " ".join(parts)
