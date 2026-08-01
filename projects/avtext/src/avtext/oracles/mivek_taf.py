"""mivek metar-taf-parser TAF adapter — the second, independent decode voice.

Separate lineage from avwx (different failure mode: mivek raises), and — crucially — a **different
timing convention**: mivek reports each change group's *stated* window (a `BECMG 0107/0109` stays
1:7→1:9), whereas avwx *resolves* it. That divergence is exactly why two voices are worth having,
and it's the thing the consensus layer must reconcile (align by change-type + sequence, not by
exact timestamps).

Structural notes (verified against the corpus 2026-08-01):
  • mivek does NOT emit an INITIAL trend — the base conditions live on the top-level object
    (`t.wind`, `t.visibility`, `t.clouds`, `t.vertical_visibility`); `t.trends` are only the
    change groups. So we synthesise the INITIAL period from the base + the header's validity start.
  • cloud height is already in feet; visibility unit is on `v._unit` ('SM' vs metres).
  • an `FM` trend carries an `FMValidity` with a start but no end (open until the next group);
    `BECMG`/`TEMPO`/`PROB` carry a start+end `Validity`.
Weather groups and wind-shear are deferred (parity with the METAR adapters), left empty.
"""

from __future__ import annotations

from metar_taf_parser.parser.parser import TAFParser

from avtext.oracles.base import mps_to_kt, sm_to_m
from avtext.oracles.taf_base import (
    COVER,
    SKY_CLEAR,
    TafOracle,
    change_type,
    parse_taf_header,
)
from avtext.schema import CloudLayer, ForecastPeriod, TafForecast, Wind


def _name(x: object) -> str:
    return (getattr(x, "name", "") or "").upper()


def _mivek_vis(v: object) -> tuple[int | None, bool]:
    """mivek visibility -> (metres, plus). Handles SM fractions/mixed numbers ('1 1/2'), the
    'P'/'>' 'or greater' marker (P6SM), and metric metres (unit on `v._unit`)."""
    if v is None:
        return None, False
    dist = getattr(v, "distance", None)
    if dist is None:
        return None, False
    s = str(dist)
    plus = s.startswith("P") or ">" in s
    total = 0.0
    for part in (
        s.replace("P6SM", "6")
        .replace("P", "")
        .replace(">", "")
        .replace("<", "")
        .replace("SM", "")
        .split()
    ):
        if "/" in part:
            num, den = part.split("/")
            total += float(num) / float(den)
        else:
            try:
                total += float(part)
            except ValueError:
                pass
    is_sm = "SM" in str(getattr(v, "_unit", "")).upper()
    metres = int(round(sm_to_m(total) if is_sm else total))
    return metres, (plus or (not is_sm and metres >= 9999))


def _wind(w: object) -> Wind | None:
    if w is None or getattr(w, "speed", None) is None:
        return None
    mps = str(getattr(w, "unit", "")).upper() in ("MPS", "M/S")
    spd = mps_to_kt(w.speed) if mps else w.speed
    gust = getattr(w, "gust", None)
    gust_kt = (mps_to_kt(gust) if mps else gust) if gust is not None else None
    deg = getattr(w, "degrees", None)
    return Wind(
        direction=int(deg) if isinstance(deg, int) else None,
        variable=deg is None,
        speed_kt=int(round(spd)),
        gust_kt=int(round(gust_kt)) if gust_kt is not None else None,
    )


def _clouds(items: object) -> tuple[list[CloudLayer], object]:
    """(layers, sky_clear) — mivek gives cloud height directly in feet."""
    layers: list[CloudLayer] = []
    sky_clear = None
    for c in items or []:
        cover = _name(c.quantity)
        if cover in COVER:
            layers.append(CloudLayer(cover=COVER[cover], base_ft=c.height))
        elif cover in SKY_CLEAR:
            sky_clear = SKY_CLEAR[cover]
    return layers, sky_clear


class MivekTafOracle(TafOracle):
    name = "mivek"

    def _decode(self, raw: str) -> TafForecast | None:
        t = TAFParser().parse(raw if raw.strip().startswith("TAF") else f"TAF {raw.strip()}")
        header = parse_taf_header(raw)
        periods: list[ForecastPeriod] = []

        # INITIAL: mivek keeps the base conditions on the top-level object, not as a trend.
        vis_m, vis_plus = _mivek_vis(t.visibility)
        layers, sky_clear = _clouds(t.clouds)
        periods.append(
            ForecastPeriod(
                change_type=change_type("INITIAL"),
                from_day=header["valid_from_day"] or header["issue_day"],
                from_hour=header["valid_from_hour"] or 0,
                wind=_wind(t.wind),
                cavok=bool(getattr(t, "cavok", False)),
                visibility_m=None if getattr(t, "cavok", False) else vis_m,
                visibility_plus=vis_plus,
                clouds=layers,
                sky_clear=sky_clear,
                vertical_visibility_ft=getattr(t, "vertical_visibility", None),
            )
        )

        for tr in t.trends:
            v = tr.validity
            prob = getattr(tr, "probability", None)
            is_tempo = _name(tr.type) == "TEMPO"
            vis_m, vis_plus = _mivek_vis(tr.visibility)
            layers, sky_clear = _clouds(tr.clouds)
            periods.append(
                ForecastPeriod(
                    change_type=change_type(_name(tr.type), prob=prob, is_tempo=is_tempo),
                    from_day=v.start_day,
                    from_hour=v.start_hour,
                    to_day=getattr(v, "end_day", None),
                    to_hour=getattr(v, "end_hour", None),
                    probability=prob,
                    wind=_wind(tr.wind),
                    cavok=bool(getattr(tr, "cavok", False)),
                    visibility_m=vis_m,
                    visibility_plus=vis_plus,
                    clouds=layers,
                    sky_clear=sky_clear,
                    vertical_visibility_ft=getattr(tr, "vertical_visibility", None),
                )
            )

        return TafForecast(periods=periods, **header)
