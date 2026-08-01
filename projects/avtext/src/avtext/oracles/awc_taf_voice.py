"""AWC bundled-decode TAF voice — the 3rd consensus voice AND the gold anchor (Phase 6).

Unlike the avwx/mivek adapters, this voice does NOT parse the raw text — it reads AWC's OWN
decoded `<forecast>` blocks out of the stored cache/api XML (the official US-government decode,
the TAF analogue of NOAA's METAR decode we used as gold). It plays two roles:
  1. a third independent vote that breaks the 2-voice ties avwx+mivek leave undecided, and
  2. the gold anchor the consensus is audited against (like gold.py for METAR).

Shape (verified 2026-08-01): each <forecast> has ISO `fcst_time_from/to` (AWC RESOLVES timing,
like avwx), a `change_indicator` (absent on the initial group; FM/BECMG/TEMPO/…),
`wind_dir_degrees`/`wind_speed_kt` (already in knots), `visibility_statute_mi` ('6+' encodes
P6SM), `vert_vis_ft`, `wx_string`, and repeated `<sky_condition sky_cover=.. cloud_base_ft_agl=..>`.

ROLE (revised after measurement): AWC's decode is **SM-native** — it round-trips every visibility
through statute miles, which injects rounding noise on our metres-dominant non-US corpus (adding it
as a co-voter LOWERED the clean rate 99%→97%). And avwx+mivek already agree 99% of the time, so a
tie-breaker isn't needed. So AWC is used as the **gold ANCHOR** (audit the consensus against the
official decode, à la NOAA for METAR), NOT as a co-voting voice. The header comes from the shared
lexical parse; CAVOK isn't encoded by AWC's decode.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from avtext.oracles.base import sm_to_m
from avtext.oracles.taf_base import COVER, SKY_CLEAR, TafResult, change_type, parse_taf_header
from avtext.schema import CloudLayer, ForecastPeriod, TafForecast, Wind


def _ddhh(iso: str | None) -> tuple[int | None, int | None]:
    """AWC period times are ISO ('2026-08-01T06:00:00.000Z'); take (day, hour)."""
    if not iso or len(iso) < 13:
        return None, None
    try:
        return int(iso[8:10]), int(iso[11:13])
    except ValueError:
        return None, None


def _awc_vis(s: str | None) -> tuple[int | None, bool]:
    """AWC visibility is statute miles, '+'-suffixed for 'or greater' ('6+' = P6SM)."""
    if not s:
        return None, False
    plus = "+" in s
    try:
        return int(round(sm_to_m(float(s.replace("+", "").strip())))), plus
    except ValueError:
        return None, plus


def _int(s: str | None) -> int | None:
    try:
        return int(s) if s is not None else None
    except ValueError:
        return None


def decode_awc_taf(taf_el: ET.Element) -> TafForecast | None:
    """Map one <TAF> element's bundled <forecast> decode into a canonical TafForecast."""
    raw = taf_el.findtext("raw_text")
    if not raw:
        return None
    header = parse_taf_header(raw)
    periods: list[ForecastPeriod] = []
    for i, fc in enumerate(taf_el.findall("forecast")):
        fd, fh = _ddhh(fc.findtext("fcst_time_from"))
        if fd is None:
            continue
        td, th = _ddhh(fc.findtext("fcst_time_to"))
        ci = (fc.findtext("change_indicator") or "").upper()
        prob = _int(fc.findtext("probability"))
        if i == 0:
            ct = change_type("INITIAL")
        else:
            ct = change_type(ci, prob=prob, is_tempo=ci == "TEMPO")

        wspd = _int(fc.findtext("wind_speed_kt"))
        wind = None
        if wspd is not None:
            wdir = _int(fc.findtext("wind_dir_degrees"))
            wind = Wind(
                direction=wdir if wdir else None,
                variable=(fc.findtext("wind_dir_degrees") or "").upper() == "VRB",
                speed_kt=wspd,
                gust_kt=_int(fc.findtext("wind_gust_kt")),
            )

        vis_m, vis_plus = _awc_vis(fc.findtext("visibility_statute_mi"))

        clouds: list[CloudLayer] = []
        sky_clear = None
        cavok = False
        for sc in fc.findall("sky_condition"):
            cover = (sc.get("sky_cover") or "").upper()
            if cover in COVER:
                base = _int(sc.get("cloud_base_ft_agl"))
                ctype = sc.get("cloud_type")
                clouds.append(
                    CloudLayer(
                        cover=COVER[cover],
                        base_ft=base,
                        cloud_type=ctype if ctype in ("CB", "TCU") else None,
                    )
                )
            elif cover in SKY_CLEAR:
                sky_clear = SKY_CLEAR[cover]
            elif cover == "CAVOK":
                cavok = True

        periods.append(
            ForecastPeriod(
                change_type=ct,
                from_day=fd,
                from_hour=fh,
                to_day=td,
                to_hour=th,
                probability=prob,
                wind=wind,
                cavok=cavok,
                visibility_m=None if cavok else vis_m,
                visibility_plus=vis_plus,
                clouds=clouds,
                sky_clear=sky_clear,
                vertical_visibility_ft=_int(fc.findtext("vert_vis_ft")),
            )
        )
    return TafForecast(periods=periods, **header)


def awc_taf_result(taf_el: ET.Element) -> TafResult:
    """Wrap decode_awc_taf in a TafResult (uniform with the avwx/mivek voices)."""
    try:
        fc = decode_awc_taf(taf_el)
    except Exception as e:  # noqa: BLE001 — normalise like the other voices
        return TafResult("awc", ok=False, error=f"{type(e).__name__}: {e}")
    return TafResult("awc", ok=fc is not None, forecast=fc, error=None if fc else "no forecast")
