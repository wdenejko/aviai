"""python-metar adapter (regex-based, long-standing; raises ParserError on failure).

Notably reads the RMK T-group, so its temperature/dewpoint carry 0.1° precision —
the source of the (legitimate) temp disagreements the scope probe found.
"""

from __future__ import annotations

from metar.Metar import Metar

from avtext.oracles.base import (
    Oracle,
    altimeter_source,
    detect_modifiers,
    inhg_to_hpa,
    is_cavok,
    parse_header,
)
from avtext.schema import AltimeterSource, CloudCover, CloudLayer, MetarObservation, SkyClear, Wind

_COVER = {
    "FEW": CloudCover.FEW,
    "SCT": CloudCover.SCT,
    "BKN": CloudCover.BKN,
    "OVC": CloudCover.OVC,
}
_SKY_CLEAR = {"SKC": SkyClear.SKC, "CLR": SkyClear.CLR, "NSC": SkyClear.NSC, "NCD": SkyClear.NCD}


class PythonMetarOracle(Oracle):
    name = "python-metar"

    def _decode(self, raw: str) -> MetarObservation:
        # month=1 (Jan, 31 days) sidesteps python-metar's "day out of range" crash
        # when it assumes the current month for a day-31 report. We take the date
        # from the raw header, never from python-metar's constructed datetime.
        m = Metar(raw, month=1)  # raises metar.Metar.ParserError on genuinely bad input
        rtype, station, day, hour, minute = parse_header(raw)
        auto, cor = detect_modifiers(raw)

        wind = None
        if m.wind_speed is not None:
            wind = Wind(
                direction=int(m.wind_dir.value()) if m.wind_dir else None,
                variable=m.wind_dir is None and m.wind_speed.value("KT") > 0,
                speed_kt=int(round(m.wind_speed.value("KT"))),
                gust_kt=int(round(m.wind_gust.value("KT"))) if m.wind_gust else None,
                var_from=int(m.wind_dir_from.value()) if m.wind_dir_from else None,
                var_to=int(m.wind_dir_to.value()) if m.wind_dir_to else None,
            )

        clouds: list[CloudLayer] = []
        sky_clear = None
        vv = None
        for cover, dist, ctype in m.sky:
            if cover in _COVER:
                clouds.append(
                    CloudLayer(
                        cover=_COVER[cover],
                        base_ft=int(dist.value("FT")) if dist else None,
                        cloud_type=ctype if ctype in ("CB", "TCU") else None,
                    )
                )
            elif cover in _SKY_CLEAR:
                sky_clear = _SKY_CLEAR[cover]
            elif cover == "VV":
                vv = int(dist.value("FT")) if dist else None

        # Canonicalize the altimeter from the parser's READING rather than trusting
        # python-metar's own inHg->hPa constant, which differs from avwx's in the 5th
        # digit and flips integer rounding at the X.50 boundary (A2984 -> 1010.501 here
        # vs 1010.499 in avwx -> 1011 vs 1010, a spurious tie). Converting the inHg
        # reading with the shared canonical constant leaves only genuine misreads to
        # disagree. A Q-report is already native hPa; only an A-report needs conversion.
        alt_src = altimeter_source(raw)
        if m.press is None:
            alt_hpa = None
        elif alt_src is AltimeterSource.A:
            alt_hpa = inhg_to_hpa(m.press.value("IN"))
        else:
            alt_hpa = m.press.value("MB")

        cavok = is_cavok(raw)
        return MetarObservation(
            report_type=rtype,
            station=station,
            day=day,
            hour=hour,
            minute=minute,
            automated=auto,
            corrected=cor,
            cavok=cavok,
            wind=wind,
            visibility_m=None if cavok else (int(m.vis.value("M")) if m.vis else None),
            clouds=clouds,
            sky_clear=sky_clear,
            vertical_visibility_ft=vv,
            temperature_c=m.temp.value("C") if m.temp else None,
            dewpoint_c=m.dewpt.value("C") if m.dewpt else None,
            altimeter_hpa=alt_hpa,
            altimeter_source=alt_src,
            # weather: TODO(v2) — map m.weather tuples into WeatherGroup
        )
