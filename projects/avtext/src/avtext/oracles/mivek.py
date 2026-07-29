"""mivek metar-taf-parser adapter (separate lineage; raises on bad input).

mivek wants the report body WITHOUT the leading METAR/SPECI keyword, exposes cover
and cloud type as enums (mapped by name), gives cloud height directly in feet, and
normalizes altimeter to integer hPa (so A-reports arrive pre-converted).
"""

from __future__ import annotations

from metar_taf_parser.parser.parser import MetarParser

from avtext.oracles.base import (
    Oracle,
    altimeter_source,
    detect_modifiers,
    is_cavok,
    mps_to_kt,
    parse_header,
    sm_to_m,
)
from avtext.schema import (
    AltimeterSource,
    CloudCover,
    CloudLayer,
    CloudType,
    MetarObservation,
    SkyClear,
    Wind,
)

_COVER = {
    "FEW": CloudCover.FEW,
    "SCT": CloudCover.SCT,
    "BKN": CloudCover.BKN,
    "OVC": CloudCover.OVC,
}
_SKY_CLEAR = {"SKC": SkyClear.SKC, "CLR": SkyClear.CLR, "NSC": SkyClear.NSC, "NCD": SkyClear.NCD}


def _name(x: object) -> str:
    return (getattr(x, "name", "") or "").upper()


def _mivek_type(t: object) -> CloudType | None:
    n = _name(t)
    if "CB" in n or "CUMULONIMB" in n:
        return CloudType.CB
    if "TCU" in n or "TOWER" in n:
        return CloudType.TCU
    return None


def _mivek_vis(v: object) -> int | None:
    # mivek gives distance as "10", a fraction "1/8", a mixed number "1 1/2",
    # or ">10000" (its rendering of 9999). Sum whole + fraction parts.
    dist = getattr(v, "distance", None) or getattr(v, "_distance", None)
    if dist is None:
        return None
    total = 0.0
    for part in str(dist).replace(">", "").replace("<", "").split():
        if "/" in part:
            num, den = part.split("/")
            total += float(num) / float(den)
        else:
            total += float(part)
    is_sm = "SM" in str(getattr(v, "_unit", "M")).upper()
    return int(round(sm_to_m(total) if is_sm else total))


class MivekOracle(Oracle):
    name = "mivek"

    def _decode(self, raw: str) -> MetarObservation:
        toks = raw.split()
        while toks and toks[0] in ("METAR", "SPECI", "COR", "AMD"):
            toks.pop(0)  # mivek wants the body starting at the station id
        m = MetarParser().parse(" ".join(toks))  # raises on bad input
        rtype, station, day, hour, minute = parse_header(raw)
        auto, cor = detect_modifiers(raw)

        wind = None
        w = m.wind
        if w is not None and getattr(w, "speed", None) is not None:
            mps = str(getattr(w, "unit", "")).upper() in ("MPS", "M/S")
            spd = mps_to_kt(w.speed) if mps else w.speed
            gust = getattr(w, "gust", None)
            gust_kt = (mps_to_kt(gust) if mps else gust) if gust is not None else None
            deg = getattr(w, "degrees", None)
            wind = Wind(
                direction=int(deg) if isinstance(deg, int) else None,
                variable=deg is None,
                speed_kt=int(round(spd)),
                gust_kt=int(round(gust_kt)) if gust_kt is not None else None,
                var_from=getattr(w, "min_variation", None),
                var_to=getattr(w, "max_variation", None),
            )

        # mivek exposes altimeter only as an integer hPa. For a Q-report that integer
        # is exact; for an A-report (inHg) mivek TRUNCATES the conversion (30.14 inHg =
        # 1020.66 hPa -> 1020), a systematic ~0.5-1 hPa low bias. Rather than cast a
        # biased vote at a precision it cannot represent, mivek ABSTAINS on inHg
        # altimeters (None) and stays a full voice on hPa reports. python-metar and avwx
        # carry the inHg case at full precision; the gold seed validates that they're
        # right. (Discovered mining Phase 3: this quirk was 96% of all "dissent".)
        alt_src = altimeter_source(raw)
        alt_hpa = (
            float(m.altimeter)
            if (m.altimeter is not None and alt_src is AltimeterSource.Q)
            else None
        )

        clouds: list[CloudLayer] = []
        sky_clear = None
        for c in m.clouds:
            cover = _name(c.quantity)
            if cover in _COVER:
                clouds.append(
                    CloudLayer(
                        cover=_COVER[cover], base_ft=c.height, cloud_type=_mivek_type(c.type)
                    )
                )
            elif cover in _SKY_CLEAR:
                sky_clear = _SKY_CLEAR[cover]

        return MetarObservation(
            report_type=rtype,
            station=station,
            day=day,
            hour=hour,
            minute=minute,
            automated=auto or bool(getattr(m, "auto", False)),
            corrected=cor or bool(getattr(m, "corrected", False)),
            cavok=is_cavok(raw) or bool(getattr(m, "cavok", False)),
            wind=wind,
            visibility_m=None
            if is_cavok(raw)
            else _mivek_vis(m.visibility)
            if m.visibility
            else None,
            clouds=clouds,
            sky_clear=sky_clear,
            vertical_visibility_ft=getattr(m, "vertical_visibility", None),
            temperature_c=float(m.temperature) if m.temperature is not None else None,
            dewpoint_c=float(m.dew_point) if m.dew_point is not None else None,
            altimeter_hpa=alt_hpa,
            altimeter_source=alt_src,
            # weather: TODO(v2) — map m.weather_conditions into WeatherGroup
        )
