"""avwx-engine adapter (tolerant; returns nulls rather than raising).

Two quirks handled here: cloud base is reported in *hundreds* of feet (12 -> 1200),
and wind/visibility/altimeter units live on `report.units`, so we convert to the
schema's canonical units from there.
"""

from __future__ import annotations

import avwx

from avtext.oracles.base import (
    Oracle,
    altimeter_source,
    detect_modifiers,
    inhg_to_hpa,
    is_cavok,
    mps_to_kt,
    parse_header,
    sm_to_m,
)
from avtext.schema import CloudCover, CloudLayer, MetarObservation, SkyClear, Wind

_COVER = {
    "FEW": CloudCover.FEW,
    "SCT": CloudCover.SCT,
    "BKN": CloudCover.BKN,
    "OVC": CloudCover.OVC,
}
_SKY_CLEAR = {"SKC": SkyClear.SKC, "CLR": SkyClear.CLR, "NSC": SkyClear.NSC, "NCD": SkyClear.NCD}


def _val(x: object) -> float | None:
    return getattr(x, "value", None)


class AvwxOracle(Oracle):
    name = "avwx"

    def _decode(self, raw: str) -> MetarObservation | None:
        report = avwx.Metar.from_report(raw)
        d = report.data
        if d is None:
            return None
        u = report.units
        rtype, station, day, hour, minute = parse_header(raw)
        auto, cor = detect_modifiers(raw)
        mps = str(u.wind_speed).lower() in ("m/s", "mps")

        wind = None
        wspd = _val(d.wind_speed)
        if wspd is not None:
            wdir = _val(d.wind_direction)
            gust = _val(d.wind_gust)
            var = d.wind_variable_direction or []
            spd_kt = mps_to_kt(wspd) if mps else wspd
            gust_kt = (mps_to_kt(gust) if mps else gust) if gust is not None else None
            wind = Wind(
                direction=int(wdir) if wdir is not None else None,
                variable=getattr(d.wind_direction, "repr", "") == "VRB",
                speed_kt=int(round(spd_kt)),
                gust_kt=int(round(gust_kt)) if gust_kt is not None else None,
                var_from=int(_val(var[0])) if len(var) >= 2 else None,
                var_to=int(_val(var[1])) if len(var) >= 2 else None,
            )

        clouds: list[CloudLayer] = []
        sky_clear = None
        vv = None
        for c in d.clouds:
            t = (c.type or "").upper()
            if t in _COVER:
                clouds.append(
                    CloudLayer(
                        cover=_COVER[t],
                        base_ft=c.base * 100 if c.base is not None else None,  # hundreds -> feet
                        cloud_type=c.modifier if c.modifier in ("CB", "TCU") else None,
                    )
                )
            elif t in _SKY_CLEAR:
                sky_clear = _SKY_CLEAR[t]
            elif t == "VV":
                vv = c.base * 100 if c.base is not None else None

        cavok = is_cavok(raw)
        vis = _val(d.visibility)
        vis_m = None
        if vis is not None and not cavok:
            vis_m = int(round(sm_to_m(vis) if str(u.visibility).lower() in ("sm", "mi") else vis))

        alt = _val(d.altimeter)
        alt_hpa = None
        if alt is not None:
            in_inhg = str(u.altimeter).lower() in ("inhg", "in")
            alt_hpa = inhg_to_hpa(alt) if in_inhg else float(alt)  # keep full precision

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
            visibility_m=vis_m,
            clouds=clouds,
            sky_clear=sky_clear,
            vertical_visibility_ft=vv,
            temperature_c=_val(d.temperature),
            dewpoint_c=_val(d.dewpoint),
            altimeter_hpa=alt_hpa,
            altimeter_source=altimeter_source(raw),
            # weather: TODO(v2) — map d.wx_codes into WeatherGroup
        )
