"""avwx-engine TAF adapter — the primary TAF decode voice.

Maps `avwx.Taf` output into the canonical TafForecast. Quirks handled (all verified against the
live corpus 2026-08-01):
  • forecast[0] is the INITIAL group but avwx labels it 'FROM' like any FM — so we type index 0
    as INITIAL and the rest by their own label.
  • cloud base is in *hundreds* of feet (SCT040 -> base 40 -> 4000 ft); 'VV' is not a layer, it's
    the period's vertical visibility.
  • visibility unit is per-report (`units.visibility`): US TAFs 'sm', most others 'm'. CAVOK
    arrives as visibility.repr == 'CAVOK'; P6SM as repr 'P6' with a null value.
  • wind unit is per-report too ('kt' or 'm/s') — MPS is converted to knots.

Known v1 limitation (documented, reconciled at consensus time): **avwx RESOLVES change-group
timing** — a `BECMG 0107/0109` is reported as the resolved window (0109 → next change), not the
stated transition window. So this voice's period `from/to` reflect avwx's interpretation, and
`from_minute` is always 0 (avwx exposes DDHH only). Weather groups and wind-shear are deferred
(weather is not yet scored for METAR either — parity), left empty rather than half-decoded.
"""

from __future__ import annotations

import avwx

from avtext.oracles.base import mps_to_kt
from avtext.oracles.taf_base import (
    COVER,
    SKY_CLEAR,
    TafOracle,
    change_type,
    parse_taf_header,
    visibility_from_sm,
)
from avtext.schema import CloudLayer, ForecastPeriod, TafForecast, Wind


def _val(x: object) -> float | None:
    return getattr(x, "value", None)


def _ddhh(t: object) -> tuple[int | None, int | None]:
    """avwx exposes period times as a 'DDHH' repr; split to (day, hour). None if absent."""
    r = getattr(t, "repr", None)
    if not r or len(r) != 4 or not r.isdigit():
        return None, None
    return int(r[:2]), int(r[2:])


class AvwxTafOracle(TafOracle):
    name = "avwx"

    def _decode(self, raw: str) -> TafForecast | None:
        report = avwx.Taf.from_report(raw)
        d = report.data
        if d is None:
            return None
        header = parse_taf_header(raw)
        vis_unit = str(report.units.visibility).lower()
        mps = str(report.units.wind_speed).lower() in ("m/s", "mps")

        periods: list[ForecastPeriod] = []
        for i, fc in enumerate(d.forecast):
            fd, fh = _ddhh(fc.start_time)
            td, th = _ddhh(fc.end_time)
            if fd is None:  # a period we can't place in time is unusable
                continue
            prob = _val(fc.probability)
            if i == 0:
                ct = change_type("INITIAL")
            else:
                ct = change_type(
                    fc.type, prob=int(prob) if prob else None, is_tempo=fc.type.upper() == "TEMPO"
                )

            # -- wind --
            wind = None
            wspd = _val(fc.wind_speed)
            if wspd is not None:
                wdir = _val(fc.wind_direction)
                gust = _val(fc.wind_gust)
                spd_kt = mps_to_kt(wspd) if mps else wspd
                gust_kt = (mps_to_kt(gust) if mps else gust) if gust is not None else None
                wind = Wind(
                    direction=int(wdir) if wdir is not None else None,
                    variable=getattr(fc.wind_direction, "repr", "") == "VRB",
                    speed_kt=int(round(spd_kt)),
                    gust_kt=int(round(gust_kt)) if gust_kt is not None else None,
                )

            # -- visibility (+ CAVOK / plus) --
            vis_m: int | None = None
            vis_plus = False
            cavok = False
            v = fc.visibility
            if v is not None:
                if (v.repr or "").upper() == "CAVOK":
                    cavok = True
                elif vis_unit in ("sm", "mi"):
                    vis_m, vis_plus = visibility_from_sm(v.value, v.repr)
                elif v.value is not None:
                    vis_m = int(v.value)
                    vis_plus = vis_m >= 9999  # 9999 m = "10 km or more"

            # -- clouds / sky-clear / vertical visibility --
            clouds: list[CloudLayer] = []
            sky_clear = None
            vv = None
            for c in fc.clouds:
                t = (c.type or "").upper()
                if t in COVER:
                    clouds.append(
                        CloudLayer(
                            cover=COVER[t],
                            base_ft=c.base * 100 if c.base is not None else None,
                            cloud_type=c.modifier if c.modifier in ("CB", "TCU") else None,
                        )
                    )
                elif t in SKY_CLEAR:
                    sky_clear = SKY_CLEAR[t]
                elif t == "VV":
                    vv = c.base * 100 if c.base is not None else None

            periods.append(
                ForecastPeriod(
                    change_type=ct,
                    from_day=fd,
                    from_hour=fh,
                    to_day=td,
                    to_hour=th,
                    probability=int(prob) if prob else None,
                    wind=wind,
                    cavok=cavok,
                    visibility_m=vis_m,
                    visibility_plus=vis_plus,
                    clouds=clouds,
                    sky_clear=sky_clear,
                    vertical_visibility_ft=vv,
                    # weather / wind_shear: TODO(v2) — deferred (weather unscored for METAR too)
                )
            )

        return TafForecast(periods=periods, **header)
