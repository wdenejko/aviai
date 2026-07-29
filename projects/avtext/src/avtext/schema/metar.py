"""Canonical decoded-METAR schema (Phase 2, Skill B).

This is the CONTRACT of the whole project: every oracle adapter maps a parser's
output *into* these models, and every scorer compares model output *against* them.
Written from what a METAR means (WMO FM-15 / US FMH-1 / AC 00-45H), deliberately
NOT copied from any parser — copying a parser would launder its choices into our
"ground truth".

Four design decisions worth internalizing (each earned the hard way):

1. CANONICAL UNITS. A METAR reports wind in KT *or* MPS and pressure in hPa (Q)
   *or* inHg (A). We store ONE canonical unit per quantity — wind in **knots**,
   pressure in **hPa**, temperature in **°C** — so cross-oracle comparison and
   scoring need no unit juggling. (The scope probe's phantom "wind disagreements"
   were purely unit mismatches; this field design is the fix.) Where the reported
   representation matters for round-tripping, we keep a small `*_source` field.

2. None MEANS ABSENT. A field is None iff the report doesn't state it. This is
   load-bearing for the eval: hallucination = the model emits a value where gold
   is None; abstention = it correctly leaves None. So None is a real value here,
   never a placeholder.

3. SCHEMA ≠ QUALITY. This layer validates STRUCTURE only — types, ranges, small
   controlled enums. Cross-field PHYSICAL invariants (dewpoint ≤ temp, gust >
   wind) and vocabulary membership against the authoritative WMO/CCT tables live
   in `avtext.quality` (tier-0), not here. Keeping them apart means the schema
   stays a pure container and the invariant suite stays independently testable.

4. DECODE WHAT'S IN THE RAW. The observation time is day/hour/minute (DDHHMMZ) —
   that is all the raw encodes. The full calendar timestamp is external metadata
   (IEM/AWC give it to us); it is NOT something a decoder can derive from the
   report text, so it does not belong in the decoded record.

v1 scope (per the plan): report metadata, wind, visibility, weather, clouds,
temperature/dewpoint, altimeter. Deferred to later iterations (marked TODO):
RVR, directional/variable visibility, structured RMK groups, and the TAF schema.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────────────────────────────────────────────────────────
# Controlled vocabulary — small, structural enums (the schema's job).
# The LARGE weather-phenomena vocabulary is intentionally left as validated
# strings (see WeatherGroup) and checked against the authoritative WMO CCT table
# in the quality layer, so that big table lives in exactly one place.
# ─────────────────────────────────────────────────────────────────────────────


class ReportType(StrEnum):
    METAR = "METAR"  # routine, issued on a fixed schedule
    SPECI = "SPECI"  # special, issued when conditions cross defined thresholds


class CloudCover(StrEnum):
    """Sky cover of a single layer, in oktas ranges."""

    FEW = "FEW"  # 1–2 oktas
    SCT = "SCT"  # 3–4 oktas (scattered)
    BKN = "BKN"  # 5–7 oktas (broken)
    OVC = "OVC"  # 8 oktas (overcast)


class SkyClear(StrEnum):
    """Whole-sky 'no cloud layers' indicators (alternatives to any layers)."""

    SKC = "SKC"  # sky clear (manual observation)
    CLR = "CLR"  # clear below 12,000 ft (automated)
    NSC = "NSC"  # no significant cloud
    NCD = "NCD"  # no cloud detected (automated)


class CloudType(StrEnum):
    CB = "CB"  # cumulonimbus
    TCU = "TCU"  # towering cumulus


class WxIntensity(StrEnum):
    LIGHT = "-"
    MODERATE = ""  # no sign
    HEAVY = "+"


class WxDescriptor(StrEnum):
    MI = "MI"  # shallow
    BC = "BC"  # patches
    PR = "PR"  # partial
    DR = "DR"  # low drifting
    BL = "BL"  # blowing
    SH = "SH"  # showers
    TS = "TS"  # thunderstorm
    FZ = "FZ"  # freezing


class AltimeterSource(StrEnum):
    Q = "Q"  # QNH reported in hPa
    A = "A"  # altimeter reported in inHg


# ─────────────────────────────────────────────────────────────────────────────
# Component models
# ─────────────────────────────────────────────────────────────────────────────


class Wind(BaseModel):
    """Surface wind, normalized to knots.

    Conventions:
      - calm ("00000KT")  -> direction=None, variable=False, speed_kt=0
      - variable ("VRB05KT") -> direction=None, variable=True,  speed_kt=5
      - specific with a variable range ("24012KT 210V270")
                            -> direction=240, var_from=210, var_to=270
    """

    model_config = ConfigDict(extra="forbid")

    direction: int | None = Field(
        default=None, ge=0, le=360, description="degrees true; None if calm/VRB"
    )
    variable: bool = Field(default=False, description="True for VRB (direction varies, no mean)")
    speed_kt: int = Field(ge=0, description="sustained speed, canonical knots")
    gust_kt: int | None = Field(default=None, ge=0)
    var_from: int | None = Field(
        default=None, ge=0, le=360, description="variable-range start (dndndnVdxdxdx)"
    )
    var_to: int | None = Field(default=None, ge=0, le=360)
    source_unit: str = Field(
        default="KT", description="unit as reported (KT|MPS) — kept for round-trip"
    )


class CloudLayer(BaseModel):
    """A single reported cloud layer. VV (vertical visibility) is NOT a layer —
    it lives on the observation as `vertical_visibility_ft`."""

    model_config = ConfigDict(extra="forbid")

    cover: CloudCover
    base_ft: int | None = Field(default=None, ge=0, description="AGL feet; None if unknown ('///')")
    cloud_type: CloudType | None = None


class WeatherGroup(BaseModel):
    """One present-weather group, e.g. '-SHRA' or '+TSRA' or 'VCFG'.

    `descriptor` is a small controlled enum (structural). `phenomena` stays as
    2-letter strings validated against the WMO CCT table in the quality layer —
    the schema does not duplicate that authoritative table.
    """

    model_config = ConfigDict(extra="forbid")

    intensity: WxIntensity = WxIntensity.MODERATE
    in_vicinity: bool = Field(default=False, description="VC — 5–10 SM (8–16 km) from the station")
    descriptor: WxDescriptor | None = None
    phenomena: list[str] = Field(
        default_factory=list, description="e.g. ['RA'], ['FG'] (validated in quality/)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The observation
# ─────────────────────────────────────────────────────────────────────────────


class MetarObservation(BaseModel):
    """A decoded METAR/SPECI — the canonical record all oracles map into."""

    model_config = ConfigDict(extra="forbid")

    # -- report metadata --
    report_type: ReportType = ReportType.METAR
    station: str = Field(pattern=r"^[A-Z][A-Z0-9]{2,3}$", description="ICAO id (upper-case)")
    day: int = Field(ge=1, le=31, description="day-of-month from DDHHMMZ")
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)
    automated: bool = Field(default=False, description="AUTO modifier")
    corrected: bool = Field(default=False, description="COR modifier")

    # -- wind --
    wind: Wind | None = None

    # -- visibility (v1: prevailing only; RVR + directional deferred) --
    cavok: bool = Field(
        default=False, description="vis ≥10 km, no cloud <5000 ft/MSA, no significant wx"
    )
    visibility_m: int | None = Field(
        default=None, ge=0, description="prevailing visibility, canonical metres"
    )
    # TODO(v2): visibility_is_minimum (9999 / P6SM = 'or greater'), directional min vis, RVR.

    # -- present weather (empty list = none reported) --
    weather: list[WeatherGroup] = Field(default_factory=list)

    # -- sky: exactly one of layers / sky_clear / vertical_visibility is populated --
    clouds: list[CloudLayer] = Field(default_factory=list)
    sky_clear: SkyClear | None = None
    vertical_visibility_ft: int | None = Field(default=None, ge=0, description="VV — sky obscured")

    # -- temperature (prefer the RMK T-group's 0.1° precision over the body's integer) --
    temperature_c: float | None = None
    dewpoint_c: float | None = None

    # -- pressure, canonical hPa (A-reports converted from inHg by the adapter) --
    altimeter_hpa: float | None = Field(default=None, gt=0)
    altimeter_source: AltimeterSource | None = Field(
        default=None, description="Q or A, as reported"
    )

    # -- remarks (v1: raw text; structured RMK groups are a v2 target) --
    remarks: str | None = None
