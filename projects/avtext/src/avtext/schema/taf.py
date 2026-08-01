"""Canonical decoded-TAF schema (Phase 6 — the TAF extension).

A TAF (Terminal Aerodrome Forecast, WMO FM-51 / ICAO Annex 3 / US FMH-1 & AC 00-45H) is a
*forecast*, so where a METAR is one snapshot this is a header plus an ordered sequence of
**change groups** — the prevailing forecast and the ways it is expected to change over the
validity period. That nesting is the whole reason TAF is a better size-discriminator than
METAR (the research doc's point): whole-forecast exact-match requires getting the *structure*
right, not just a dozen scalar fields.

This schema deliberately mirrors the METAR one (schema/metar.py) and reuses its component
models (`Wind`, `CloudLayer`, `WeatherGroup`, `SkyClear`) so a future single model can decode
both into one canonical space. The four METAR design decisions carry over verbatim; two are
worth re-stating because they are load-bearing and TAF-specific:

  • None MEANS "NOT STATED IN THIS GROUP" (not "resolved value"). A change group states only
    the elements that change; unchanged elements carry over *in meteorological reality*. We do
    NOT resolve that inheritance — we decode what the group's text literally contains, exactly
    as for METAR. So a `TEMPO 4SM -RA BR BKN015` group has `wind=None` (no wind token present),
    and that None is a correct decode, not a gap. This keeps the hallucination axis meaningful:
    the model must reproduce the sparsity of the raw, not hallucinate carried-over values.
    (Verified against all three voices — IEM/avwx/mivek all report no wind for that TEMPO.)

  • CANONICAL UNITS, converted at the adapter. Wind in **knots**, visibility in **metres**,
    cloud base in **feet** — same as METAR, so cross-voice voting and cross-product reuse need
    no unit juggling. US TAFs report visibility in **statute miles** (`6SM`, `1 1/2SM`, `P6SM`);
    the adapter converts SM→m. `P6SM` ("more than 6 SM") is the US analogue of METAR's 9999 and
    is captured as `visibility_m` = 6 SM in metres **plus** `visibility_plus=True`, so ">6SM"
    and "exactly 6SM" never collapse to the same value.

Timing model: TAF times are day-of-month + hour (`DDHH`) at period boundaries, with the issue
time and `FM` groups carrying minutes (`DDHHMM`). We store `(day, hour, minute)` on the issue
time and each period's start; `from_minute` is 0 except for `FM`. Windowed groups
(`BECMG`/`TEMPO`/`PROB…`) carry an explicit `to_(day,hour)`; `INITIAL`/`FM` leave it None (their
effective end is the next group's start — a resolution step we leave to the consumer, not the
decoder).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# Reuse the METAR component models verbatim — a forecast group's wind/clouds/weather have the
# same structure as an observation's. Only the *containing* structure (periods) is new.
from avtext.schema.metar import CloudLayer, SkyClear, WeatherGroup, Wind


class ChangeType(StrEnum):
    """The kind of change group. Ordered from 'prevailing' to 'conditional'.

    INITIAL is the base forecast stated right after the validity period; the rest are the
    standard Annex-3 change indicators. PROBnn can stand alone (a probability of a condition)
    or combine with TEMPO (`PROB30 TEMPO` — a 30% chance of a temporary fluctuation)."""

    INITIAL = "INITIAL"  # the prevailing forecast (no change keyword; follows the DDHH/DDHH)
    FM = "FM"  # FROM — a rapid, lasting change at a specific time (FMDDHHMM)
    BECMG = "BECMG"  # BECOMING — a gradual, lasting change across a DDHH/DDHH window
    TEMPO = "TEMPO"  # TEMPORARY — fluctuations <1 h, recurring, within a DDHH/DDHH window
    PROB30 = "PROB30"  # 30% probability of the stated conditions
    PROB40 = "PROB40"  # 40% probability
    PROB30_TEMPO = "PROB30 TEMPO"  # 30% chance of a temporary fluctuation
    PROB40_TEMPO = "PROB40 TEMPO"  # 40% chance of a temporary fluctuation


class WindShear(BaseModel):
    """Non-convective low-level wind shear, `WS<level>/<dir><speed>KT` (e.g. WS020/32035KT =
    shear at 2000 ft AGL, wind 320° 35 kt). Rare but safety-critical, so it's a first-class
    field rather than buried in remarks."""

    model_config = ConfigDict(extra="forbid")

    level_ft: int = Field(ge=0, description="height AGL in feet (WS020 -> 2000)")
    direction: int = Field(ge=0, le=360, description="degrees true")
    speed_kt: int = Field(ge=0, description="canonical knots")


class ForecastPeriod(BaseModel):
    """One change group: its type, when it applies, and the conditions it states.

    Every meteorological field is optional and defaults to 'not stated' (None / empty / False),
    because a change group states only what changes. `change_type=INITIAL` is the exception in
    spirit — it usually states a full set — but the schema does not enforce that, since a decoder
    should never invent a field the text omits."""

    model_config = ConfigDict(extra="forbid")

    change_type: ChangeType

    # -- when this group applies (DDHH, with FM carrying minutes) --
    from_day: int = Field(ge=1, le=31)
    from_hour: int = Field(ge=0, le=24, description="24 is legal in TAF boundaries (end-of-day)")
    from_minute: int = Field(default=0, ge=0, le=59, description="non-zero only for FM groups")
    to_day: int | None = Field(default=None, ge=1, le=31, description="windowed groups only")
    to_hour: int | None = Field(default=None, ge=0, le=24)

    probability: int | None = Field(default=None, description="30 or 40 for PROB groups")

    # -- the forecast conditions (all 'not stated' by default) --
    wind: Wind | None = None
    cavok: bool = Field(default=False, description="CAVOK — vis ≥10 km, no sig cloud/wx")
    visibility_m: int | None = Field(default=None, ge=0, description="prevailing vis, metres")
    visibility_plus: bool = Field(
        default=False, description="P6SM / 'or greater' — vis is a floor, not an exact value"
    )
    weather: list[WeatherGroup] = Field(default_factory=list)
    clouds: list[CloudLayer] = Field(default_factory=list)
    sky_clear: SkyClear | None = Field(default=None, description="SKC/NSC/CLR/NCD (no layers)")
    wind_shear: WindShear | None = None


class TafForecast(BaseModel):
    """A decoded TAF bulletin — the canonical record all TAF oracles map into.

    `periods[0]` is normally the INITIAL group; a NIL or cancelled (CNL) TAF has no periods.
    The header fields decode straight from the lexical prefix `[AMD|COR] <station> DDHHMMZ
    DDHH/DDHH`; the rest is the period sequence."""

    model_config = ConfigDict(extra="forbid")

    station: str = Field(pattern=r"^[A-Z][A-Z0-9]{2,3}$", description="ICAO id (upper-case)")

    # -- issue time (DDHHMMZ) --
    issue_day: int = Field(ge=1, le=31)
    issue_hour: int = Field(ge=0, le=23)
    issue_minute: int = Field(ge=0, le=59)

    # -- validity window (DDHH/DDHH); None for NIL where no window is given --
    valid_from_day: int | None = Field(default=None, ge=1, le=31)
    valid_from_hour: int | None = Field(default=None, ge=0, le=24)
    valid_to_day: int | None = Field(default=None, ge=1, le=31)
    valid_to_hour: int | None = Field(default=None, ge=0, le=24)

    # -- modifiers (literal tokens, like METAR AUTO/COR) --
    amended: bool = Field(default=False, description="AMD — an amended forecast")
    corrected: bool = Field(default=False, description="COR — a corrected forecast")
    is_nil: bool = Field(default=False, description="NIL — no forecast available")
    is_cancelled: bool = Field(default=False, description="CNL — a cancelled forecast")

    periods: list[ForecastPeriod] = Field(default_factory=list)
