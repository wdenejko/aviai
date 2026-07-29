"""Schema tests — structural correctness of the canonical MetarObservation.

These are hand-decoded examples (the oracle adapters will produce these
automatically in the next step). The point here is only: does the schema hold the
right shape, and does its structural validation reject malformed input?
"""

import pytest
from pydantic import ValidationError

from avtext.schema import (
    CloudCover,
    CloudLayer,
    MetarObservation,
    ReportType,
    Wind,
)


def test_real_metar_roundtrips_into_the_schema():
    # A real EPGD report from the corpus:
    #   "EPGD 300000Z 26008KT 9999 BKN009 16/15 Q1005"
    obs = MetarObservation(
        station="EPGD",
        day=30,
        hour=0,
        minute=0,
        wind=Wind(direction=260, speed_kt=8),
        visibility_m=9999,  # '9999' == 10 km or more (the '≥' nuance is a v2 TODO)
        clouds=[CloudLayer(cover=CloudCover.BKN, base_ft=900)],  # BKN009 -> 900 ft AGL
        temperature_c=16.0,
        dewpoint_c=15.0,
        altimeter_hpa=1005.0,
    )
    assert obs.report_type is ReportType.METAR  # default
    assert obs.wind.direction == 260 and obs.wind.speed_kt == 8
    assert obs.clouds[0].cover is CloudCover.BKN
    # None-means-absent: nothing said about gusts or weather
    assert obs.wind.gust_kt is None
    assert obs.weather == []


def test_calm_and_variable_wind_conventions():
    calm = Wind(speed_kt=0)  # "00000KT"
    assert calm.direction is None and calm.variable is False
    vrb = Wind(variable=True, speed_kt=3)  # "VRB03KT"
    assert vrb.direction is None and vrb.variable is True


@pytest.mark.parametrize(
    "bad",
    [
        {"station": "EPGD", "day": 30, "hour": 25, "minute": 0},  # hour out of range
        {"station": "epgd", "day": 30, "hour": 0, "minute": 0},  # station not upper-case
        {"station": "EPGD", "day": 0, "hour": 0, "minute": 0},  # day < 1
    ],
)
def test_structural_validation_rejects_bad_input(bad):
    with pytest.raises(ValidationError):
        MetarObservation(**bad)


def test_extra_fields_are_forbidden():
    # A typo'd/unknown field should fail loudly, not be silently dropped —
    # this catches oracle-adapter mapping bugs early.
    with pytest.raises(ValidationError):
        MetarObservation(station="EPGD", day=1, hour=0, minute=0, altimeter_hPa=1013)
