"""Tier-0 round-trip property test (Hypothesis).

The self-checking oracle: generate a MetarObservation, encode it to a METAR string,
decode it back, and assert the core fields survive. No gold labels needed — the
generated object IS the label. If encode/decode ever drift on wind/temp/altimeter,
Hypothesis finds and shrinks a counterexample.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from avtext.oracles import AvwxOracle
from avtext.quality import encode_metar
from avtext.schema import MetarObservation, Wind

_avwx = AvwxOracle()  # the most tolerant parser — best decode target for the trip


@st.composite
def _observations(draw):
    temp = draw(st.integers(-40, 45))
    dew = draw(st.integers(-50, temp))  # dewpoint <= temp, so the record is valid
    speed = draw(st.integers(0, 55))  # realistic 2-digit sustained wind
    # VRB (variable direction) only occurs at light winds — a 75-kt "variable" wind
    # is not a real METAR, and generating one produces an ambiguous encoding.
    variable = draw(st.booleans()) if 0 < speed <= 6 else False
    direction = None if (variable or speed == 0) else draw(st.integers(1, 36)) * 10
    gust = None if speed < 5 else draw(st.one_of(st.none(), st.integers(speed + 1, speed + 20)))
    return MetarObservation(
        station="EPGD",
        day=draw(st.integers(1, 28)),
        hour=draw(st.integers(0, 23)),
        minute=draw(st.integers(0, 59)),
        wind=Wind(direction=direction, variable=variable, speed_kt=speed, gust_kt=gust),
        temperature_c=float(temp),
        dewpoint_c=float(dew),
        altimeter_hpa=float(draw(st.integers(950, 1050))),
    )


@settings(deadline=None, max_examples=200, derandomize=True)  # deterministic: no flaky CI
@given(_observations())
def test_decode_of_encode_reproduces_core_fields(obs):
    result = _avwx.decode(encode_metar(obs))
    assert result.ok, result.error
    o = result.obs
    assert o.temperature_c == obs.temperature_c
    assert o.dewpoint_c == obs.dewpoint_c
    assert o.altimeter_hpa == obs.altimeter_hpa
    assert o.wind.speed_kt == obs.wind.speed_kt
    if obs.wind.gust_kt is not None:
        assert o.wind.gust_kt == obs.wind.gust_kt
    if not obs.wind.variable and obs.wind.direction is not None:
        assert o.wind.direction == obs.wind.direction
