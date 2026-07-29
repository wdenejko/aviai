"""Oracle-adapter tests — the three parsers map real METARs into the schema
consistently, normalize units, and fail uniformly.

Parametrized across every adapter in ORACLES so a new oracle is covered for free.
Assertions use values all three agree on (no RMK T-group temp, Q-altimeter only) —
the genuine parser differences (mivek inHg rounding, T-group precision) are the
consensus layer's job, not something to pin here.
"""

import pytest

from avtext.oracles import ORACLES
from avtext.schema import CloudCover

_IDS = [o.name for o in ORACLES]


@pytest.mark.parametrize("oracle", ORACLES, ids=_IDS)
def test_clean_metar_decodes_consistently(oracle):
    r = oracle.decode("EPGD 300000Z 26008KT 9999 BKN009 16/15 Q1005")
    assert r.ok, r.error
    o = r.obs
    assert (o.station, o.day, o.hour, o.minute) == ("EPGD", 30, 0, 0)
    assert o.wind.direction == 260 and o.wind.speed_kt == 8 and o.wind.gust_kt is None
    assert o.temperature_c == 16 and o.dewpoint_c == 15
    assert o.altimeter_hpa == 1005  # Q-report: no unit conversion, so all agree
    assert len(o.clouds) == 1
    assert o.clouds[0].cover is CloudCover.BKN and o.clouds[0].base_ft == 900


@pytest.mark.parametrize("oracle", ORACLES, ids=_IDS)
def test_garbage_is_a_uniform_failure(oracle):
    # python-metar/mivek raise, avwx nulls + schema rejects — all surface as ok=False.
    r = oracle.decode("ZZZZ 999999Z NOTAREPORT ///// XX")
    assert not r.ok and r.error


@pytest.mark.parametrize("oracle", ORACLES, ids=_IDS)
def test_mps_wind_normalized_to_knots(oracle):
    # 5 m/s ≈ 9.7 kt, 10 m/s ≈ 19.4 kt — the fix for the probe's phantom disagreements.
    r = oracle.decode("UUEE 011200Z 09005G10MPS 9999 FEW040 10/05 Q1013")
    assert r.ok, r.error
    assert r.obs.wind.speed_kt in (9, 10)
    assert r.obs.wind.gust_kt in (19, 20)


@pytest.mark.parametrize("oracle", ORACLES, ids=_IDS)
def test_leading_cor_modifier_is_handled(oracle):
    # IEM emits the correction flag before the station ("COR EGCC ...").
    r = oracle.decode("COR EGCC 271850Z 15009KT 120V210 9999 OVC045 07/03 Q0974")
    assert r.ok, r.error
    assert r.obs.corrected is True and r.obs.station == "EGCC"
