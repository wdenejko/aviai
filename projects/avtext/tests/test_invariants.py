"""Tier-0 invariant tests — each rule fires on a violating record and a clean
record passes. (Whether real corpus records trip these is measured separately by
the hard-case scan, not asserted here.)"""

from avtext.quality import check_invariants
from avtext.schema import CloudCover, CloudLayer, MetarObservation, Wind


def _obs(**kw) -> MetarObservation:
    return MetarObservation(station="EPGD", day=1, hour=0, minute=0, **kw)


def test_clean_record_passes():
    obs = _obs(temperature_c=16, dewpoint_c=15, wind=Wind(direction=260, speed_kt=8, gust_kt=20))
    assert check_invariants(obs) == []


def test_dewpoint_above_temperature_flagged():
    assert any("dewpoint" in s for s in check_invariants(_obs(temperature_c=10, dewpoint_c=12)))


def test_gust_not_exceeding_wind_flagged():
    obs = _obs(wind=Wind(direction=260, speed_kt=20, gust_kt=18))
    assert any("gust" in s for s in check_invariants(obs))


def test_non_ascending_cloud_bases_flagged():
    obs = _obs(
        clouds=[
            CloudLayer(cover=CloudCover.BKN, base_ft=3000),
            CloudLayer(cover=CloudCover.SCT, base_ft=1000),
        ]
    )
    assert any("cloud bases" in s for s in check_invariants(obs))


def test_cavok_with_clouds_flagged():
    obs = _obs(cavok=True, clouds=[CloudLayer(cover=CloudCover.FEW, base_ft=2000)])
    assert any("CAVOK" in s for s in check_invariants(obs))


def test_variable_range_single_endpoint_flagged():
    obs = _obs(wind=Wind(direction=260, speed_kt=8, var_from=200))
    assert any("variable" in s for s in check_invariants(obs))
