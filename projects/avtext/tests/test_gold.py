"""Gold-calibration tests — consensus scored against authoritative reference records
matches when it should and flags a divergence when the reference disagrees."""

from avtext.consensus.gold import GoldRecord, calibrate

_RAW = "EPGD 300000Z 26008KT 9999 BKN009 16/15 Q1005"


def test_matching_reference_scores_perfect():
    report = calibrate([GoldRecord(_RAW, {"temperature_c": 16, "wind_speed": 8}, "test")])
    assert report.per_field["temperature_c"] == (1, 1, 1.0)
    assert report.per_field["wind_speed"] == (1, 1, 1.0)
    assert report.divergences == []


def test_divergent_reference_is_flagged():
    # a deliberately wrong reference value -> a recorded divergence (consensus 16 vs 99)
    report = calibrate([GoldRecord(_RAW, {"temperature_c": 99}, "test")])
    n, matches, accuracy = report.per_field["temperature_c"]
    assert (n, matches, accuracy) == (1, 0, 0.0)
    assert len(report.divergences) == 1
    assert report.divergences[0][1] == "temperature_c"
