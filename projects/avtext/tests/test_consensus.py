"""Consensus (majority-vote) tests — majority wins, ties are hard cases, and a
field no voice has an opinion on stays undecided."""

from avtext.consensus import consensus
from avtext.schema import MetarObservation, Wind


def _obs(temp: float, speed: int) -> MetarObservation:
    return MetarObservation(
        station="EPGD", day=1, hour=0, minute=0,
        temperature_c=temp, wind=Wind(direction=260, speed_kt=speed),
    )  # fmt: skip


def test_majority_wins_and_is_not_a_hard_case():
    # temperature 16 (x2) vs 15 (x1): a clear majority, so it's decided, not flagged.
    result = consensus([_obs(16, 8), _obs(16, 8), _obs(15, 8)])
    assert result.values["temperature_c"] == 16
    assert result.agreement["temperature_c"] == 2 / 3
    assert "temperature_c" not in result.disagreements


def test_three_way_tie_is_a_hard_case():
    # wind_speed 8/9/10 all differ -> no majority -> undecided + flagged.
    result = consensus([_obs(16, 8), _obs(16, 9), _obs(16, 10)])
    assert result.values["wind_speed"] is None
    assert "wind_speed" in result.disagreements
    # temperature is unanimous in the same records
    assert result.values["temperature_c"] == 16


def test_field_no_voice_reports_stays_undecided():
    blank = [MetarObservation(station="EPGD", day=1, hour=0, minute=0) for _ in range(3)]
    result = consensus(blank)
    assert result.values["temperature_c"] is None
    assert result.agreement["temperature_c"] is None
    assert "temperature_c" not in result.disagreements  # abstention isn't disagreement
