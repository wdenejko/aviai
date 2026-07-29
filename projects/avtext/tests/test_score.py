"""Scorer tests — the None-vs-value taxonomy, which is the whole point. Abstention and
hallucination are DIFFERENT events (safe miss vs dangerous fabrication) and must never
collapse into one 'not correct' bucket."""

from avtext.harness.score import Outcome, aggregate, score_field, score_record


def test_field_taxonomy():
    assert score_field(16, 16) is Outcome.HIT
    assert score_field(16, 20) is Outcome.WRONG
    assert score_field(16, None) is Outcome.ABSTAIN  # declined — safe
    assert score_field(None, 20) is Outcome.HALLUCINATE  # fabricated — dangerous
    assert score_field(None, None) is Outcome.TRUE_ABSTAIN


def test_clouds_list_tuple_normalised():
    # JSON round-trips the cloud tuple to a list; a matching decode must still be a HIT
    assert score_field([["BKN", 900]], (("BKN", 900),)) is Outcome.HIT


def test_perfect_record_is_exact_match():
    ref = {"temperature_c": 16, "wind_speed": 8}
    m = aggregate([score_record(ref, dict(ref))])
    assert m.precision == 1.0 and m.recall == 1.0 and m.hallucinate == 0
    assert m.exact_match == 1.0


def test_abstention_lowers_recall_but_is_not_hallucination():
    s = score_record({"temperature_c": 16}, {"temperature_c": None})
    assert s.outcomes["temperature_c"] is Outcome.ABSTAIN and not s.exact_match
    m = aggregate([s])
    assert m.recall == 0.0 and m.hallucinate == 0 and m.precision is None  # asserted nothing


def test_hallucination_is_penalised_and_breaks_exact_match():
    # reference has NO gust; the model invents one
    ref = {"temperature_c": 16, "wind_gust": None}
    s = score_record(ref, {"temperature_c": 16, "wind_gust": 25})
    assert s.outcomes["wind_gust"] is Outcome.HALLUCINATE
    assert not s.exact_match
    m = aggregate([s])
    assert m.hallucinate == 1 and m.precision == 0.5  # 1 hit / (1 hit + 1 fabrication)
