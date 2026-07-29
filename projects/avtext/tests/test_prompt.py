"""Prompt/parser tests — the model-agnostic half of the runner. The parser must survive real
LLM output (prose, ```json fences, truncation) and canonicalise into the reference's space."""

from avtext.harness.prompt import format_prompt, parse_prediction


def test_format_prompt_includes_the_raw_and_the_abstain_rule():
    p = format_prompt("EPGD 300000Z 26008KT 9999 BKN009 16/15 Q1005")
    assert "EPGD 300000Z" in p
    assert "Do NOT guess" in p  # the instruction that makes hallucination the model's own fault


def test_parse_clean_json():
    p = parse_prediction('{"temperature_c": 16, "wind_speed": 8, "clouds": [["BKN", 900]]}')
    assert p is not None
    assert p["temperature_c"] == 16 and p["wind_speed"] == 8
    assert p["clouds"] == (("BKN", 900),)  # list -> tuple, comparable to the reference
    assert p["wind_gust"] is None  # missing key -> abstention, not a crash


def test_parse_survives_prose_and_fences():
    txt = 'Sure! Here is the decode:\n```json\n{"temperature_c": 5}\n```\nHope that helps!'
    p = parse_prediction(txt)
    assert p is not None and p["temperature_c"] == 5


def test_parse_invalid_returns_none():
    assert parse_prediction("I cannot decode this report.") is None
    assert parse_prediction('{"temperature_c": ') is None  # truncated -> no balanced object


def test_canonicalisation_matches_reference_space():
    p = parse_prediction('{"visibility_m": 9999, "altimeter_hpa": 1013.4, "temperature_c": null}')
    assert p is not None
    assert p["visibility_m"] == 10000  # bucketed to 100 m, exactly like vote.flatten
    assert p["altimeter_hpa"] == 1013  # rounded to int
    assert p["temperature_c"] is None  # explicit null preserved as abstention
