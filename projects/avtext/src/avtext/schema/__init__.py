"""schema — the canonical decoded record (Skill B, Phase 2).

pydantic v2 models for a decoded METAR/TAF: wind, visibility, weather, clouds,
temperature/dewpoint, QNH, and TAF forecast periods.

Write this by *reading FMH-1 / AC 00-45H tables*, not by copying a parser's
output model. This is the step where you actually learn the format — the schema
is the contract every oracle must map into, so its correctness is upstream of
everything else.
"""
