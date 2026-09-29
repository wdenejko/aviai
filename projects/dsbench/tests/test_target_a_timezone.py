"""Tests for Target A's timezone family after ADR-004 Revision 2: the offsets are the premise.

Revision 1 scored answers against June offsets for timestamps across all of 2025 and never said
so. Now the question states the offsets and holds them for every row, so the truth must follow from
what the question says, and nothing else.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
from dsbench.sftgen import decontaminate as dc
from dsbench.sftgen import synth
from dsbench.sftgen.conventions import TimezoneDirection
from dsbench.sftgen.engines import DuckDBEngine

TZ = TimezoneDirection()
WINDOWS = [(18, 23), (0, 5), (12, 17), (6, 11)]


def _params(offsets, variant=0, window=(18, 23)):
    return {"lo": window[0], "hi": window[1], "variant": variant, "offsets": offsets}


def _stated(question: str) -> dict[str, int]:
    """The offsets a question states, read back from its text."""
    pairs = re.findall(r"([A-Z][a-z]+(?: [A-Z][a-z]+)?) UTC([+-]\d+)", question)
    return {city: int(offset) for city, offset in pairs}


@pytest.fixture(scope="module")
def domain():
    return synth.build("payments", seed=20260929, n=1500)


@pytest.mark.parametrize("offsets", ["daylight", "standard"])
@pytest.mark.parametrize("variant", [0, 1])
def test_the_question_states_every_citys_offset_for_every_row(domain, offsets, variant):
    question = TZ.question(domain, _params(offsets, variant))
    assert _stated(question) == synth.US_CITY_UTC_OFFSETS[offsets]
    assert "for every row regardless of date" in question


@pytest.mark.parametrize("offsets", ["daylight", "standard"])
@pytest.mark.parametrize("window", WINDOWS)
def test_the_truth_follows_from_what_the_question_states(domain, offsets, window):
    params = _params(offsets, window=window)
    stated = _stated(TZ.question(domain, params))
    local = domain.df[domain.ts_col].dt.hour
    utc = (local - domain.df[domain.location_col].map(stated)) % 24  # UTC = local - offset
    assert TZ.truth(domain, params) == int(utc.between(*window).sum())


@pytest.mark.parametrize("offsets", ["daylight", "standard"])
@pytest.mark.parametrize("window", WINDOWS)
def test_the_gold_sql_returns_the_truth_on_duckdb(domain, offsets, window):
    params = _params(offsets, window=window)
    engine = DuckDBEngine()
    engine.setup()
    try:
        engine.load(domain.name, domain.df)
        assert engine.scalar(TZ.sql(domain, "duckdb", params)) == TZ.truth(domain, params)
    finally:
        engine.teardown()


def test_the_stated_set_decides_the_answer(domain):
    # A model that ignores the question and applies a memorised table gets one of the two wrong.
    differs = [TZ.truth(domain, _params("daylight", window=w))
               != TZ.truth(domain, _params("standard", window=w)) for w in WINDOWS]
    assert all(differs)


def test_the_hint_names_the_direction_with_the_stated_set():
    hint = TZ.thinking("clickhouse", _params("standard"))
    assert "New York is UTC-5" in hint and "UTC = local + 5" in hint


def test_the_prose_shares_no_13_gram_with_dsbench(domain):
    # dsbench's `da_utc_peak_hour` states offsets too; decontamination would reject a copy.
    deny = dc.build_denylist()
    for offsets in ("daylight", "standard"):
        for variant in (0, 1):
            params = _params(offsets, variant)
            for text in (TZ.question(domain, params), TZ.thinking("clickhouse", params)):
                assert dc.scan_text(text, deny) is None


def test_the_synthetic_tables_are_unchanged():
    # Gate-2 rows are re-verified by rebuilding their table from (domain, seed, n), so the data a
    # seed draws must not move. Fingerprint recorded before Revision 2's change to synth.py.
    df = synth.build("iot_readings", 115848, 1500).df
    assert df["site_city"].head(6).tolist() == [
        "Chicago", "Dallas", "Denver", "New York", "Atlanta", "Dallas"]
    assert int(pd.util.hash_pandas_object(df, index=False).sum()) % (2**61 - 1) == (
        1099975327838780460)
