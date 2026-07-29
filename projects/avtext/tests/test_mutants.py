"""Mutant-audit tests — the pipeline catches what it should, and (the point)
does NOT catch a plausible-but-wrong value. That last one is a feature of the
demonstration, not a bug: it's why the trust ladder needs an authority."""

from avtext.consensus.mutants import (
    detect,
    dewpoint_exceeds_temp,
    garble,
    gust_below_wind,
    perturb_temp_plausible,
)

CLEAN = "EPGD 300000Z 26008KT 9999 BKN009 16/15 Q1005"


def test_clean_report_is_accepted():
    assert detect(CLEAN) is None


def test_physical_impossibility_caught_by_invariant():
    assert detect(dewpoint_exceeds_temp(CLEAN)) == "invariant"  # 16/15 -> 16/19
    assert detect(gust_below_wind(CLEAN)) == "invariant"  # 26008KT -> 26008G05KT


def test_garble_caught_by_parse():
    assert detect(garble(CLEAN)) == "parse"


def test_plausible_wrong_value_slips_through():
    mutated = perturb_temp_plausible(CLEAN)  # 16 -> 20; still physically fine
    assert mutated is not None and mutated != CLEAN
    assert detect(mutated) is None  # the blind spot only the gold seed can cover
