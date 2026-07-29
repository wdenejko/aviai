"""Hard-case classifier tests — the ladder assigns the right headline bucket, and
the consensus reference rides along. Reuses the Phase 2 mutant injectors as a source
of inputs whose correct label we already know (that's the whole point of mutants)."""

from avtext.consensus.mutants import dewpoint_exceeds_temp, garble
from avtext.harness.hardcases import CLEAN, INVARIANT, PARSE_FAIL, classify

_CLEAN = "EPGD 300000Z 26008KT 9999 BKN009 16/15 Q1005"


def test_clean_report_is_clean():
    v = classify(_CLEAN)
    assert v.label == CLEAN
    assert v.n_ok == 3 and not v.failed
    assert v.reference["temperature_c"] == 16  # consensus rides along as the scoring target


def test_impossible_report_is_invariant():
    v = classify(dewpoint_exceeds_temp(_CLEAN))  # 16/15 -> 16/19, dewpoint > temp
    assert v.label == INVARIANT
    assert any("dewpoint" in m for m in v.invariant_hits)


def test_garbled_report_is_parse_fail():
    v = classify(garble(_CLEAN))  # nonsense group -> at least one parser rejects it
    assert v.label == PARSE_FAIL
    assert v.failed
