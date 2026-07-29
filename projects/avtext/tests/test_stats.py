"""Stats tests — bootstrap CI brackets the point estimate and is reproducible; McNemar
flags a real paired difference and stays silent when there's nothing to tell apart."""

from avtext.harness.score import Outcome, RecordScore
from avtext.harness.stats import bootstrap_ci, mcnemar


def _rec(em: bool) -> RecordScore:
    # minimal record whose only signal is exact_match, for testing the CI machinery
    o = Outcome.HIT if em else Outcome.WRONG
    return RecordScore(id="", split="", label="", outcomes={"temperature_c": o}, exact_match=em)


def _em_rate(scores):
    return sum(s.exact_match for s in scores) / len(scores)


def test_bootstrap_brackets_point_and_is_deterministic():
    scores = [_rec(True)] * 8 + [_rec(False)] * 2  # EM = 0.8
    point, lo, hi = bootstrap_ci(scores, _em_rate, n_boot=500, seed=0)
    assert point == 0.8
    assert lo <= 0.8 <= hi
    assert bootstrap_ci(scores, _em_rate, n_boot=500, seed=0) == (point, lo, hi)  # seeded → stable


def test_mcnemar_flags_a_real_difference():
    # B fixes 10 of A's failures and breaks none → strongly favours B
    a = [False] * 10 + [True] * 2
    b = [True] * 12
    r = mcnemar(a, b)
    assert (r.b, r.c) == (0, 10) and r.p_value < 0.05


def test_mcnemar_silent_when_no_discordant_pairs():
    a = [True, True, False, False]
    r = mcnemar(a, a)  # identical → no discordant pairs
    assert (r.b, r.c) == (0, 0) and r.p_value == 1.0
