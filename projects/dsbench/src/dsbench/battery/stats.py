"""Paired statistics for base vs adapter, and the ADR-001 Gate-2 acceptance verdicts.

Each item is scored in both states, so the comparison is paired. Items both states get right, or
both get wrong, say nothing about the difference. The information is in the discordant pairs:
    b = base right, adapter wrong   (a loss)
    c = base wrong, adapter right   (a gain)
Under "no difference", each discordant item is a fair coin, so the exact McNemar test is a binomial
test of c against b + c. The delta is (c - b) / n. Its 95% interval is the matching EXACT one: a
Clopper-Pearson interval for the gain share q = c / (b + c), mapped through
delta = (b + c) / n * (2q - 1). (The first draft used the Wald interval. With 16 discordant
HumanEval+ items it gave (-9.7, -0.2), excluding zero, while the exact test gave p = 0.077: the two
disagreed. The exact interval and the exact test now always agree about zero.)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

# ADR-001 Gate 2, item 3. Target benchmarks must GAIN (at least two of them); general benchmarks
# must not regress by more than the threshold (percentage points). BFCL is scored the way its own
# leaderboard reports it: "bfcl_ast" (the call categories: choosing and filling the right tool, the
# ADR's tool-call target) apart from "bfcl_irrelevance" (declining when no tool fits), which the ADR
# sets no threshold for but which the report must not average away.
TARGET_BENCHES = ("ds1000", "bird", "dsbench", "bfcl_ast")
REGRESSION_THRESHOLDS = {"ifeval": 1.0, "mmlu_pro": 1.0, "gpqa": 1.0, "lcb": 2.0,
                         "humaneval_plus": 2.0}
ALPHA = 0.05


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Binomial(n, p), summed in log space (no overflow at the n used here)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp
             + (n - i) * lq for i in range(k + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def clopper_pearson(c: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact interval for a binomial proportion c / n, found by bisection on the CDF."""
    def solve(f) -> float:  # f is decreasing in p; find f(p) = alpha / 2
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    low = 0.0 if c == 0 else 1.0 - solve(lambda p: 1.0 - _binom_cdf(c - 1, n, 1.0 - p))
    high = 1.0 if c == n else solve(lambda p: _binom_cdf(c, n, p))
    return low, high


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: 2 * P(X <= min(b, c)), X ~ Binomial(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass
class Paired:
    bench: str
    n: int
    acc_base: float  # percent
    acc_adapter: float  # percent
    delta: float  # percentage points, adapter - base
    ci_low: float
    ci_high: float
    losses: int  # b: base right, adapter wrong
    gains: int  # c: base wrong, adapter right
    p: float

    def as_dict(self) -> dict:
        return asdict(self)


def paired(bench: str, base: dict[str, bool], adapter: dict[str, bool]) -> Paired:
    """Compare two per-item outcome maps on the items both contain."""
    ids = sorted(set(base) & set(adapter))
    n = len(ids)
    if n == 0:
        raise ValueError(f"{bench}: no items scored in both states")
    b = sum(1 for i in ids if base[i] and not adapter[i])
    c = sum(1 for i in ids if adapter[i] and not base[i])
    d = (c - b) / n
    if b + c:
        q_low, q_high = clopper_pearson(c, b + c)
        lo, hi = (b + c) / n * (2 * q_low - 1), (b + c) / n * (2 * q_high - 1)
    else:
        lo = hi = 0.0
    return Paired(
        bench=bench,
        n=n,
        acc_base=100 * sum(base[i] for i in ids) / n,
        acc_adapter=100 * sum(adapter[i] for i in ids) / n,
        delta=100 * d,
        ci_low=100 * lo,
        ci_high=100 * hi,
        losses=b,
        gains=c,
        p=mcnemar_exact(b, c),
    )


def verdict(p: Paired) -> str:
    """The ADR reading of one benchmark's paired result.

    Target benchmark: 'gain' only if the improvement is significant (p < ALPHA); otherwise 'no
    gain'. A point estimate inside the noise does not count toward "at least two".
    General benchmark: 'fail' if the point estimate regresses past the threshold, as the ADR
    words it; 'fail (n.s.)' marks a failing point estimate whose exact test is not significant
    (still a fail: the ADR bounds the size of the regression, it doesn't ask for significance).
    A pass is 'confirmed' only if the whole interval clears the threshold, else 'unresolved': the
    point estimate passes, but the sample can't rule out a regression of that size.
    """
    if p.bench in TARGET_BENCHES:
        return "gain" if p.delta > 0 and p.p < ALPHA else "no gain"
    threshold = REGRESSION_THRESHOLDS.get(p.bench)
    if threshold is None:  # measured, but not an ADR criterion: say so when the change is real
        if p.p < ALPHA:
            return "regression (no ADR limit)" if p.delta < 0 else "improvement (no ADR limit)"
        return "info"
    if p.delta < -threshold:
        return "fail" if p.p < ALPHA else "fail (n.s.)"
    return "pass (confirmed)" if p.ci_low >= -threshold else "pass (unresolved)"


def aa_flip_rate(first: dict[str, bool], second: dict[str, bool]) -> tuple[int, int]:
    """(flipped, n) between two passes of the SAME state: the harness's own noise floor."""
    ids = set(first) & set(second)
    return sum(1 for i in ids if first[i] != second[i]), len(ids)


def gate2_decision(results: dict[str, Paired]) -> dict:
    """Combine per-benchmark verdicts into the Gate-2 call, listing what was not measured."""
    verdicts = {bench: verdict(p) for bench, p in results.items()}
    gains = [b for b in TARGET_BENCHES if verdicts.get(b) == "gain"]
    fails = [b for b in REGRESSION_THRESHOLDS if verdicts.get(b, "").startswith("fail")]
    missing = [b for b in (*TARGET_BENCHES, *REGRESSION_THRESHOLDS) if b not in results]
    return {
        "verdicts": verdicts,
        "target_gains": gains,
        "regressions": fails,
        "not_measured": missing,
        "target_criterion_met": len(gains) >= 2,
        "regression_criterion_met": not fails,
    }
