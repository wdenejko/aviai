"""Stats — uncertainty on the scores (Phase 3).

Two tools, matched to the two questions a run report has to answer honestly.

`bootstrap_ci` — "how precise is this number?" A single accuracy off 620 records is a
point estimate; report it without a band and a 0.91 looks meaningfully better than a 0.89
when the data can't tell them apart. We resample the RECORDS (the independent unit — not
the fields, which are correlated within a report) with replacement, recompute the metric on
each resample, and read the interval off the percentiles of that distribution. Seeded, so
the interval is reproducible.

`mcnemar` — "did B actually beat A, or is it noise?" For Phase 4's paired baseline-vs-
finetuned comparison on the SAME items. Only the disagreements carry information: b (A right,
B wrong) vs c (A wrong, B right). Shared successes and shared failures cancel. Included now so
the stats layer is complete and unit-tested before a second model exists.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from avtext.harness.score import RecordScore

Metric = Callable[[list[RecordScore]], float | None]


def bootstrap_ci(
    scores: Sequence[RecordScore],
    metric: Metric,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    """Return (point, lo, hi) for `metric` at the (1-alpha) level. (None, None, None) if the
    metric is undefined on the data (e.g. hallucination_rate with no absent fields)."""
    pool = list(scores)
    point = metric(pool)
    if point is None or not pool:
        return (point, None, None)
    rng = random.Random(seed)  # local RNG -> deterministic, no global-state surprises
    n = len(pool)
    boots = sorted(
        v for _ in range(n_boot) if (v := metric(rng.choices(pool, k=n))) is not None
    )
    if not boots:
        return (point, None, None)
    lo = boots[round((alpha / 2) * (len(boots) - 1))]
    hi = boots[round((1 - alpha / 2) * (len(boots) - 1))]
    return (point, lo, hi)


@dataclass
class McNemar:
    b: int  # A correct, B wrong
    c: int  # A wrong, B correct
    statistic: float
    p_value: float


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> McNemar:
    """Paired test on two models' per-item correctness (same items, same order).

    Uses the continuity-corrected statistic (|b-c|-1)^2 / (b+c), which is chi-square with
    1 df under the null; since chi-sq(1) = Z^2, the two-sided p-value is erfc(sqrt(stat/2)).
    No SciPy needed. p=1.0 when there are no discordant pairs (nothing to tell apart).
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired inputs must be the same length")
    b = sum(1 for a, bb in zip(a_correct, b_correct, strict=True) if a and not bb)
    c = sum(1 for a, bb in zip(a_correct, b_correct, strict=True) if not a and bb)
    if b + c == 0:
        return McNemar(b, c, 0.0, 1.0)
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(stat / 2))
    return McNemar(b, c, stat, p)


def fmt_ci(point: float | None, lo: float | None, hi: float | None, *, pct: bool = True) -> str:
    """Compact 'value [lo, hi]' for report tables; '—' when undefined."""
    if point is None:
        return "—"
    s = 100 if pct else 1
    u = "%" if pct else ""
    if lo is None:
        return f"{point * s:.1f}{u}"
    return f"{point * s:.1f}{u} [{lo * s:.1f}, {hi * s:.1f}]"
