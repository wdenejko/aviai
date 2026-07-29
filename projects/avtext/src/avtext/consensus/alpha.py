"""Krippendorff's alpha — the panel-agreement KPI (Phase 2, Skill B).

Hand-rolled on purpose (the plan: write the stats yourself). Nominal alpha measures
agreement BEYOND CHANCE across raters (oracles) on units (per-report fields), and —
crucially for us — tolerates missing ratings: a parser that failed, or left a field
None, simply doesn't rate that unit.

    alpha = 1 - Do/De     (observed disagreement / chance-expected disagreement)
    alpha = 1  perfect · 0  chance-level · <0  systematic disagreement

Nominal metric (two ratings match or they don't), computed via the coincidence
matrix (Krippendorff 2004). Units with fewer than two ratings carry no agreement
information and are dropped. Why alpha and not plain % agreement: % agreement is
inflated when one value dominates (three parsers all say "no gust" agree trivially);
alpha discounts that chance agreement.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from itertools import permutations


def krippendorff_alpha(units: list[list[Hashable | None]]) -> float | None:
    """Nominal alpha over `units` (one list of ratings per unit; None = missing).

    Returns None when there isn't enough data to measure agreement.
    """
    # Coincidence matrix: each unit with m>=2 ratings contributes every ordered
    # pair, weighted 1/(m-1) so units count equally regardless of how many rated them.
    coincidence: Counter[tuple[Hashable, Hashable]] = Counter()
    for ratings in units:
        vals = [r for r in ratings if r is not None]
        m = len(vals)
        if m < 2:
            continue
        for a, b in permutations(vals, 2):
            coincidence[(a, b)] += 1.0 / (m - 1)
    if not coincidence:
        return None

    marginal: Counter[Hashable] = Counter()
    for (a, _b), c in coincidence.items():
        marginal[a] += c
    n = sum(marginal.values())
    if n < 2:
        return None

    off_diagonal = sum(c for (a, b), c in coincidence.items() if a != b)
    sum_sq = sum(c * c for c in marginal.values())
    denom = n * n - sum_sq
    if denom == 0:
        return 1.0  # every rating identical — perfect agreement
    # alpha = 1 - Do/De, closed form: Do = off_diag/n, De = denom/(n(n-1)).
    return 1.0 - off_diagonal * (n - 1) / denom
