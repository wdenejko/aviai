"""Compare two runs (before vs after, or step-N vs step-M) as a PAIRED test.

Why paired and not two independent percentages: the benchmark is small, so per-run noise is a
few points (a 20-problem set has ~1 sigma ~= 2-3 problems). The high-power question is not "did
the average move" but "which specific problems flipped". We report the per-cell delta, the list
of regressions (passed before, fails now) and gains, and an exact McNemar p-value over the
flips so you can tell signal from a coin toss before you trust a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

from dsbench.schema import CATEGORIES, CATEGORY_NAMES, DIFFICULTIES


def _load(path: str) -> tuple[dict, dict[str, dict]]:
    obj = json.loads(Path(path).read_text())
    by_id = {r["id"]: r for r in obj["results"]}
    return obj["meta"], by_id


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p over discordant pairs (b: pass->fail, c: fail->pass)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired before/after comparison of two dsbench runs.")
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out", default=None, help="write the markdown report here too")
    args = ap.parse_args()

    mb, before = _load(args.before)
    ma, after = _load(args.after)
    ids = sorted(set(before) & set(after))
    only_b = set(before) - set(after)
    only_a = set(after) - set(before)

    b = c = both = neither = 0
    regressions, gains = [], []
    cell = {(cat, d): [0, 0, 0] for cat in CATEGORIES for d in DIFFICULTIES}  # [before, after, n]
    for pid in ids:
        pb, pa = before[pid]["passed"], after[pid]["passed"]
        key = (before[pid]["category"], before[pid]["difficulty"])
        cell[key][0] += int(pb)
        cell[key][1] += int(pa)
        cell[key][2] += 1
        if pb and not pa:
            b += 1
            regressions.append(pid)
        elif not pb and pa:
            c += 1
            gains.append(pid)
        elif pb and pa:
            both += 1
        else:
            neither += 1

    pb_total = sum(before[i]["passed"] for i in ids)
    pa_total = sum(after[i]["passed"] for i in ids)
    p = mcnemar_exact(b, c)
    n = len(ids)
    delta = pa_total - pb_total
    sig = "significant" if p < 0.05 else "not significant"
    only = f"  (before-only: {len(only_b)}, after-only: {len(only_a)})" if only_b or only_a else ""

    lines = [
        f"# dsbench compare: {mb.get('label')} -> {ma.get('label')}",
        "",
        f"- paired problems: {n}{only}",
        f"- overall: {pb_total}/{n} -> {pa_total}/{n}  (delta {delta:+d})",
        f"- gains (fail->pass): {c}   regressions (pass->fail): {b}   unchanged: {both + neither}",
        f"- McNemar exact p = {p:.4f}  ({sig} at 0.05)",
        "",
        # Header follows DIFFICULTIES so adding a tier (e.g. `expert`) needs no edit here.
        "| category | " + " | ".join(DIFFICULTIES) + " |",
        "|" + "|".join(["---"] * (len(DIFFICULTIES) + 1)) + "|",
    ]
    for cat in CATEGORIES:
        row = [CATEGORY_NAMES[cat]]
        for d in DIFFICULTIES:
            bef, aft, n = cell[(cat, d)]
            row.append(f"{bef}->{aft} /{n}" if n else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if regressions:
        lines += ["## regressions (were passing, now fail) - investigate these", ""]
        lines += [f"- {pid}" for pid in sorted(regressions)] + [""]
    if gains:
        lines += ["## gains (were failing, now pass)", ""]
        lines += [f"- {pid}" for pid in sorted(gains)] + [""]

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
