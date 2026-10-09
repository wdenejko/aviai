"""Compare two pi runs of one suite, the base against the adapter (ADR-004 Revision 2, item 10).

The held-out probe (sftgen/probe) and dsbench run through the owner's pi harness, k runs a problem,
once with the server's adapter scale at 0 and once at 1 (patches/rev2_probe_mac.sh). Each run
writes reports/agentic-runs/<stem>.json. This reads two of them and answers, problem by problem:
how many of its k runs pass in each state, and how they end (pi's statuses: ok, wrong, timeout,
model_error, ...).

Two tests, because the unit is a problem with k runs, not an item:
- **Per problem, Fisher's exact test** on its 2 x 2 table (passing and failing runs, per state).
  Runs at temperature 0 still vary under the server's batching, so k runs are k draws. With k = 5,
  0 of 5 against 5 of 5 gives p = 0.008, and 1 of 5 against 4 of 5 gives p = 0.21: only a problem
  that flips almost wholly shows.
- **Over problems, the battery's pairing** (battery/dsbench_suite.py): a problem passes when most of
  its runs pass, and the exact McNemar test runs on the problems that pass in one state only.

The runs must be comparable: the same harness, thinking level, k and reply cap (pi_runner records
the cap; the probe runner does too since 2026-10-06), and the same problems. Otherwise this
refuses.

    uv run python -m dsbench.agentic.compare_runs --base reports/agentic-runs/<base>.json \\
        --adapter reports/agentic-runs/<adapter>.json --out <out>.json

Either side can be several runs of one state (`merge`): Revision 2 against Revision 2.1 in one
window runs each adapter in blocks of k = 5, interleaved so that a drift during the window falls
on both (patches/rev2_h2h_mac.sh). Their runs then count together, k = 15 a problem.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from dsbench.battery.stats import mcnemar_exact

# What must match between the two runs for the comparison to mean anything.
SAME = ("harness", "thinking", "repeat", "reply_max_tokens", "provider", "model")


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact test of the table [[a, b], [c, d]]: the chance, with the margins
    fixed, of a table as unlikely as this one or less."""
    n, row1, col1 = a + b + c + d, a + b, a + c

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(n - row1, col1 - x) / math.comb(n, col1)

    seen = prob(a)
    lo, hi = max(0, col1 - (n - row1)), min(row1, col1)
    return min(1.0, sum(p for x in range(lo, hi + 1) if (p := prob(x)) <= seen * (1 + 1e-9)))


def _runs(payload: dict) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    for r in payload["results"]:
        runs[r["id"]].append(r)
    return runs


def merge(payloads: list[dict]) -> dict:
    """Several runs of one state as one: their results together, k the sum of theirs. They must
    share the settings that `compare` checks, all but k, and the problems."""
    first = payloads[0]
    for other in payloads[1:]:
        differ = {k: (first["meta"].get(k), other["meta"].get(k)) for k in SAME
                  if k != "repeat" and first["meta"].get(k) != other["meta"].get(k)}
        if differ:
            raise ValueError(f"the runs to merge differ in {differ}")
        if set(_runs(first)) != set(_runs(other)):
            raise ValueError("the runs to merge have different problems: "
                             f"{sorted(set(_runs(first)) ^ set(_runs(other)))}")
    meta = {**first["meta"],
            "repeat": sum(p["meta"].get("repeat") or 0 for p in payloads),
            "label": "+".join(str(p["meta"].get("label")) for p in payloads),
            "timestamp": "+".join(str(p["meta"].get("timestamp")) for p in payloads)}
    return {"meta": meta, "results": [r for p in payloads for r in p["results"]]}


def compare(base: dict, adapter: dict) -> dict:
    """Base against adapter, problem by problem, then over problems."""
    differ = {k: (base["meta"].get(k), adapter["meta"].get(k)) for k in SAME
              if base["meta"].get(k) != adapter["meta"].get(k)}
    if differ:
        raise ValueError(f"the runs differ in {differ}: not comparable")
    b_runs, a_runs = _runs(base), _runs(adapter)
    if set(b_runs) != set(a_runs):
        raise ValueError(f"the runs have different problems: {sorted(set(b_runs) ^ set(a_runs))}")
    problems = {}
    for pid in sorted(b_runs):
        bp = sum(bool(r["passed"]) for r in b_runs[pid])
        ap = sum(bool(r["passed"]) for r in a_runs[pid])
        bn, an = len(b_runs[pid]), len(a_runs[pid])
        problems[pid] = {
            "category": b_runs[pid][0]["category"], "runs": [bn, an], "base": bp, "adapter": ap,
            "base_majority": bp * 2 > bn, "adapter_majority": ap * 2 > an,
            "p_fisher": float(f"{fisher_exact(bp, bn - bp, ap, an - ap):.3g}"),
            "statuses": {"base": dict(Counter(r["status"] for r in b_runs[pid])),
                         "adapter": dict(Counter(r["status"] for r in a_runs[pid]))},
            "latency_s_median": {
                "base": sorted(r["latency_s"] for r in b_runs[pid])[bn // 2],
                "adapter": sorted(r["latency_s"] for r in a_runs[pid])[an // 2]},
        }
    lost = sum(p["base_majority"] and not p["adapter_majority"] for p in problems.values())
    gained = sum(p["adapter_majority"] and not p["base_majority"] for p in problems.values())
    return {
        "base": {k: base["meta"].get(k) for k in ("label", "timestamp")},
        "adapter": {k: adapter["meta"].get(k) for k in ("label", "timestamp")},
        "settings": {k: base["meta"].get(k) for k in SAME},
        "problems": problems,
        "runs_passed": {"base": sum(p["base"] for p in problems.values()),
                        "adapter": sum(p["adapter"] for p in problems.values()),
                        "of": sum(p["runs"][0] for p in problems.values())},
        "problems_passed": {"base": sum(p["base_majority"] for p in problems.values()),
                            "adapter": sum(p["adapter_majority"] for p in problems.values()),
                            "of": len(problems)},
        "majority_flips": {"lost": lost, "gained": gained,
                           "p_mcnemar": float(f"{mcnemar_exact(lost, gained):.3g}")},
        "statuses": {"base": dict(Counter(r["status"] for r in base["results"])),
                     "adapter": dict(Counter(r["status"] for r in adapter["results"]))},
    }


def _statuses(counts: dict[str, int]) -> str:
    return ", ".join(f"{s}×{n}" for s, n in sorted(counts.items()))


def markdown(result: dict) -> str:
    """The comparison as the report's table."""
    lines = ["| problem | category | base | adapter | Fisher p | base statuses "
             "| adapter statuses |", "|---|---|---:|---:|---:|---|---|"]
    for pid, p in result["problems"].items():
        lines.append(f"| {pid} | {p['category']} | {p['base']}/{p['runs'][0]} | "
                     f"{p['adapter']}/{p['runs'][1]} | {p['p_fisher']:.3g} | "
                     f"{_statuses(p['statuses']['base'])} | "
                     f"{_statuses(p['statuses']['adapter'])} |")
    runs, probs, flips = result["runs_passed"], result["problems_passed"], result["majority_flips"]
    lines += ["", f"Runs passed: base {runs['base']}/{runs['of']}, adapter {runs['adapter']}/"
              f"{runs['of']}. Problems passed by majority: base {probs['base']}/{probs['of']}, "
              f"adapter {probs['adapter']}/{probs['of']} ({flips['gained']} gained, "
              f"{flips['lost']} lost, McNemar p = {flips['p_mcnemar']:.3g})."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, nargs="+", required=True,
                    help="one run, or several runs of the same state to count together")
    ap.add_argument("--adapter", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = compare(merge([json.loads(p.read_text()) for p in args.base]),
                     merge([json.loads(p.read_text()) for p in args.adapter]))
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(markdown(result))


if __name__ == "__main__":
    main()
