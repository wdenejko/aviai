"""Trust the checkers before trusting a score: run every problem's REFERENCE solution through the
real sandbox + checker and assert they all pass.

This is the oracle step. If a reference fails, the bug is in the problem (bad fixtures, wrong
checker, wrong gold), not in the model -- fix it before running any model. Run this in CI and
after every edit to a problem.

  uv run --package dsbench dsbench-selftest
"""

from __future__ import annotations

import argparse

from dsbench.harness.sandbox import run_candidate
from dsbench.loader import load_problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate every problem's reference solution.")
    ap.add_argument("--category", choices=["de", "da", "ds"], default=None)
    args = ap.parse_args()

    problems = load_problems(args.category)
    failures = []
    for p in problems:
        passed, status, reason = run_candidate(p, p.reference)
        mark = "ok  " if passed else "FAIL"
        print(f"  {mark} {p.id:16} [{p.difficulty}] {p.title}")
        if not passed:
            failures.append((p.id, status, reason))

    print(f"\n{len(problems) - len(failures)}/{len(problems)} references pass")
    if failures:
        print("\nBROKEN PROBLEMS (fix the fixtures/checker/reference):")
        for pid, status, reason in failures:
            print(f"  {pid}: {status} - {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
