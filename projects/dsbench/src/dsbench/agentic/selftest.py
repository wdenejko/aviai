"""dsbench-agent-selftest: the oracle gate for the agentic set.

Runs each problem's reference solution (setup -> reference -> check) with NO model in the loop, and
asserts it reaches a passing state. If a reference fails, the bug is in the problem (setup,
fixtures, or checker), not the model. Must be green before any agent score is trusted. A smoke test
that ClickHouse is up and the checkers run.

It also checks each problem's agent login (`access.check_access`): the agent must see none of the
labels its grader holds back, and still be able to work in its scratch database. A problem whose
login lets a label through fails the gate as surely as a wrong reference.
"""
from __future__ import annotations

from dsbench.agentic.access import check_access
from dsbench.agentic.loader import load_problems
from dsbench.agentic.loop import grade, prepare_context


def main() -> None:
    problems = load_problems()
    if not problems:
        raise SystemExit("no agentic problems found")
    ok = 0
    for p in problems:
        try:
            ctx = prepare_context(p.id)
            if p.setup:
                p.setup(ctx)
            ctx.answer = p.reference(ctx)
        except Exception as e:  # noqa: BLE001
            print(f"  REF-ERROR  {p.id:16} {type(e).__name__}: {str(e)[:160]}")
            continue
        passed, _status, reason = grade(p, ctx)
        try:
            leaks = check_access(p, ctx)
        except Exception as e:  # noqa: BLE001
            leaks = [f"access check raised {type(e).__name__}: {str(e)[:160]}"]
        ok += passed and not leaks
        mark = "ok  " if passed and not leaks else "FAIL"
        tail = "" if passed else f"  -> {reason}"
        tail += "".join(f"  -> agent login: {leak}" for leak in leaks)
        print(f"  {mark} {p.id:16} [{p.difficulty}] {p.title}{tail}")
    print(f"\n{ok}/{len(problems)} references pass with nothing withheld in the agent's reach")
    if ok != len(problems):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
