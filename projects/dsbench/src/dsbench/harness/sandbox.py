"""Run a candidate solution in an isolated subprocess and grade it with the problem's checker.

The split is deliberate: the model's code runs in _worker (throwaway process, killed on timeout);
the checker runs here in the trusted parent. The only thing that crosses the boundary is the
pickled result value.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

from dsbench.loader import module_path
from dsbench.schema import Problem


def run_candidate(problem: Problem, code: str) -> tuple[bool, str, str]:
    """Execute `code` against `problem` and check the result.

    Returns (passed, status, reason). status is one of:
      ok      - ran and the checker accepted the result
      wrong   - ran but the result is incorrect
      error   - the code raised (reason carries the last traceback line)
      timeout - exceeded problem.timeout
    """
    if not code.strip():
        return False, "no_code", "no code block found in the response"

    with tempfile.TemporaryDirectory() as d:
        code_file = Path(d) / "candidate.txt"
        result_file = Path(d) / "result.pkl"
        code_file.write_text(code)

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "dsbench.harness._worker",
                 module_path(problem.id), str(code_file), str(result_file)],
                capture_output=True,
                text=True,
                timeout=problem.timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout", f"exceeded {problem.timeout:.0f}s"

        if not result_file.exists():
            tail = (proc.stderr or "process died with no result").strip().splitlines()
            return False, "error", tail[-1] if tail else "unknown worker failure"

        payload = pickle.loads(result_file.read_bytes())

    if not payload["ok"]:
        tail = payload["error"].strip().splitlines()
        return False, "error", tail[-1] if tail else "error"

    # Trusted checker, parent process. Guard against a buggy checker so it fails loud, not silent.
    try:
        verdict = problem.check(payload["result"])
    except Exception as e:  # noqa: BLE001
        return False, "error", f"checker raised: {type(e).__name__}: {e}"

    if isinstance(verdict, tuple):
        passed, reason = verdict
    else:
        passed, reason = bool(verdict), ""
    return (True, "ok", "") if passed else (False, "wrong", reason or "incorrect result")
