"""Execute scoring jobs INSIDE the battery sandbox container. Stdlib only; never run on the host.

The host (`score.py`) starts one rootless podman container per scoring batch:
    podman run --rm --network none --read-only --cap-drop all --memory 32g --pids-limit 2048 ...
        -v HARNESS:/harness:ro -v DATA:/data:ro -v JOBS:/work:ro  IMAGE
        python /harness/dsbench/battery/sandbox_exec.py /work/jobs.jsonl --workers 12
Every job runs as its own subprocess with a wall-clock timeout, in a fresh temp directory on the
container's tmpfs. Results go to STDOUT as JSONL, never to a mounted directory: the container has
no writable host path at all, so model code cannot touch the host filesystem.

Job kinds:
  program  run a whole Python program (HumanEval+, DS-1000); pass = exit status 0.
  lcb      LiveCodeBench's own `run_test` (fetched to /data/refs) on the problem's tests;
           pass = every test True, as in lcb_runner's pass@1.
  sqlite   BIRD: run predicted and gold SQL on a read-only copy of the database;
           pass = set(pred rows) == set(gold rows), BIRD's official execution accuracy.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

REFS = "/data/refs"
CHILD_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "MPLBACKEND": "Agg",
    "MPLCONFIGDIR": "/tmp/mpl",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "CUDA_VISIBLE_DEVICES": "-1",
    # One BLAS thread per job: N parallel jobs x N threads would oversubscribe the cores, and
    # LiveCodeBench's per-test time limits would then fail correct-but-starved solutions.
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}

_LCB_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
sys.set_int_max_str_digits(50000)  # as compute_code_generation_metrics.py
from lcb_testing_util import run_test
spec = open(sys.argv[2]).read()
code = open(sys.argv[3]).read()
results, meta = run_test({"input_output": spec}, test=code, debug=False, timeout=int(sys.argv[4]))
print("@@RESULT@@" + json.dumps({"results": [r if isinstance(r, (bool, int)) else str(r)
                                           for r in results], "meta": str(meta)[:300]}))
"""

_SQLITE_CHILD = r"""
import json, sqlite3, sys
db, pred, gold = sys.argv[1], open(sys.argv[2]).read(), open(sys.argv[3]).read()
def run(sql):
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()
gold_rows = run(gold)
try:
    pred_rows = run(pred)
except Exception as e:
    print("@@RESULT@@" + json.dumps({"passed": False, "status": "error",
                                     "detail": f"{type(e).__name__}: {e}"[:300]}))
    sys.exit(0)
ok = set(pred_rows) == set(gold_rows)
print("@@RESULT@@" + json.dumps({"passed": ok, "status": "ok" if ok else "wrong",
                                 "detail": f"pred {len(pred_rows)} rows, gold {len(gold_rows)}"}))
"""


def _tail(text: str, n: int = 300) -> str:
    lines = (text or "").strip().splitlines()
    return lines[-1][:n] if lines else ""


def _child(argv: list[str], cwd: str, timeout: float) -> tuple[str, subprocess.CompletedProcess]:
    # The container's own environment (image ENV: dataset caches) plus the overrides above. No host
    # variable reaches it: podman does not forward the caller's environment.
    env = {**os.environ, **CHILD_ENV}
    try:
        proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", None
    return "done", proc


def _payload(proc: subprocess.CompletedProcess) -> dict | None:
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return None


def run_job(job: dict) -> dict:
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        kind = job["kind"]
        if kind == "program":
            path = os.path.join(tmp, "prog.py")
            with open(path, "w") as fh:
                fh.write(job["program"])
            state, proc = _child([sys.executable, path], tmp, job.get("timeout", 60))
            if state == "timeout":
                res = {"passed": False, "status": "timeout", "detail": ""}
            elif proc.returncode == 0:
                res = {"passed": True, "status": "ok", "detail": ""}
            else:
                res = {"passed": False, "status": "error", "detail": _tail(proc.stderr)}
        elif kind == "lcb":
            code = os.path.join(tmp, "sol.py")
            with open(code, "w") as fh:
                fh.write(job["code"])
            per_test = job.get("timeout", 6)
            budget = (per_test + 1) * job["n_tests"] + 5  # lcb_runner's process-level cap
            state, proc = _child(
                [sys.executable, "-c", _LCB_CHILD, REFS, f"/data/{job['tests']}", code,
                 str(per_test)], tmp, budget)
            payload = _payload(proc) if state == "done" else None
            if payload is None:
                detail = "" if state == "timeout" else _tail(proc.stderr)
                res = {"passed": False, "status": "timeout" if state == "timeout" else "error",
                       "detail": detail}
            else:
                results = payload["results"]
                # lcb_runner: pass iff every result > 0 (True); failures are negative codes and
                # run_test stops at the first one.
                ok = bool(results) and all(r is True for r in results)
                res = {"passed": ok, "status": "ok" if ok else "wrong",
                       "detail": f"{sum(r is True for r in results)}/{job['n_tests']} tests"
                                 + ("" if ok else f"; {payload['meta']}")}
        elif kind == "sqlite":
            pred, gold = os.path.join(tmp, "pred.sql"), os.path.join(tmp, "gold.sql")
            with open(pred, "w") as fh:
                fh.write(job["pred"])
            with open(gold, "w") as fh:
                fh.write(job["gold"])
            state, proc = _child([sys.executable, "-c", _SQLITE_CHILD, f"/data/{job['db']}",
                                  pred, gold], tmp, job.get("timeout", 30))
            payload = _payload(proc) if state == "done" else None
            if payload is None:
                res = {"passed": False, "status": "timeout" if state == "timeout" else "error",
                       "detail": "" if state == "timeout" else _tail(proc.stderr)}
            else:
                res = payload
        else:
            res = {"passed": False, "status": "error", "detail": f"unknown job kind {kind!r}"}
    return {"key": job["key"], **res, "elapsed_s": round(time.monotonic() - t0, 2)}


def main() -> None:
    jobs_path = sys.argv[1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    with open(jobs_path) as fh:
        jobs = [json.loads(line) for line in fh if line.strip()]
    os.makedirs("/tmp/mpl", exist_ok=True)
    print(f"[sandbox] {len(jobs)} jobs, {workers} workers", file=sys.stderr, flush=True)
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for res in pool.map(run_job, jobs):
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
            done += 1
            if done % 100 == 0:
                print(f"[sandbox] {done}/{len(jobs)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
