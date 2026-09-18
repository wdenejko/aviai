"""dsbench v2 agentic baseline via the `pi` coding agent (public harness), thinking ON.

WHY a public harness (ADR-003, revised): an *accurate* agentic baseline should be measured through a
real, reproducible agent scaffold, not our bespoke loop. We drive `pi` (badlogic/pi-mono) headless:
it operates the sandbox through its `bash` tool, calling the run_sql / run_python wrappers on PATH
(sandbox/pi/bin) — the identical tool contract our native loop gave, but now the *harness* is the
public, pinned one. Grading is unchanged: our independent oracle reads the ClickHouse end-state.

pi uses NATIVE tool-calls, so the fork's tool-call-arguments quirk *can* surface. We therefore
split the outcome: `wrong` = pi finished but the end-state is wrong (a CAPABILITY miss);
`model_error` = pi could not complete because the serving/tool-call layer failed (the
TOOL-RELIABILITY axis — the fine-tune target). Keeping the two apart is the point of measuring both.

Run:  uv run --package dsbench dsbench-pi-run --ids de_hub_daily --verbose
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from dsbench.agentic.loader import load_problems
from dsbench.agentic.loop import SCHEMA_DOC, grade, prepare_context
from dsbench.agentic.schema import AgentProblem, AgentResult, GradeContext

_PROJECT = Path(__file__).resolve().parents[3]
_RUNS = _PROJECT / "reports" / "agentic-runs"
_BIN = _PROJECT / "sandbox" / "pi" / "bin"  # run_sql / run_python wrappers pi calls via bash

SYSTEM_PI = """You are a senior data engineer/analyst operating a REAL aviation data warehouse \
(ClickHouse). You work through your shell (the `bash` tool). Two commands are on your PATH:

  run_sql "<ONE ClickHouse statement>"
      Runs one SQL statement. Shared, read-only data is in the `aviation` database — query it \
fully-qualified (e.g. aviation.flights). Your scratch DB `{namespace}` is the DEFAULT, so \
`CREATE TABLE t ...` / `INSERT` land there. SELECT/SHOW/DESCRIBE print a table; DDL prints \
little. ClickHouse has no correlated subqueries — use JOINs.

  run_python   (reads Python 3 from STDIN, e.g.  run_python < script.py)
      Runs in a container with pandas, numpy, scipy, scikit-learn, clickhouse_connect. To reach \
ClickHouse use exactly:  client = clickhouse_connect.get_client(host='clickhouse', port=8123, \
username='avbench', password='avbench', database=os.environ['CLICKHOUSE_DB'])  (do NOT use \
host='localhost' inside the container). Read with client.query_df('SELECT ... FROM aviation.x'). \
To WRITE a result table, create it then insert a DataFrame: \
client.command('CREATE TABLE t (col Type, ...) ENGINE = MergeTree ORDER BY col'); \
client.insert_df('t', df)  (df columns must match, in order). It defaults to your scratch DB.

Work step by step and produce EXACTLY the deliverable asked for. If the task asks you to BUILD a \
table, create it in your scratch database with the exact name and columns requested. If the task \
asks a QUESTION, your FINAL message must contain ONLY the answer — the bare value or the exact \
JSON requested, nothing else. There is no "finish" tool: you are done when you stop.

{schema}"""


def _parse_trajectory(events_path: Path) -> dict:
    """Reduce pi's JSONL event stream to what grading + reporting need."""
    events = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    settled = any(e.get("type") == "agent_settled" for e in events)
    # last agent_end holds the full final message list (there can be >1 if pi retried a turn)
    agent_ends = [e for e in events if e.get("type") == "agent_end"]
    messages = agent_ends[-1].get("messages", []) if agent_ends else []
    will_retry = bool(agent_ends and agent_ends[-1].get("willRetry"))
    steps = sum(1 for e in events if e.get("type") == "turn_end")
    tool_calls = sum(1 for e in events if e.get("type") == "tool_execution_start")
    tool_errors = sum(
        1 for e in events if e.get("type") == "tool_execution_end" and e.get("isError")
    )
    # answer = text blocks of the last assistant message (we instruct "final message = answer only")
    answer = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            texts = [b.get("text", "") for b in m.get("content", []) if b.get("type") == "text"]
            joined = "\n".join(t for t in texts if t).strip()
            answer = joined or None
            break
    return {
        "settled": settled, "messages": messages, "will_retry": will_retry,
        "steps": steps, "tool_calls": tool_calls, "tool_errors": tool_errors, "answer": answer,
    }


def run_pi_agent(problem: AgentProblem, ctx: GradeContext, *, provider: str, model: str,
                 thinking: str, timeout: float, verbose: bool = False) -> AgentResult:
    t0 = time.time()
    system = SYSTEM_PI.format(namespace=ctx.namespace, schema=SCHEMA_DOC)
    workdir = Path(tempfile.mkdtemp(prefix=f"pi-{problem.id}-"))
    env = os.environ.copy()
    env["PATH"] = f"{_BIN}:{env['PATH']}"
    env["CLICKHOUSE_DB"] = ctx.namespace  # run_sql/run_python default to this problem's scratch DB
    cmd = [
        "pi", "--print", "--mode", "json", "--no-session", "--no-context-files",
        "--provider", provider, "--model", model, "--thinking", thinking,
        "-t", "bash,read,write",
        "--append-system-prompt", system,
        "--", problem.prompt,
    ]

    def result(passed, status, reason, steps=0, tool_calls=0, traj=None):
        return AgentResult(problem.id, problem.category, problem.difficulty, passed, status,
                           reason=reason, steps=steps, tool_calls=tool_calls, answer=ctx.answer,
                           trajectory=traj or [], latency_s=round(time.time() - t0, 1))

    try:
        p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return result(False, "timeout", f"pi exceeded {timeout:.0f}s")
    (workdir / "events.json").write_text(p.stdout or "")
    (workdir / "pi.err").write_text(p.stderr or "")

    tr = _parse_trajectory(workdir / "events.json")
    if verbose:
        print(f"    {problem.id}: exit={p.returncode} settled={tr['settled']} "
              f"steps={tr['steps']} tools={tr['tool_calls']}(err {tr['tool_errors']}) "
              f"answer={tr['answer']!r}")

    # Did pi complete? If not, the serving / tool-call layer failed -> tool-reliability axis.
    if p.returncode != 0 or not tr["settled"]:
        err = (p.stderr or "").strip().splitlines()
        tail = err[-1] if err else f"exit {p.returncode}, no agent_settled"
        return result(False, "model_error", f"pi did not complete: {tail[:200]}",
                      tr["steps"], tr["tool_calls"], tr["messages"])

    ctx.answer = tr["answer"]  # the agent's final message (for answer-graded problems)
    passed, status, reason = grade(problem, ctx)
    if not passed and tr["tool_errors"]:
        reason = f"{reason} [{tr['tool_errors']} tool error(s) during run]".strip()
    return result(passed, status, reason, tr["steps"], tr["tool_calls"], tr["messages"])


def _render(meta: dict, results: list[AgentResult]) -> str:
    by: dict[str, list[AgentResult]] = defaultdict(list)
    for r in results:
        by[r.id].append(r)
    rep = meta.get("repeat", 1)
    total_pass = sum(r.passed for r in results)
    total = len(results)
    maj = sum(1 for rs in by.values() if sum(x.passed for x in rs) * 2 > len(rs))
    overall = f"- overall: **{total_pass}/{total}** passing runs"
    if rep > 1:
        overall += f"  ·  **{maj}/{len(by)}** problems pass by majority"
    lines = [
        f"# dsbench agentic run (pi harness): {meta.get('label')}",
        "",
        f"- harness: `pi` (thinking {meta.get('thinking')})  model: `{meta.get('model')}`  "
        f"provider: `{meta.get('provider')}`  k={rep}",
        f"- when: {meta.get('timestamp')}",
        overall,
        "",
        "| id | category | difficulty | passes | statuses | median latency | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for pid in sorted(by):
        rs = by[pid]
        p = sum(x.passed for x in rs)
        st = ", ".join(f"{k}×{v}" for k, v in Counter(x.status for x in rs).most_common())
        note = ""
        if p < len(rs):
            fail = next((x for x in rs if not x.passed), None)
            note = fail.reason.replace("|", "\\|")[:70] if fail else ""
        lines.append(f"| {pid} | {rs[0].category} | {rs[0].difficulty} | {p}/{len(rs)} | "
                     f"{st} | {round(median([x.latency_s for x in rs]), 1)}s | {note} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the agentic aviation benchmark via pi.")
    ap.add_argument("--provider", default="dashi-ornith")
    ap.add_argument("--model", default="ornith")
    ap.add_argument("--thinking", default="high",
                    help="pi thinking level: off|minimal|low|medium|high|xhigh|max (default high)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--category", choices=["de", "da", "ds"], default=None)
    ap.add_argument("--ids", default=None, help="comma-separated problem ids")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-problem wall-clock cap (s)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per problem (k); temp-0 thinking is nondeterministic, so k>1 gives "
                         "per-problem pass rates instead of a noisy single number")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-dir", default=str(_RUNS))
    args = ap.parse_args()

    ids = set(args.ids.split(",")) if args.ids else None
    problems = load_problems(args.category, ids)
    if not problems:
        raise SystemExit("no agentic problems matched the filters")

    label = args.label or dt.datetime.now().strftime("pi-%Y%m%d-%H%M%S")
    print(f"running {len(problems)} problem(s) via pi, thinking {args.thinking}, k={args.repeat}")
    results: list[AgentResult] = []
    for pr in problems:
        for k in range(args.repeat):
            try:
                ctx = prepare_context(pr.id)
                if pr.setup:
                    pr.setup(ctx)
            except Exception as e:  # noqa: BLE001
                results.append(AgentResult(pr.id, pr.category, pr.difficulty, False, "setup_error",
                                           reason=str(e)[:200]))
                print(f"  SETUP_ERROR   {pr.id}: {e}")
                continue
            r = run_pi_agent(pr, ctx, provider=args.provider, model=args.model,
                             thinking=args.thinking, timeout=args.timeout, verbose=args.verbose)
            results.append(r)
            mark = "PASS" if r.passed else f"FAIL[{r.status}]"
            tag = f" [{k + 1}/{args.repeat}]" if args.repeat > 1 else ""
            extra = f"  {r.reason[:80]}" if r.reason and not r.passed else ""
            print(f"  {mark:16} {pr.id}{tag} ({r.steps} steps, {r.tool_calls} tools, "
                  f"{r.latency_s}s){extra}")

    meta = {
        "label": label, "harness": "pi", "model": args.model, "provider": args.provider,
        "thinking": args.thinking, "repeat": args.repeat,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "n": len(results),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{label}"
    payload = {"meta": meta, "results": [dataclasses.asdict(r) for r in results]}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / f"{stem}.md").write_text(_render(meta, results))
    print("\n" + _render(meta, results))
    npass = sum(r.passed for r in results)
    print(f"score: {npass}/{len(results)}  (wrote {out_dir / f'{stem}.json'})")


if __name__ == "__main__":
    main()
