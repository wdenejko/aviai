"""Target C generator -- agentic ML-workflow delivery trajectories (ADR-004).

The measured gap: the models are competent at the ML but fail to DELIVER -- they over-tune or end
the turn without writing the exact-schema output table (`ds_*` all land at 1-4/5). This generator
runs a
licence-clean teacher AS AN AGENT in the ADR-003 sandbox on the synthetic `ml_tasks`, and KEEPS ONLY
trajectories that PASS the oracle (correct-schema table, every row predicted, metric beats the bar).
A kept trajectory is therefore, by construction, a demonstration of finishing the loop -- exactly
the behaviour to reinforce.

It reuses the measurement harness's executors (`tools.run_sql/run_python`) and stream reassembly
(`loop._reassemble_stream`), but with a GENERIC, aviation-free system prompt and tool schemas
(`ml_tasks.SYSTEM_C` / `TOOL_SCHEMAS_C`) so the trajectories never mention the benchmark schema.
Output is training-ready: OpenAI-format messages + the tool schemas + an assistant-only loss mask,
which the Qwen chat template renders into its native tool-call form at train time.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx

from dsbench.agentic import tools as T
from dsbench.agentic.ch import get_client
from dsbench.agentic.loop import _reassemble_stream, grade
from dsbench.agentic.schema import AgentProblem, GradeContext
from dsbench.sftgen.ml_tasks import ML_TASKS, SYSTEM_C, TOOL_SCHEMAS_C


def _call(messages: list, *, base_url: str, model: str, timeout: float = 300.0,
          max_tokens: int = 4096, attempts: int = 3) -> dict:
    """One streamed chat completion with the Target C tools; reassembles tool-calls from deltas."""
    body = {"model": model, "messages": messages, "tools": TOOL_SCHEMAS_C, "tool_choice": "auto",
            "temperature": 0.3, "max_tokens": max_tokens, "stream": True}
    last: Exception | None = None
    for i in range(attempts):
        try:
            with httpx.stream("POST", f"{base_url}/chat/completions", json=body,
                              timeout=timeout) as r:
                r.raise_for_status()
                return _reassemble_stream(r)
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last  # type: ignore[misc]


def run_teacher_agent(problem: AgentProblem, ctx: GradeContext, *, base_url: str, model: str,
                      verbose: bool = False) -> dict:
    """Drive the teacher through the sandbox. Returns {passed, status, reason, trajectory, ...}."""
    t0 = time.time()
    messages: list = [
        {"role": "system", "content": SYSTEM_C},
        {"role": "user", "content": problem.prompt},
    ]
    n_tool = 0
    for step in range(1, problem.max_steps + 1):
        try:
            choice = _call(messages, base_url=base_url, model=model)
        except Exception as e:  # noqa: BLE001
            return {"passed": False, "status": "model_error", "reason": str(e)[:200],
                    "trajectory": messages, "steps": step, "tool_calls": n_tool,
                    "latency_s": round(time.time() - t0, 1)}
        msg = choice.get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        if verbose:
            print(f"    step {step}: {len(tool_calls)} tool_call(s)"
                  + (f'; says {content[:60]!r}' if content and not tool_calls else ""))
        if not tool_calls:
            ctx.answer = content.strip()
            passed, status, reason = grade(problem, ctx)
            return {"passed": passed, "status": status, "reason": reason, "trajectory": messages,
                    "steps": step, "tool_calls": n_tool, "latency_s": round(time.time() - t0, 1)}
        # Malformed-tool-call recovery, mirroring the measurement loop.
        parsed: list[dict | None] = []
        for tc in tool_calls:
            try:
                parsed.append(json.loads((tc.get("function", {}) or {}).get("arguments") or "{}"))
            except json.JSONDecodeError:
                parsed.append(None)
        safe_calls = [
            {"id": tc.get("id"), "type": "function",
             "function": {"name": (tc.get("function", {}) or {}).get("name"),
                          "arguments": json.dumps(a if a is not None else {})}}
            for tc, a in zip(tool_calls, parsed, strict=True)
        ]
        messages.append({"role": "assistant", "content": content, "tool_calls": safe_calls})
        for tc, args in zip(tool_calls, parsed, strict=True):
            n_tool += 1
            name = (tc.get("function", {}) or {}).get("name")
            if args is None:
                obs = ("tool-call error: arguments were not valid JSON (truncated or bad "
                       "escaping). Resend with valid JSON; split long code across calls.")
            elif name == "finish":
                ctx.answer = args.get("answer")
                passed, status, reason = grade(problem, ctx)
                return {"passed": passed, "status": status, "reason": reason,
                        "trajectory": messages, "steps": step, "tool_calls": n_tool,
                        "latency_s": round(time.time() - t0, 1)}
            elif name == "run_sql":
                obs = T.run_sql(ctx.client, args.get("query", ""))
            elif name == "run_python":
                obs = T.run_python(args.get("code", ""), ctx.namespace)
            else:
                obs = f"error: unknown tool {name!r}"
            messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": obs})
    passed, status, reason = grade(problem, ctx)
    return {"passed": passed, "status": "ok" if passed else "budget", "reason": reason,
            "trajectory": messages, "steps": problem.max_steps, "tool_calls": n_tool,
            "latency_s": round(time.time() - t0, 1)}


def _prepare(task_id: str, run_ix: int) -> GradeContext:
    ns = f"sftc_{task_id}_{run_ix}"
    admin = get_client(database="default")
    admin.command(f"DROP DATABASE IF EXISTS {ns}")
    admin.command(f"CREATE DATABASE {ns}")
    return GradeContext(client=get_client(database=ns), namespace=ns)


def _record(task: AgentProblem, res: dict, model: str, run_ix: int) -> dict:
    return {
        "messages": res["trajectory"],
        "tools": TOOL_SCHEMAS_C,
        "loss_mask_roles": ["assistant"],
        "meta": {
            "id": f"C-{task.id}-{run_ix}", "target": "C", "family": "ml-delivery",
            "task": task.id, "tags": list(task.tags),
            "provenance": {"generator": "ml_delivery_trajectories", "method": "teacher-in-sandbox",
                           "teacher": model, "licence": "Apache-2.0"},
            "verification": {"engine": "clickhouse", "oracle_passed": True, "reason": res["reason"],
                             "steps": res["steps"], "tool_calls": res["tool_calls"]},
        },
    }


def generate(*, base_url: str, model: str, reps: int = 1, tasks: list[str] | None = None,
             verbose: bool = False, sink=None) -> tuple[list[dict], dict]:
    # `sink(record)` is called as each trajectory passes the oracle -- the caller writes+flushes it,
    # so a multi-hour Ling run never loses a delivered trajectory to a crash.
    picked = [t for t in ML_TASKS if not tasks or t.id in tasks]
    report: dict[str, Any] = {"emitted": 0, "failed": 0, "by_task": {},
                              "statuses": {}, "fail_reasons": []}
    records: list[dict] = []
    for run_ix in range(reps):
        for task in picked:
            ctx = _prepare(task.id, run_ix)
            try:
                task.setup(ctx)
                res = run_teacher_agent(task, ctx, base_url=base_url, model=model, verbose=verbose)
            finally:
                get_client(database="default").command(f"DROP DATABASE IF EXISTS {ctx.namespace}")
            report["statuses"][res["status"]] = report["statuses"].get(res["status"], 0) + 1
            if res["passed"]:
                rec = _record(task, res, model, run_ix)
                records.append(rec)
                if sink is not None:
                    sink(rec)
                report["emitted"] += 1
                report["by_task"][task.id] = report["by_task"].get(task.id, 0) + 1
            else:
                report["failed"] += 1
                report["fail_reasons"].append({"task": task.id, "status": res["status"],
                                               "reason": res["reason"][:120]})
    return records, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Target C: agentic ML-delivery trajectories (teacher)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--tasks", default="", help="comma list of task ids; default all")
    ap.add_argument("--base-url", default="http://localhost:18080/v1")
    ap.add_argument("--model", default="ling-3.0-flash-q6-mtp")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_fh = open(args.out, "w") if args.out else None  # closed in the finally below

    def _sink(rec) -> None:  # write+flush each passing trajectory immediately
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()

    try:
        records, report = generate(
            base_url=args.base_url, model=args.model, reps=args.reps,
            tasks=[t for t in args.tasks.split(",") if t] or None, verbose=args.verbose,
            sink=_sink if out_fh else None,
        )
    finally:
        if out_fh:
            out_fh.close()
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
    print(f"emitted: {report['emitted']}  failed: {report['failed']}")
    print(f"by task: {report['by_task']}  statuses: {report['statuses']}")
    for fr in report["fail_reasons"][:8]:
        print(f"  FAIL [{fr['task']}/{fr['status']}] {fr['reason']}")


if __name__ == "__main__":
    main()
