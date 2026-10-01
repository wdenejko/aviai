"""Target C generator -- agentic ML-workflow delivery trajectories (ADR-004).

The measured gap: the models are competent at the ML but fail to DELIVER -- they over-tune or end
the turn without writing the exact-schema output table (`ds_*` all land at 1-4/5). This generator
runs an agent in the ADR-003 sandbox on the synthetic `ml_tasks`, and KEEPS ONLY trajectories that
PASS the oracle (correct-schema table, every row predicted, metric beats the bar). A kept trajectory
is therefore, by construction, a demonstration of finishing the loop -- exactly the behaviour to
reinforce.

It reuses the measurement harness's executors (`tools.run_sql/run_python`) and stream reassembly
(`loop._reassemble_stream`), but with a GENERIC, aviation-free system prompt and tool schemas
(`ml_tasks.SYSTEM_C` / `TOOL_SCHEMAS_C`) so the trajectories never mention the benchmark schema.
Output is training-ready: OpenAI-format messages + the tool schemas + an assistant-only loss mask,
which the Qwen chat template renders into its native tool-call form at train time.

**The agent.** Gate 2's trajectories came from a teacher, Ling-3.0-flash (MIT), thinking off in
effect: its trace was never kept. ADR-004 Revision 2 makes the base the agent (`--thinking`):
Qwen3.6 runs the loop itself, thinking on, with Qwen's sampling, and every assistant turn keeps
its reasoning. A loop has one user message, so the chat template renders every turn's reasoning,
and the retrain labels each turn with it. The base sees its own earlier reasoning in each request
too, as the training row will show it. Ling stays the fallback if the base's yield is too low.

**Parallel runs.** `--workers` runs that many loops at once against a server with as many slots.
Each run has its own ClickHouse database, as before, and its own working directory in the shared
workspace container (`tools.run_python(workdir=...)`). Runs are ordered by run index, so the ones
running together are mostly different tasks.

**Seeds.** The run index is the dataset's seed (`ml_tasks._run_seed`), and topping a slice up must
start past the indices already used (`--run-offset`). `--oracle-only` runs the oracle on exactly
the datasets a run will use (setup, reference, check, no model) before any GPU time is spent.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import threading
import time
from typing import Any

import httpx

from dsbench.agentic import tools as T
from dsbench.agentic.ch import get_client
from dsbench.agentic.loop import _reassemble_stream, grade
from dsbench.agentic.schema import AgentProblem, GradeContext
from dsbench.sftgen.ml_tasks import ML_TASKS, SYSTEM_C, TOOL_SCHEMAS_C

# Qwen's recommended thinking-mode sampling, as the other Revision 2 pools use it
# (reasoning_pilot.SAMPLING). Gate 2's teacher ran at temperature 0.3.
BASE_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
TEACHER_SAMPLING = {"temperature": 0.3}


def request_body(messages: list, *, model: str, sampling: dict | None = None,
                 thinking: bool = False, max_tokens: int = 4096,
                 seed: int | None = None) -> dict:
    body: dict[str, Any] = {"model": model, "messages": messages, "tools": TOOL_SCHEMAS_C,
                            "tool_choice": "auto", **(sampling or TEACHER_SAMPLING),
                            "max_tokens": max_tokens, "stream": True,
                            "stream_options": {"include_usage": True}}
    if thinking:
        # The battery server's default is thinking off; this turns it on for these requests.
        body["chat_template_kwargs"] = {"enable_thinking": True}
    if seed is not None:
        body["seed"] = seed
    return body


def _call(messages: list, *, base_url: str, model: str, sampling: dict | None = None,
          thinking: bool = False, max_tokens: int = 4096, seed: int | None = None,
          timeout: float = 1800.0, attempts: int = 3) -> dict:
    """One streamed chat completion with the Target C tools; reassembles tool-calls from deltas.
    Returns {message, finish_reason, usage, timings}; the message keeps `reasoning_content`."""
    body = request_body(messages, model=model, sampling=sampling, thinking=thinking,
                        max_tokens=max_tokens, seed=seed)
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


def _assistant(content: str, reasoning: str, tool_calls: list | None = None) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning:
        message["reasoning_content"] = reasoning
    return message


def run_teacher_agent(problem: AgentProblem, ctx: GradeContext, *, base_url: str, model: str,
                      verbose: bool = False, sampling: dict | None = None, thinking: bool = False,
                      max_tokens: int = 4096, workdir: str | None = None,
                      seed: int | None = None) -> dict:
    """Drive the agent through the sandbox. Returns {passed, status, reason, trajectory, ...}.

    `turns` records each model call: its finish reason, token counts, reasoning length, tool calls
    and seconds. A turn cut by `max_tokens` ends the run with status `turn_cap`, never kept: its
    reasoning or call never closed, so the row couldn't be trained.
    """
    t0 = time.time()
    messages: list = [
        {"role": "system", "content": SYSTEM_C},
        {"role": "user", "content": problem.prompt},
    ]
    turns: list[dict] = []
    n_tool = 0

    def done(passed: bool, status: str, reason: str, step: int) -> dict:
        return {"passed": passed, "status": status, "reason": reason, "trajectory": messages,
                "steps": step, "tool_calls": n_tool, "turns": turns,
                "latency_s": round(time.time() - t0, 1)}

    for step in range(1, problem.max_steps + 1):
        t_call = time.time()
        try:
            choice = _call(messages, base_url=base_url, model=model, sampling=sampling,
                           thinking=thinking, max_tokens=max_tokens,
                           seed=None if seed is None else seed + step)
        except Exception as e:  # noqa: BLE001
            return done(False, "model_error", str(e)[:200], step)
        msg = choice.get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        usage = choice.get("usage") or {}
        turns.append({"step": step, "finish_reason": choice.get("finish_reason"),
                      "prompt_tokens": usage.get("prompt_tokens"),
                      "completion_tokens": usage.get("completion_tokens"),
                      "reasoning_chars": len(reasoning), "tool_calls": len(tool_calls),
                      "seconds": round(time.time() - t_call, 1)})
        if verbose:
            print(f"    step {step}: {len(tool_calls)} tool_call(s)"
                  + (f'; says {content[:60]!r}' if content and not tool_calls else ""))
        if choice.get("finish_reason") == "length":
            messages.append(_assistant(content, reasoning, tool_calls))
            passed, _, reason = grade(problem, ctx)
            return done(False, "turn_cap", f"turn {step} hit max_tokens (oracle would have "
                                           f"{'passed' if passed else 'failed'}: {reason})"[:300],
                        step)
        if not tool_calls:
            # The model ended its turn in text: that is its final message, kept with its reasoning.
            messages.append(_assistant(content, reasoning))
            ctx.answer = content.strip()
            passed, status, reason = grade(problem, ctx)
            return done(passed, status, reason, step)
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
        messages.append(_assistant(content, reasoning, safe_calls))
        for tc, args in zip(tool_calls, parsed, strict=True):
            n_tool += 1
            name = (tc.get("function", {}) or {}).get("name")
            if args is None:
                obs = ("tool-call error: arguments were not valid JSON (truncated or bad "
                       "escaping). Resend with valid JSON; split long code across calls.")
            elif name == "finish":
                ctx.answer = args.get("answer")
                passed, status, reason = grade(problem, ctx)
                return done(passed, status, reason, step)
            elif name == "run_sql":
                obs = T.run_sql(ctx.client, args.get("query", ""))
            elif name == "run_python":
                obs = T.run_python(args.get("code", ""), ctx.namespace, workdir=workdir)
            else:
                obs = f"error: unknown tool {name!r}"
            messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": obs})
    passed, status, reason = grade(problem, ctx)
    return done(passed, "ok" if passed else "budget", reason, problem.max_steps)


def _prepare(task_id: str, run_ix: int) -> GradeContext:
    ns = f"sftc_{task_id}_{run_ix}"
    admin = get_client(database="default")
    admin.command(f"DROP DATABASE IF EXISTS {ns}")
    admin.command(f"CREATE DATABASE {ns}")
    return GradeContext(client=get_client(database=ns), namespace=ns)


def _seed(task_id: str, run_ix: int) -> int:
    """A sampling seed per run (each step adds its number), so a rerun repeats its samples."""
    return int(hashlib.sha1(f"{task_id}:{run_ix}".encode()).hexdigest()[:8], 16)


def _record(task: AgentProblem, res: dict, model: str, run_ix: int,
            thinking: bool = False, sampling: dict | None = None) -> dict:
    method = "base-in-sandbox" if thinking else "teacher-in-sandbox"
    return {
        "messages": res["trajectory"],
        "tools": TOOL_SCHEMAS_C,
        "loss_mask_roles": ["assistant"],
        "meta": {
            "id": f"C-{task.id}-{run_ix}", "target": "C", "family": "ml-delivery",
            "task": task.id, "tags": list(task.tags), "run_ix": run_ix,
            "provenance": {"generator": "ml_delivery_trajectories", "method": method,
                           "teacher": model, "thinking": thinking,
                           "sampling": sampling or TEACHER_SAMPLING, "licence": "Apache-2.0"},
            "verification": {"engine": "clickhouse", "oracle_passed": True, "reason": res["reason"],
                             "status": res["status"], "steps": res["steps"],
                             "tool_calls": res["tool_calls"], "latency_s": res["latency_s"]},
            "turns": res.get("turns", []),
        },
    }


def jobs_for(tasks: list[str] | None, reps: int, run_offset: int) -> list[tuple[int, AgentProblem]]:
    """(run index, task) pairs, by run index first: runs that overlap are mostly different tasks."""
    picked = [t for t in ML_TASKS if not tasks or t.id in tasks]
    return [(run_ix, task) for run_ix in range(run_offset, run_offset + reps) for task in picked]


def oracle_gate(jobs: list[tuple[int, AgentProblem]]) -> dict:
    """setup -> reference -> check, no model, on exactly the datasets `jobs` will generate.

    `ml_tasks.selftest` checks the base seeds; a volume run uses run indices past them. A dataset
    whose own reference fails would discard a good trajectory, so it is caught before the run.
    """
    out: dict[str, Any] = {"passed": 0, "failed": []}
    for run_ix, task in jobs:
        ctx = _prepare(task.id, run_ix)
        try:
            task.setup(ctx)
            ctx.answer = task.reference(ctx)
            ok, _, reason = grade(task, ctx)
        finally:
            get_client(database="default").command(f"DROP DATABASE IF EXISTS {ctx.namespace}")
        if ok:
            out["passed"] += 1
        else:
            out["failed"].append({"task": task.id, "run_ix": run_ix, "reason": reason[:200]})
    return out


def generate(*, base_url: str, model: str, reps: int = 1, tasks: list[str] | None = None,
             verbose: bool = False, sink=None, run_offset: int = 0,
             fail_sink=None, workers: int = 1, thinking: bool = False,
             sampling: dict | None = None, max_tokens: int = 4096) -> tuple[list[dict], dict]:
    # `sink(record)` is called as each trajectory passes the oracle -- the caller writes+flushes it,
    # so a multi-hour run never loses a delivered trajectory to a crash.
    #
    # `run_offset` exists because the run index IS the dataset seed (ml_tasks._run_seed reads it off
    # the namespace). Topping an existing slice up therefore has to START past the indices already
    # generated -- otherwise a second pass silently re-creates the same datasets and the pool fills
    # with duplicate trajectories that nothing downstream would flag: the assembler decontaminates
    # against dsbench, not against targetC itself.
    report: dict[str, Any] = {"emitted": 0, "failed": 0, "by_task": {},
                              "statuses": {}, "fail_reasons": [], "runs": []}
    records: list[dict] = []
    lock = threading.Lock()

    def one(job: tuple[int, AgentProblem]) -> None:
        run_ix, task = job
        ctx = None
        try:
            ctx = _prepare(task.id, run_ix)
            workdir = f"/tmp/sftc/{ctx.namespace}"
            T.make_workdir(workdir)
            task.setup(ctx)
            res = run_teacher_agent(task, ctx, base_url=base_url, model=model, verbose=verbose,
                                    sampling=sampling, thinking=thinking, max_tokens=max_tokens,
                                    workdir=workdir, seed=_seed(task.id, run_ix))
        except Exception as e:  # noqa: BLE001 - a harness failure is recorded, not a crash
            res = {"passed": False, "status": "harness_error", "reason": str(e)[:200],
                   "trajectory": [], "steps": 0, "tool_calls": 0, "turns": [], "latency_s": 0}
        finally:
            if ctx is not None:
                get_client(database="default").command(f"DROP DATABASE IF EXISTS {ctx.namespace}")
        rec = _record(task, res, model, run_ix, thinking, sampling)
        with lock:
            report["statuses"][res["status"]] = report["statuses"].get(res["status"], 0) + 1
            report["runs"].append({"task": task.id, "run_ix": run_ix, "status": res["status"],
                                   "passed": res["passed"], "steps": res["steps"],
                                   "tool_calls": res["tool_calls"],
                                   "latency_s": res["latency_s"]})
            if res["passed"]:
                records.append(rec)
                if sink is not None:
                    sink(rec)
                report["emitted"] += 1
                report["by_task"][task.id] = report["by_task"].get(task.id, 0) + 1
            else:
                report["failed"] += 1
                report["fail_reasons"].append({"task": task.id, "status": res["status"],
                                               "reason": res["reason"][:120]})
                # Failures are NOT training data, but discarding them silently makes a low-yield
                # family undiagnosable: the report gives the metric and nothing about the
                # reasoning that produced it. mlc_energy_load ran at ~64% yield with no way to
                # see why.
                if fail_sink is not None:
                    rec["meta"]["verification"]["oracle_passed"] = False
                    fail_sink(rec)
            if verbose:
                print(f"[{len(report['runs'])}] {task.id} #{run_ix}: {res['status']} "
                      f"steps={res['steps']} {res['latency_s']}s", flush=True)

    with cf.ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, jobs_for(tasks, reps, run_offset)))
    return records, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Target C: agentic ML-delivery trajectories")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--run-offset", type=int, default=0,
                    help="first run index; the index seeds the dataset, so topping up an existing "
                         "slice must start past the indices it already used")
    ap.add_argument("--tasks", default="", help="comma list of task ids; default all")
    ap.add_argument("--base-url", default="http://localhost:18080/v1")
    ap.add_argument("--model", default="ling-3.0-flash-q6-mtp")
    ap.add_argument("--thinking", action="store_true",
                    help="the base as the agent (ADR-004 Revision 2): thinking on, Qwen's sampling")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--max-tokens", type=int, default=4096, help="per model turn")
    ap.add_argument("--workers", type=int, default=1, help="loops at once (the server's slots)")
    ap.add_argument("--oracle-only", action="store_true",
                    help="run the oracle on the run's datasets (no model) and exit")
    ap.add_argument("--out", default="")
    ap.add_argument("--fail-out", default="",
                    help="also write REJECTED trajectories here, for diagnosing a low-yield family "
                         "(never training data -- meta.verification.oracle_passed is false)")
    ap.add_argument("--report", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    tasks = [t for t in args.tasks.split(",") if t] or None

    if args.oracle_only:
        gate = oracle_gate(jobs_for(tasks, args.reps, args.run_offset))
        print(json.dumps(gate, indent=1))
        raise SystemExit(1 if gate["failed"] else 0)

    sampling = dict(BASE_SAMPLING if args.thinking else TEACHER_SAMPLING)
    for key, value in (("temperature", args.temperature), ("top_p", args.top_p),
                       ("top_k", args.top_k)):
        if value is not None:
            sampling[key] = value

    out_fh = open(args.out, "w") if args.out else None  # closed in the finally below
    fail_fh = open(args.fail_out, "w") if args.fail_out else None

    def _sink(rec) -> None:  # write+flush each passing trajectory immediately
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()

    def _fail_sink(rec) -> None:
        fail_fh.write(json.dumps(rec) + "\n")
        fail_fh.flush()

    try:
        records, report = generate(
            base_url=args.base_url, model=args.model, reps=args.reps, tasks=tasks,
            verbose=args.verbose, sink=_sink if out_fh else None, run_offset=args.run_offset,
            fail_sink=_fail_sink if fail_fh else None, workers=args.workers,
            thinking=args.thinking, sampling=sampling, max_tokens=args.max_tokens,
        )
    finally:
        if out_fh:
            out_fh.close()
        if fail_fh:
            fail_fh.close()
    report["params"] = {"model": args.model, "thinking": args.thinking, "sampling": sampling,
                        "max_tokens": args.max_tokens, "workers": args.workers,
                        "reps": args.reps, "run_offset": args.run_offset}
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
    print(f"emitted: {report['emitted']}  failed: {report['failed']}")
    print(f"by task: {report['by_task']}  statuses: {report['statuses']}")
    for fr in report["fail_reasons"][:8]:
        print(f"  FAIL [{fr['task']}/{fr['status']}] {fr['reason']}")


if __name__ == "__main__":
    main()
