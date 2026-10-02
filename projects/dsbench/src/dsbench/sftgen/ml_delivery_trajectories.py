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

**The agent's own login** (ADR-003 §7). Each task writes its withheld labels into the run's own
database (`<table>_test_key`), where the oracle reads them. The agent used to run as the sandbox's
admin and could see them: Gate 2's Ling trajectories listed them in 15 of 350 runs, and read
none. Since 2026-10-01 it runs as a ClickHouse login made for the run (`access.open_login`). The
login has its database without the key, which drops out of SHOW TABLES, and no shared data, since
the tasks are synthetic. Its run_sql client and its run_python credentials are the login's; the
system prompt already told it to connect with whatever CLICKHOUSE_* holds. The oracle keeps the
admin's client.

**Which runs become rows** (`selection`). The oracle passing is necessary, not sufficient. With
thinking on, every assistant turn must carry reasoning, or `thinking_record` rejects the row. No
tool call may reach for the withheld key or the admin's login (`access.breach`): such a run would
pass by copying the labels, and its row would teach exactly that. The agent must have ended the
loop itself, with `finish` or a final reply: a run that used up its steps after writing a passing
table teaches tool calls that never stop, which Gate 2's battery found. With `--block`, the row
must also fit the training block; its length comes from the server's own counts.

**Volume: quotas** (`--quota`, `generate_quota`). ADR-004 Revision 2's pilot kept 1 run in 3 for
credit_leak, energy_load and upsell_join and every run for three other families, so a fixed
number of runs per family fills the slice mostly with the easy ones. In quota mode each family
runs until it has its rows:
- run indices count up from `--run-offset`, per family, and each dataset passes its own oracle
  before the agent sees it;
- a free slot goes to the family furthest from its quota, counting each of its running loops at
  its keep rate so far;
- `--max-minutes` stops new runs before the GPU window ends, `--resume` continues from the
  output files, and a run of consecutive model errors (the server gone) stops it.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from dsbench.agentic import tools as T
from dsbench.agentic.access import AgentLogin, breach, close_login, open_login
from dsbench.agentic.ch import get_client
from dsbench.agentic.loop import _reassemble_stream, grade
from dsbench.agentic.schema import AgentProblem, GradeContext
from dsbench.sftgen.ml_tasks import ML_TASKS, SYSTEM_C, TOOL_SCHEMAS_C

# Qwen's recommended thinking-mode sampling, as the other Revision 2 pools use it
# (reasoning_pilot.SAMPLING). Gate 2's teacher ran at temperature 0.3.
BASE_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
TEACHER_SAMPLING = {"temperature": 0.3}

# Runs that say nothing about a family's yield: its dataset failed its own oracle, or the agent
# never got a verdict because the server or the sandbox failed.
NO_VERDICT = ("dataset_failed_oracle", "model_error", "harness_error")


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
          timeout: float = 1800.0, attempts: int = 3, reconnect_s: float = 180.0,
          sleep=time.sleep, clock=time.monotonic) -> dict:
    """One streamed chat completion with the Target C tools; reassembles tool-calls from deltas.
    Returns {message, finish_reason, usage, timings}; the message keeps `reasoning_content`.

    An HTTP error is retried `attempts` times. A lost connection is retried for `reconnect_s`
    seconds. The volume run reaches the box through an SSH tunnel that restarts itself within
    seconds (patches/target_c_volume_mac.sh); without the wait, one drop would fail every loop in
    flight, and their errors in a row would stop the run. A request that times out has taken
    longer than that, and fails at once.
    """
    body = request_body(messages, model=model, sampling=sampling, thinking=thinking,
                        max_tokens=max_tokens, seed=seed)
    t0 = clock()
    tries = 0
    while True:
        try:
            with httpx.stream("POST", f"{base_url}/chat/completions", json=body,
                              timeout=timeout) as r:
                r.raise_for_status()
                return _reassemble_stream(r)
        except httpx.HTTPStatusError:
            tries += 1
            if tries >= attempts:
                raise
            sleep(2 * tries)
        except httpx.TransportError:
            if clock() - t0 >= reconnect_s:
                raise
            sleep(5)


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
                      seed: int | None = None, login: AgentLogin | None = None) -> dict:
    """Drive the agent through the sandbox. Returns {passed, status, reason, trajectory, ...}.

    `turns` records each model call: its finish reason, token counts, reasoning length, tool calls
    and seconds. A turn cut by `max_tokens` ends the run with status `turn_cap`, never kept: its
    reasoning or call never closed, so the row couldn't be trained.

    The agent's tools run as `login`, which `run_job` always passes. Without one they run as the
    admin `ctx.client` logs in as. The oracle grades through `ctx.client` either way.
    """
    t0 = time.time()
    agent_client = login.client() if login else ctx.client
    agent_env = login.env if login else None
    messages: list = [
        {"role": "system", "content": SYSTEM_C},
        {"role": "user", "content": problem.prompt},
    ]
    turns: list[dict] = []
    n_tool = 0

    def done(passed: bool, status: str, reason: str, step: int, finished: bool = False) -> dict:
        # `finished`: the agent ended the loop itself (`finish`, or a reply in text). The oracle
        # also grades a run that used up its steps, as the measurement loop does.
        return {"passed": passed, "status": status, "reason": reason, "trajectory": messages,
                "steps": step, "tool_calls": n_tool, "turns": turns, "finished": finished,
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
            return done(passed, status, reason, step, finished=True)
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
                return done(passed, status, reason, step, finished=True)
            elif name == "run_sql":
                obs = T.run_sql(agent_client, args.get("query", ""))
            elif name == "run_python":
                obs = T.run_python(args.get("code", ""), ctx.namespace, workdir=workdir,
                                   env=agent_env)
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
            "verification": {"engine": "clickhouse", "oracle_passed": bool(res.get("passed")),
                             "reason": res["reason"], "status": res["status"],
                             "steps": res["steps"], "tool_calls": res["tool_calls"],
                             "latency_s": res["latency_s"]},
            "turns": res.get("turns", []),
        },
    }


def selection(res: dict, *, block: int | None = None, reasoning: bool = False) -> dict:
    """Whether a run's trajectory becomes a training row, and if not, why (`why_not`).

    In order: the oracle passed (else `why_not` is the run's status); with `reasoning`, every
    assistant turn carries some (`empty_reasoning`); no tool call reaches for a table the run
    withheld (`res["withheld"]`, set by `run_job`) or the admin's login (`withheld_access`); the
    agent ended the loop itself (`res["finished"]`, else `out_of_steps`); with a `block`, the row
    fits it (`over_block`).

    The length is the server's: the loop's last request, prompt plus completion, is the whole
    conversation. The training row is at most one token longer, the newline the template writes
    after the final `<|im_end|>`. On the pilot's 21 runs it was exactly one longer on 16 and no
    longer on the other 5, whose `finish` arguments the template rebuilds from the parsed value;
    on the volume run's 282, the same, except where tool output follows the last request. A turn
    that calls a tool beside `finish` ends the row on that tool's output, which no request
    carried: it counts as at most a token a character (numeric output tokenizes digit by digit),
    plus the response's wrapper (`_unsent_tail`). In the block, the row is followed by its
    separator (`tokenize_masked.pack_thinking`), so the row itself can be at most `block - 1`.
    A withheld table seen in a tool's output is recorded (`saw_withheld`) but not held against the
    run: its row shows the agent leaving it alone. The login keeps the key out of listings, so it
    now shows up only in the refusal of a call that named it.
    """
    trajectory = res.get("trajectory") or []
    withheld = tuple(res.get("withheld") or ())
    assistant = [m for m in trajectory if m.get("role") == "assistant"]
    last = (res.get("turns") or [{}])[-1]
    served = None
    if last.get("prompt_tokens") is not None and last.get("completion_tokens") is not None:
        served = last["prompt_tokens"] + last["completion_tokens"]
    tail = _unsent_tail(trajectory)
    row = None if served is None else served + 1 + tail  # at most; the template's newline
    out = {
        "block": block, "served_tokens": served, "unsent_tail_max": tail,
        "fits": None if block is None else row is not None and row + 1 <= block,  # + separator
        "finished": res.get("finished", True),
        "reasoning_in_every_turn": bool(assistant) and all(
            (m.get("reasoning_content") or "").strip() for m in assistant),
        "withheld_access": breach(trajectory, withheld),
        "saw_withheld": any(re.search(rf"\b{re.escape(t)}\b", m.get("content") or "")
                            for t in withheld for m in trajectory if m.get("role") == "tool"),
    }
    if not res.get("passed"):
        out["why_not"] = res.get("status") or "failed"
    elif reasoning and not out["reasoning_in_every_turn"]:
        out["why_not"] = "empty_reasoning"
    elif out["withheld_access"]:
        out["why_not"] = "withheld_access"
    elif not out["finished"]:
        out["why_not"] = "out_of_steps"
    elif out["fits"] is False:
        out["why_not"] = "over_block"
    else:
        out["why_not"] = None
    out["kept"] = out["why_not"] is None
    return out


def _unsent_tail(trajectory: list) -> int:
    """At most how many tokens the tool output after the last request adds to the row. In the
    volume run, 521 characters of numeric output took 504 tokens with their wrapper."""
    tail = 0
    for message in reversed(trajectory):
        if message.get("role") != "tool":
            break
        tail += len(message.get("content") or "") + 16
    return tail


def _skipped(status: str, reason: str) -> dict:
    """A run that never reached a verdict from the agent, in run_teacher_agent's result shape."""
    return {"passed": False, "status": status, "reason": reason[:200], "trajectory": [],
            "steps": 0, "tool_calls": 0, "turns": [], "latency_s": 0}


def dataset_ok(task: AgentProblem, run_ix: int) -> tuple[bool, str]:
    """setup -> reference -> check on one run's dataset, with no model, in that run's database.

    `ml_tasks.selftest` checks the base seeds, and a volume run uses run indices past them. A
    dataset whose own reference fails its bar would cost an agent run and say nothing about the
    agent: `mlc_churn_rare` at run index 100 has AP 0.149 against a bar of 0.15.
    """
    ctx = _prepare(task.id, run_ix)
    try:
        task.setup(ctx)
        ctx.answer = task.reference(ctx)
        ok, _, reason = grade(task, ctx)
    finally:
        _drop(ctx.namespace)
    return ok, reason


def run_job(task: AgentProblem, run_ix: int, *, base_url: str, model: str, verbose: bool = False,
            sampling: dict | None = None, thinking: bool = False, max_tokens: int = 4096,
            gate: bool = False) -> dict:
    """One run in its own database and working directory, dropped afterwards. With `gate`, the
    dataset first passes its own oracle, or the agent never sees it (`dataset_failed_oracle`).
    A harness failure is returned as the run's result, never raised.

    The agent runs as a login made after setup and dropped after the run. The result records the
    tables it withheld (`withheld`), for `selection`."""
    if gate:
        try:
            ok, reason = dataset_ok(task, run_ix)
        except Exception as e:  # noqa: BLE001
            return _skipped("harness_error", f"oracle gate: {e}")
        if not ok:
            return _skipped("dataset_failed_oracle", reason)
    ctx = login = None
    try:
        ctx = _prepare(task.id, run_ix)
        workdir = f"/tmp/sftc/{ctx.namespace}"
        T.make_workdir(workdir)
        task.setup(ctx)
        login = open_login(ctx, task.withheld_rows, shared=None)  # synthetic: no shared data
        res = run_teacher_agent(task, ctx, base_url=base_url, model=model, verbose=verbose,
                                sampling=sampling, thinking=thinking, max_tokens=max_tokens,
                                workdir=workdir, seed=_seed(task.id, run_ix), login=login)
        res["withheld"] = list(login.withheld)
        return res
    except Exception as e:  # noqa: BLE001
        return _skipped("harness_error", str(e))
    finally:
        if login is not None:
            # Best effort, like _drop: a login left behind reaches only its dropped database, and
            # the next login of its namespace replaces it.
            with contextlib.suppress(Exception):
                close_login(login)
        if ctx is not None:
            _drop(ctx.namespace)


def _drop(namespace: str) -> None:
    # Best effort: a database left behind is dropped by the next `_prepare` of its namespace, and
    # an error here must not replace the run's result.
    try:
        get_client(database="default").command(f"DROP DATABASE IF EXISTS {namespace}")
    except Exception:  # noqa: BLE001, S110
        pass


def jobs_for(tasks: list[str] | None, reps: int, run_offset: int) -> list[tuple[int, AgentProblem]]:
    """(run index, task) pairs, by run index first: runs that overlap are mostly different tasks."""
    picked = [t for t in ML_TASKS if not tasks or t.id in tasks]
    return [(run_ix, task) for run_ix in range(run_offset, run_offset + reps) for task in picked]


def oracle_gate(jobs: list[tuple[int, AgentProblem]]) -> dict:
    """`dataset_ok` on exactly the datasets `jobs` will generate, before any GPU time is spent."""
    out: dict[str, Any] = {"passed": 0, "failed": []}
    for run_ix, task in jobs:
        ok, reason = dataset_ok(task, run_ix)
        if ok:
            out["passed"] += 1
        else:
            out["failed"].append({"task": task.id, "run_ix": run_ix, "reason": reason[:200]})
    return out


def _new_report() -> dict[str, Any]:
    return {"emitted": 0, "failed": 0, "by_task": {}, "statuses": {}, "why_not": {},
            "fail_reasons": [], "runs": []}


def _settle(task: AgentProblem, run_ix: int, res: dict, *, report: dict, records: list,
            model: str, thinking: bool, sampling: dict | None, block: int | None,
            sink=None, fail_sink=None, verbose: bool = False) -> dict:
    """Record one finished run: its row goes to `sink` if `selection` keeps it, else to
    `fail_sink`. Returns the selection."""
    sel = selection(res, block=block, reasoning=thinking)
    rec = _record(task, res, model, run_ix, thinking, sampling)
    rec["meta"]["selection"] = sel
    report["statuses"][res["status"]] = report["statuses"].get(res["status"], 0) + 1
    report["runs"].append({"task": task.id, "run_ix": run_ix, "status": res["status"],
                           "passed": res["passed"], "kept": sel["kept"], "why_not": sel["why_not"],
                           "served_tokens": sel["served_tokens"], "steps": res["steps"],
                           "tool_calls": res["tool_calls"], "latency_s": res["latency_s"]})
    if sel["kept"]:
        records.append(rec)
        if sink is not None:
            sink(rec)
        report["emitted"] += 1
        report["by_task"][task.id] = report["by_task"].get(task.id, 0) + 1
    else:
        report["failed"] += 1
        report["why_not"][sel["why_not"]] = report["why_not"].get(sel["why_not"], 0) + 1
        report["fail_reasons"].append({"task": task.id, "status": res["status"],
                                       "why_not": sel["why_not"], "reason": res["reason"][:120]})
        # Not training data, but discarding these silently makes a low-yield family
        # undiagnosable: the report gives the metric and nothing about the reasoning that produced
        # it. mlc_energy_load ran at ~64% yield with no way to see why. A row that passed the
        # oracle but is over the block keeps `oracle_passed: true`, for a longer step later.
        if fail_sink is not None:
            fail_sink(rec)
    if verbose:
        note = "" if sel["kept"] else f" (not kept: {sel['why_not']})"
        print(f"[{len(report['runs'])}] {task.id} #{run_ix}: {res['status']}{note} "
              f"steps={res['steps']} tokens={sel['served_tokens']} {res['latency_s']}s",
              flush=True)
    return sel


def generate(*, base_url: str, model: str, reps: int = 1, tasks: list[str] | None = None,
             verbose: bool = False, sink=None, run_offset: int = 0,
             fail_sink=None, workers: int = 1, thinking: bool = False,
             sampling: dict | None = None, max_tokens: int = 4096,
             block: int | None = None) -> tuple[list[dict], dict]:
    # `sink(record)` is called as each kept trajectory arrives -- the caller writes+flushes it, so
    # a multi-hour run never loses a delivered trajectory to a crash.
    #
    # `run_offset` exists because the run index IS the dataset seed (ml_tasks._run_seed reads it off
    # the namespace). Topping an existing slice up therefore has to START past the indices already
    # generated -- otherwise a second pass silently re-creates the same datasets and the pool fills
    # with duplicate trajectories that nothing downstream would flag: the assembler decontaminates
    # against dsbench, not against targetC itself.
    report = _new_report()
    records: list[dict] = []
    lock = threading.Lock()

    def one(job: tuple[int, AgentProblem]) -> None:
        run_ix, task = job
        res = run_job(task, run_ix, base_url=base_url, model=model, verbose=verbose,
                      sampling=sampling, thinking=thinking, max_tokens=max_tokens)
        with lock:
            _settle(task, run_ix, res, report=report, records=records, model=model,
                    thinking=thinking, sampling=sampling, block=block, sink=sink,
                    fail_sink=fail_sink, verbose=verbose)

    with cf.ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, jobs_for(tasks, reps, run_offset)))
    return records, report


@dataclass
class _Family:
    """One task family's progress toward its quota."""

    task: AgentProblem
    next_ix: int
    started: int = 0  # run indices used, whatever came of them
    running: int = 0
    judged: int = 0  # runs that reached a verdict (not in NO_VERDICT)
    kept: int = 0
    skipped: int = 0  # datasets that failed their own oracle
    errors: int = 0  # model or harness errors

    def need(self, quota: int) -> float:
        # Rows still missing, counting each running loop at the family's keep rate so far
        # (Laplace's estimate: 1/2 before any verdict). A family whose loops are mostly kept
        # waits for them; one that keeps 1 in 3 gets more slots near the end.
        rate = (self.kept + 1) / (self.judged + 2)
        return quota - self.kept - self.running * rate


def _pick(families: list[_Family], quota: int, max_runs: int | None) -> _Family | None:
    """The family furthest from its quota that may still start a run; ties go to ML_TASKS order."""
    best, best_need = None, 0.0
    for fam in families:
        if fam.kept >= quota or (max_runs is not None and fam.started >= max_runs):
            continue
        need = fam.need(quota)
        if need > best_need:
            best, best_need = fam, need
    return best


def read_records(path: str) -> list[dict]:
    """The records in a JSONL output, or none if the file isn't there."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def generate_quota(*, base_url: str, model: str, quota: int, tasks: list[str] | None = None,
                   run_offset: int = 0, max_runs: int | None = None, block: int | None = 8192,
                   workers: int = 1, thinking: bool = False, sampling: dict | None = None,
                   max_tokens: int = 4096, max_seconds: float | None = None,
                   prior: list[dict] = (), sink=None, fail_sink=None, verbose: bool = False,
                   clock=time.monotonic) -> tuple[list[dict], dict]:
    """Run each family until `quota` of its rows are kept (`selection`), `max_runs` datasets are
    used, or `max_seconds` pass; runs already started finish either way.

    `prior` holds the records of an earlier pass (`--resume`): their kept rows count toward the
    quota, and each family starts past the highest run index they used. Indices of runs lost in
    flight are skipped, never reused, so no dataset appears twice.

    The run stops early after `max(3, workers)` consecutive runs end in a model or harness error:
    the server or the sandbox is gone, and every further run would use up a dataset in seconds.
    """
    picked = [t for t in ML_TASKS if not tasks or t.id in tasks]
    families = {t.id: _Family(task=t, next_ix=run_offset) for t in picked}
    prior_counts: dict[str, dict[str, int]] = {}
    for rec in prior:
        meta = rec["meta"]
        fam = families.get(meta["task"])
        if fam is None:
            continue
        if "selection" not in meta:
            raise ValueError(f"{meta['id']}: no selection record; --resume needs quota-mode output")
        status = meta["verification"]["status"]
        fam.started += 1
        fam.next_ix = max(fam.next_ix, meta["run_ix"] + 1)
        fam.kept += bool(meta["selection"]["kept"])
        fam.judged += status not in NO_VERDICT
        fam.skipped += status == "dataset_failed_oracle"
        counts = prior_counts.setdefault(meta["task"], {"runs": 0, "kept": 0})
        counts["runs"] += 1
        counts["kept"] += bool(meta["selection"]["kept"])

    report = _new_report()
    report.update({"mode": "quota", "quota": quota, "block": block, "run_offset": run_offset,
                   "max_runs": max_runs, "max_seconds": max_seconds, "prior": prior_counts})
    records: list[dict] = []
    t0 = clock()
    stop = None
    error_streak = 0
    order = list(families.values())
    with cf.ThreadPoolExecutor(max(1, workers)) as pool:
        running: dict[cf.Future, tuple[_Family, int]] = {}
        while True:
            while stop is None and len(running) < max(1, workers):
                if max_seconds is not None and clock() - t0 >= max_seconds:
                    stop = "time limit"
                    break
                fam = _pick(order, quota, max_runs)
                if fam is None:
                    break
                run_ix = fam.next_ix
                fam.next_ix += 1
                fam.started += 1
                fam.running += 1
                future = pool.submit(run_job, fam.task, run_ix, base_url=base_url, model=model,
                                     verbose=verbose, sampling=sampling, thinking=thinking,
                                     max_tokens=max_tokens, gate=True)
                running[future] = (fam, run_ix)
            if not running:
                break
            finished, _ = cf.wait(list(running), return_when=cf.FIRST_COMPLETED)
            for future in finished:
                fam, run_ix = running.pop(future)
                res = future.result()
                fam.running -= 1
                status = res["status"]
                fam.judged += status not in NO_VERDICT
                fam.skipped += status == "dataset_failed_oracle"
                fam.errors += status in ("model_error", "harness_error")
                error_streak = error_streak + 1 if status in ("model_error", "harness_error") else 0
                sel = _settle(fam.task, run_ix, res, report=report, records=records, model=model,
                              thinking=thinking, sampling=sampling, block=block, sink=sink,
                              fail_sink=fail_sink, verbose=verbose)
                fam.kept += sel["kept"]
                if verbose:
                    print(f"    {fam.task.id}: {fam.kept}/{quota} kept, {fam.running} running",
                          flush=True)
            if stop is None and error_streak >= max(3, workers):
                stop = f"{error_streak} consecutive model or harness errors"
    if stop is None:
        short = [f.task.id for f in order if f.kept < quota]
        stop = "quotas filled" if not short else f"max runs reached: {', '.join(short)}"
    report["stop"] = stop
    report["elapsed_s"] = round(clock() - t0, 1)
    report["families"] = {
        f.task.id: {"kept": f.kept, "shortfall": max(0, quota - f.kept),
                    "runs_with_a_verdict": f.judged, "datasets_used": f.started,
                    "skipped_datasets": f.skipped, "errors": f.errors,
                    "keep_rate": round(f.kept / f.judged, 3) if f.judged else None,
                    "next_run_ix": f.next_ix}
        for f in order}
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
    ap.add_argument("--quota", type=int,
                    help="run each family until this many of its rows are kept, instead of --reps "
                         "runs; each dataset passes its own oracle first")
    ap.add_argument("--block", type=int,
                    help="training block in tokens: a kept row must fit it (default 8192 with "
                         "--quota, no limit otherwise)")
    ap.add_argument("--max-runs", type=int,
                    help="with --quota: datasets per family at most (default 5x the quota)")
    ap.add_argument("--max-minutes", type=float,
                    help="with --quota: start no run after this many minutes; running ones finish. "
                         "Leave room for the longest run before the GPU window ends")
    ap.add_argument("--resume", action="store_true",
                    help="with --quota: append to --out and --fail-out, counting their rows toward "
                         "the quota and starting past their run indices")
    ap.add_argument("--out", default="", help="kept rows (meta.selection.kept)")
    ap.add_argument("--fail-out", default="",
                    help="every other run, for diagnosing a family: failed the oracle, or passed "
                         "but not kept (meta.selection.why_not). Never training data; a row over "
                         "the block keeps meta.verification.oracle_passed true")
    ap.add_argument("--report", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    tasks = [t for t in args.tasks.split(",") if t] or None

    if args.oracle_only:
        if args.quota:
            ap.error("--oracle-only takes --reps: quota mode gates each dataset as it runs")
        gate = oracle_gate(jobs_for(tasks, args.reps, args.run_offset))
        print(json.dumps(gate, indent=1))
        raise SystemExit(1 if gate["failed"] else 0)

    quota_mode = args.quota is not None
    if quota_mode:
        if args.quota < 1:
            ap.error("--quota must be at least 1")
        if not (args.out and args.fail_out):
            # --resume rebuilds the run indices from both files; without the failures it would
            # start a family on an index a failed run already used.
            ap.error("--quota needs --out and --fail-out")
        if not args.resume:
            for path in (args.out, args.fail_out):
                if os.path.exists(path) and os.path.getsize(path):
                    ap.error(f"{path} already has rows: pass --resume to continue from them, or "
                             f"choose another path")
    elif args.resume or args.max_runs or args.max_minutes:
        ap.error("--resume, --max-runs and --max-minutes go with --quota")
    block = args.block if args.block is not None else (8192 if quota_mode else None)

    sampling = dict(BASE_SAMPLING if args.thinking else TEACHER_SAMPLING)
    for key, value in (("temperature", args.temperature), ("top_p", args.top_p),
                       ("top_k", args.top_k)):
        if value is not None:
            sampling[key] = value

    prior = read_records(args.out) + read_records(args.fail_out) if args.resume else []
    mode = "a" if args.resume else "w"
    out_fh = open(args.out, mode) if args.out else None  # closed in the finally below
    fail_fh = open(args.fail_out, mode) if args.fail_out else None

    def _sink(rec) -> None:  # write+flush each kept trajectory immediately
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()

    def _fail_sink(rec) -> None:
        fail_fh.write(json.dumps(rec) + "\n")
        fail_fh.flush()

    common = dict(base_url=args.base_url, model=args.model, tasks=tasks, verbose=args.verbose,
                  sink=_sink if out_fh else None, fail_sink=_fail_sink if fail_fh else None,
                  run_offset=args.run_offset, workers=args.workers, thinking=args.thinking,
                  sampling=sampling, max_tokens=args.max_tokens, block=block)
    try:
        if quota_mode:
            records, report = generate_quota(
                quota=args.quota, max_runs=args.max_runs or 5 * args.quota,
                max_seconds=None if args.max_minutes is None else args.max_minutes * 60,
                prior=prior, **common)
        else:
            records, report = generate(reps=args.reps, **common)
    finally:
        if out_fh:
            out_fh.close()
        if fail_fh:
            fail_fh.close()
    report["params"] = {"model": args.model, "thinking": args.thinking, "sampling": sampling,
                        "max_tokens": args.max_tokens, "workers": args.workers,
                        "reps": None if quota_mode else args.reps, "quota": args.quota,
                        "block": block, "run_offset": args.run_offset, "resume": args.resume}
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
    print(f"kept: {report['emitted']}  not kept: {report['failed']}")
    print(f"by task: {report['by_task']}  statuses: {report['statuses']}  "
          f"not kept because: {report['why_not']}")
    if quota_mode:
        print(f"stop: {report['stop']} after {report['elapsed_s']}s")
        for task_id, fam in report["families"].items():
            print(f"  {task_id}: {fam['kept']}/{args.quota} kept, {fam['datasets_used']} datasets, "
                  f"keep rate {fam['keep_rate']}, next run index {fam['next_run_ix']}")
    for fr in report["fail_reasons"][:8]:
        print(f"  NOT KEPT [{fr['task']}/{fr['why_not']}] {fr['reason']}")


if __name__ == "__main__":
    main()
