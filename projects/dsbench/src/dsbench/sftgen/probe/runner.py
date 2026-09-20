"""Run the held-out probe through the SAME pi harness as dsbench, with a NEUTRAL system prompt.

This is the generalisation measurement (ADR-004): baseline the current base (Qwen3.6) here to record
the "before", then re-run the same command on the fine-tune for the "after". Because the probe lives
in its own package, it is never part of the eval's 23 problems and never trained on. The only
difference from the eval runner is the system prompt: the probe's domains are not aviation, so it
injects no warehouse schema — just the tool contract.

Run:  uv run --package dsbench python -m dsbench.sftgen.probe.runner \
          --provider dashi-qwen36 --model qwen36 --repeat 5 --label qwen36-probe
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path

from dsbench.agentic.loop import prepare_context
from dsbench.agentic.pi_runner import _RUNS, _render, run_pi_agent
from dsbench.agentic.schema import AgentResult
from dsbench.sftgen.probe.tasks import PROBE_TASKS

NEUTRAL_SYSTEM = """You are a senior data scientist working in a sandbox. You work through your \
shell (the `bash` tool). Two commands are on your PATH:

  run_sql "<ONE ClickHouse statement>"
      Runs one SQL statement against YOUR scratch database `{namespace}` (the DEFAULT), so \
`CREATE TABLE t ...` / `INSERT` land there. SELECT/SHOW/DESCRIBE print a table; DDL prints little. \
ClickHouse has no correlated subqueries — use JOINs.

  run_python   (reads Python 3 from STDIN, e.g.  run_python < script.py)
      Runs in a container with pandas, numpy, scipy, scikit-learn, clickhouse_connect. To reach \
ClickHouse use exactly:  client = clickhouse_connect.get_client(host='clickhouse', port=8123, \
username='avbench', password='avbench', database=os.environ['CLICKHOUSE_DB'])  (do NOT use \
host='localhost' inside the container). To WRITE a table, create it then insert a DataFrame: \
client.command('CREATE TABLE t (col Type, ...) ENGINE = MergeTree ORDER BY col'); \
client.insert_df('t', df). It defaults to your scratch DB.

Work step by step and produce EXACTLY the deliverable asked for. If the task asks you to BUILD a \
table, create it in your scratch database with the exact name and columns requested. If the task \
asks a QUESTION, your FINAL message must contain ONLY the answer — the bare value requested, \
nothing else. There is no "finish" tool: you are done when you stop."""


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the held-out generalisation probe via pi.")
    ap.add_argument("--provider", default="dashi-qwen36")
    ap.add_argument("--model", default="qwen36")
    ap.add_argument("--thinking", default="high")
    ap.add_argument("--label", default=None)
    ap.add_argument("--ids", default=None, help="comma-separated probe ids; default all")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--repeat", type=int, default=5, help="runs per problem (k)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-dir", default=str(_RUNS))
    args = ap.parse_args()

    ids = set(args.ids.split(",")) if args.ids else None
    tasks = [t for t in PROBE_TASKS if not ids or t.id in ids]
    if not tasks:
        raise SystemExit("no probe tasks matched --ids")

    label = args.label or dt.datetime.now().strftime("probe-%Y%m%d-%H%M%S")
    print(f"running {len(tasks)} PROBE task(s) via pi, thinking {args.thinking}, k={args.repeat}")
    results: list[AgentResult] = []
    for task in tasks:
        for k in range(args.repeat):
            try:
                ctx = prepare_context(task.id)
                if task.setup:
                    task.setup(ctx)
            except Exception as e:  # noqa: BLE001
                results.append(AgentResult(task.id, task.category, task.difficulty, False,
                                           "setup_error", reason=str(e)[:200]))
                print(f"  SETUP_ERROR   {task.id}: {e}")
                continue
            r = run_pi_agent(task, ctx, provider=args.provider, model=args.model,
                             thinking=args.thinking, timeout=args.timeout, verbose=args.verbose,
                             system_override=NEUTRAL_SYSTEM.format(namespace=ctx.namespace))
            results.append(r)
            mark = "PASS" if r.passed else f"FAIL[{r.status}]"
            tag = f" [{k + 1}/{args.repeat}]" if args.repeat > 1 else ""
            extra = f"  {r.reason[:80]}" if r.reason and not r.passed else ""
            print(f"  {mark:16} {task.id}{tag} ({r.steps} steps, {r.tool_calls} tools, "
                  f"{r.latency_s}s){extra}")

    meta = {"label": label, "harness": "pi-probe", "model": args.model, "provider": args.provider,
            "thinking": args.thinking, "repeat": args.repeat,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "n": len(results)}
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
