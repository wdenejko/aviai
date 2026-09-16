"""dsbench-agent-run: run the tool-calling agent on the aviation sandbox and grade final state.

  uv run --package dsbench dsbench-agent-run --ids da_hub_delay --verbose
  uv run --package dsbench dsbench-agent-run --category de --label ornith-de

Thinking is OFF by default (memory reference-fork-thinking-eval); pass --think to turn it on.
Each run writes reports/agentic-runs/<ts>-<label>.{json,md}; the JSON keeps the full trajectory.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path

from dsbench.agentic.loader import load_problems
from dsbench.agentic.loop import prepare_context, run_agent
from dsbench.agentic.schema import AgentResult

_RUNS = Path(__file__).resolve().parents[3] / "reports" / "agentic-runs"


def _render(meta: dict, results: list[AgentResult]) -> str:
    passed = sum(r.passed for r in results)
    n = len(results)
    think = "on" if meta.get("think") else "off"
    lines = [
        f"# dsbench agentic run: {meta.get('label')}",
        "",
        f"- model: `{meta.get('model')}`  endpoint: `{meta.get('base_url')}`  thinking: {think}",
        f"- when: {meta.get('timestamp')}",
        f"- score: **{passed}/{n}**" + (f" ({100 * passed / n:.0f}%)" if n else ""),
        "",
        "| id | category | difficulty | status | tool calls | steps | latency | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r.id):
        note = "" if r.passed else r.reason.replace("|", "\\|")[:90]
        lines.append(
            f"| {r.id} | {r.category} | {r.difficulty} | {r.status} | {r.tool_calls} | "
            f"{r.steps} | {r.latency_s}s | {note} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the agentic aviation benchmark against a model.")
    ap.add_argument("--base-url", default="http://localhost:18080")
    ap.add_argument("--model", default="ornith")
    ap.add_argument("--label", default=None)
    ap.add_argument("--category", choices=["de", "da", "ds"], default=None)
    ap.add_argument("--ids", default=None, help="comma-separated problem ids")
    ap.add_argument("--think", action="store_true", help="enable model thinking (default: off)")
    ap.add_argument("--verbose", action="store_true", help="print each step's tool calls")
    ap.add_argument("--out-dir", default=str(_RUNS))
    args = ap.parse_args()

    ids = set(args.ids.split(",")) if args.ids else None
    problems = load_problems(args.category, ids)
    if not problems:
        raise SystemExit("no agentic problems matched the filters")

    label = args.label or dt.datetime.now().strftime("run-%Y%m%d-%H%M%S")
    think = "on" if args.think else "off"
    print(f"running {len(problems)} agentic problem(s), thinking {think} ...")
    results: list[AgentResult] = []
    for p in problems:
        try:
            ctx = prepare_context(p.id)
            if p.setup:
                p.setup(ctx)
        except Exception as e:  # noqa: BLE001
            results.append(AgentResult(p.id, p.category, p.difficulty, False, "setup_error",
                                       reason=str(e)[:200]))
            print(f"  SETUP_ERROR   {p.id}: {e}")
            continue
        r = run_agent(p, ctx, base_url=args.base_url, model=args.model,
                      no_think=not args.think, verbose=args.verbose)
        results.append(r)
        mark = "PASS" if r.passed else f"FAIL[{r.status}]"
        extra = f"  {r.reason[:90]}" if r.reason and not r.passed else ""
        stats = f"{r.tool_calls} calls, {r.steps} steps, {r.latency_s}s"
        print(f"  {mark:14} {p.id}  ({stats}){extra}")

    meta = {
        "label": label, "model": args.model, "base_url": args.base_url, "think": args.think,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "n": len(results),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{label}"
    payload = {"meta": meta, "results": [dataclasses.asdict(r) for r in results]}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str))
    md = _render(meta, results)
    (out_dir / f"{stem}.md").write_text(md)
    print("\n" + md)
    npass = sum(r.passed for r in results)
    print(f"score: {npass}/{len(results)}  (wrote {out_dir / f'{stem}.json'})")


if __name__ == "__main__":
    main()
