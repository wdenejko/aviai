"""Bring the owner's agentic suite (dsbench v2 via the pi harness) into the battery.

The pi harness sends its own requests and knows nothing about per-request LoRA, so for this suite
the state is set on the SERVER: `set-scale` posts the adapter scale to llama-server's
/lora-adapters and reads it back. Requests that name no scale then get that one. The window sets
the global scale to 0 when the server starts (`--lora-init-without-apply` alone leaves it at 1.0),
and the pi runner sets it back to 0 when it finishes, so a forgotten switch measures the base,
never the adapter. The public benchmarks name their state per request, so they are unaffected.

`convert` turns one pi run (reports/agentic-runs/<stem>.json) into RUN/scores/dsbench.<state>.jsonl
with one row per problem. A problem passes if most of its k runs pass. That makes the pairing
per problem (the unit the suite was designed around), and it is conservative: a single lucky run
can't flip a problem.

    python -m dsbench.battery.dsbench_suite set-scale --base-url http://127.0.0.1:8093 --scale 1
    python -m dsbench.battery.dsbench_suite convert --pi-run R.json --state adapter --run-dir RUN
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import httpx

from dsbench.battery.items import write_jsonl


def set_scale(base_url: str, scale: float) -> list[dict]:
    with httpx.Client(base_url=base_url, timeout=60) as client:
        client.post("/lora-adapters", json=[{"id": 0, "scale": scale}]).raise_for_status()
        adapters = client.get("/lora-adapters").json()
    if [a["scale"] for a in adapters] != [scale]:
        raise RuntimeError(f"server reports {adapters}, wanted scale {scale}")
    return adapters


def convert(pi_run: Path, state: str, run_dir: Path) -> list[dict]:
    payload = json.loads(pi_run.read_text())
    runs: dict[str, list[dict]] = defaultdict(list)
    for r in payload["results"]:
        runs[r["id"]].append(r)
    rows = []
    for pid, rs in sorted(runs.items()):
        n_pass = sum(bool(r["passed"]) for r in rs)
        rows.append({"bench": "dsbench", "id": pid, "state": state,
                     "passed": n_pass * 2 > len(rs), "status": f"{n_pass}/{len(rs)} runs",
                     "detail": "; ".join(sorted({r["status"] for r in rs})),
                     "extra": {"runs_passed": n_pass, "runs": len(rs),
                               "category": rs[0]["category"]}})
    write_jsonl(run_dir / "scores" / f"dsbench.{state}.jsonl", rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set-scale")
    s.add_argument("--base-url", default="http://127.0.0.1:8093")
    s.add_argument("--scale", type=float, required=True)
    c = sub.add_parser("convert")
    c.add_argument("--pi-run", type=Path, required=True)
    c.add_argument("--state", required=True, choices=["base", "adapter", "base_rep"])
    c.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "set-scale":
        print(json.dumps(set_scale(args.base_url, args.scale)))
    else:
        rows = convert(args.pi_run, args.state, args.run_dir)
        print(f"dsbench {args.state}: {sum(r['passed'] for r in rows)}/{len(rows)} problems pass "
              f"by majority")


if __name__ == "__main__":
    main()
