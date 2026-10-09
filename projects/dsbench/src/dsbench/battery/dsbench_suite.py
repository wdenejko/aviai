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

A server with two adapters (battery_server.sh with LORA2, to run two adapters against each other in
one window) takes one scale per adapter, in the order they were loaded: `set-scale --scales 0,1`
serves the second alone.

`set-scale` also empties every slot's prompt cache. pi's requests name no adapter, so the server
never notices that the global scale changed under a slot's cache: the next request could reuse a
prefix computed at the old scales (llama-server drops a slot's cache only for a request that names
other scales than the slot's previous one). The server also keeps a RAM copy of prompts, which a
slot erase doesn't reach, so battery_server.sh turns that copy off (`--cache-ram 0`). Until
2026-10-09 the probe windows had neither guard. In the adapter's step, two probe problems started
all their runs from about 1,250 prompt tokens the base had computed, loaded from the RAM copy
(patches/README.md).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import httpx

from dsbench.battery.generate import erase_slots
from dsbench.battery.items import write_jsonl


def set_scale(base_url: str, scale: float | list[float]) -> list[dict]:
    """Set the server's global adapter scales, read them back, and empty every slot's cache. One
    scale is for a server that loaded one adapter; a list sets every adapter of a server that
    loaded several, by id. The read must list exactly these scales, so a list too short for the
    server is refused, not padded. The erase needs the server's --slot-save-path
    (battery_server.sh sets it); without it the switch fails rather than run on stale caches."""
    scales = list(scale) if isinstance(scale, list) else [scale]
    with httpx.Client(base_url=base_url, timeout=60) as client:
        client.post("/lora-adapters",
                    json=[{"id": i, "scale": s} for i, s in enumerate(scales)]).raise_for_status()
        adapters = client.get("/lora-adapters").json()
    if [a["scale"] for a in adapters] != scales:
        raise RuntimeError(f"server reports {adapters}, wanted scales {scales}")
    erase_slots(base_url)
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
    one = s.add_mutually_exclusive_group(required=True)
    one.add_argument("--scale", type=float, help="the adapter's scale, on a one-adapter server")
    one.add_argument("--scales", help="one scale per adapter, by id: 0,1 serves the second alone")
    c = sub.add_parser("convert")
    c.add_argument("--pi-run", type=Path, required=True)
    c.add_argument("--state", required=True, choices=["base", "adapter", "base_rep"])
    c.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "set-scale":
        scale = ([float(x) for x in args.scales.split(",")] if args.scales is not None
                 else args.scale)
        print(json.dumps(set_scale(args.base_url, scale)))
    else:
        rows = convert(args.pi_run, args.state, args.run_dir)
        print(f"dsbench {args.state}: {sum(r['passed'] for r in rows)}/{len(rows)} problems pass "
              f"by majority")


if __name__ == "__main__":
    main()
