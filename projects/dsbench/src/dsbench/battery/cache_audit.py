"""Did a pi window's runs reuse prompt KV computed at another adapter state? Read it off the log.

pi names no adapter, so a window serves it a state by setting the server's global scales between
steps (dsbench_suite set-scale). llama-server reuses cached prompts by tokens alone. It drops a
slot's cache only for a request that names other scales than the slot's previous request, and it
loads prompts from its RAM copy (`--cache-ram`) without asking which scales computed them. Until
2026-10-09 the windows had the RAM copy on and didn't erase the slots at a switch. In both probe
windows (Revision 2's and 2.1's), the adapter's runs of two probe problems all started from about
1,250 prompt tokens the base had computed in the step before. Since then battery_server.sh runs
with `--cache-ram 0` and set-scale erases every slot.

The server log says, per request, how many prompt tokens it computed; the prompt's length is the
slot's token count at release minus the tokens generated, plus one (the last one sampled is never
decoded). The audit splits pi's requests into runs: a run's prompts grow, since each turn adds the
tool call and its result, and a new run's first prompt (system and task) is no longer than the
last request's. It matches the runs, in order, to the run files' results. Per step (run file) and
problem, it lists the cached tokens of each run's first request. A problem's runs 2 to k reuse run
1's prefix (the same state); its first run in a step has no run of its own to reuse.

A window is clean when the server ran without the RAM copy (its log warns that idle-slot caching is
off without it) and each step's first request took nothing from cache (the erase emptied every
slot). Then nothing cached at another step's scales could be reached. A later problem's first run
may still reuse the step's own system-prompt prefix, which is the same state.

    python -m dsbench.battery.cache_audit --server-log ~/benchlab/logs/battery-server-STAMP.log \\
        --runs reports/agentic-runs/<stamp>-<label>-....json ...   (in the order they ran)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

RAM_OFF = "--cache-idle-slots requires --cache-ram, disabling"
_LAUNCH = re.compile(r"launch_slot_: id\s+\d+ \| task (\d+)")
_PROMPT = re.compile(r"task (\d+) \| prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens")
_GEN = re.compile(r"task (\d+) \|\s+eval time =\s+[\d.]+ ms /\s+(\d+) tokens")
_END = re.compile(r"task (\d+) \| stop processing: n_tokens = (\d+)")


@dataclass
class Request:
    prompt: int  # the prompt's tokens
    cached: int  # of them, taken from cache rather than computed


def server_requests(log: Path) -> tuple[list[Request], bool]:
    """The last server in the log (a window's parity check runs a bare-base server first): its
    finished requests in start order, and whether it ran without the RAM copy of prompts."""
    lines = log.read_text(errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if "load_model: loading" in line]
    lines = lines[starts[-1]:] if starts else lines
    order: list[int] = []
    seen: dict[int, dict[str, int]] = {}
    for line in lines:
        if m := _LAUNCH.search(line):
            order.append(int(m[1]))
            seen[int(m[1])] = {}
        for key, pattern in (("computed", _PROMPT), ("gen", _GEN), ("end", _END)):
            if (m := pattern.search(line)) and int(m[1]) in seen:
                seen[int(m[1])][key] = int(m[2])
    requests = []
    for task in order:
        t = seen[task]
        if {"computed", "gen", "end"} <= t.keys():
            prompt = t["end"] - t["gen"] + 1
            requests.append(Request(prompt, prompt - t["computed"]))
    return requests, any(RAM_OFF in line for line in lines)


def pi_runs(requests: list[Request], min_prompt: int = 1000) -> list[list[Request]]:
    """pi's requests, split into runs. Those before the first prompt over `min_prompt` tokens are
    the parity check's (8 one-line prompts; pi's system prompt alone is longer).

    pi also shortens a run's prompt itself: past its context limit it compacts the conversation
    into a summary (the Gate-2 battery's window: 115,089 tokens, then 12,098). A run's first prompt
    is its system prompt and task, never over twice the median of the runs' first prompts, so a
    shorter prompt that is longer than that continues its run."""
    first = next((i for i, r in enumerate(requests) if r.prompt > min_prompt), len(requests))
    runs: list[list[Request]] = []
    for r in requests[first:]:
        if not runs or r.prompt <= runs[-1][-1].prompt:
            runs.append([r])
        else:
            runs[-1].append(r)
    if not runs:
        return runs
    starts = sorted(run[0].prompt for run in runs)
    limit = 2 * starts[len(starts) // 2]
    merged = [runs[0]]
    for run in runs[1:]:
        if run[0].prompt > limit:
            merged[-1].extend(run)  # a compaction, not a new run
        else:
            merged.append(run)
    return merged


def audit(log: Path, run_files: list[Path]) -> dict:
    requests, ram_off = server_requests(log)
    runs = pi_runs(requests)
    labels = [(f.stem, r["id"]) for f in run_files for r in json.loads(f.read_text())["results"]]
    if len(runs) != len(labels):
        raise ValueError(f"the log has {len(runs)} pi runs, the run files {len(labels)}: they "
                         "don't match (a run file missing, or one from another window)")
    steps: dict[str, dict[str, list[int]]] = {}
    for (stem, pid), run in zip(labels, runs, strict=True):
        steps.setdefault(stem, {}).setdefault(pid, []).append(run[0].cached)
    step_starts = {stem: next(iter(problems.values()))[0] for stem, problems in steps.items()}
    return {
        "ram_cache_off": ram_off,
        "runs": len(runs),
        "step_first_request_cached": step_starts,
        "first_runs_from_cache": [{"step": stem, "problem": pid, "cached": cached[0]}
                                  for stem, problems in steps.items()
                                  for pid, cached in problems.items() if cached[0] > 0],
        "steps": steps,
        "clean": ram_off and all(c == 0 for c in step_starts.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server-log", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path, nargs="+",
                    help="the window's pi run files, in the order they ran")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = audit(args.server_log, args.runs)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
    print(f"RAM copy of prompts off: {result['ram_cache_off']}; {result['runs']} runs")
    for step, cached in result["step_first_request_cached"].items():
        print(f"  {step}: its first request took {cached} tokens from cache")
    for f in result["first_runs_from_cache"]:
        print(f"  {f['step']} {f['problem']}: its first run took {f['cached']} tokens from cache")
    print("clean" if result["clean"] else "NOT CLEAN: a step could reach another state's KV")
    raise SystemExit(0 if result["clean"] else 1)


if __name__ == "__main__":
    main()
