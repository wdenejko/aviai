"""Score each generation pass as soon as it is complete, from a loop that runs beside the window.

A pass is complete when its generation file holds a row for every item (a row that still carries
an error after the retries is complete too: it scores as `no_generation`). Scoring a pass that is
still being written would record its missing items as failures, so incomplete passes are skipped.
A score file older than its generation file is redone (a resumed pass).

    python -m dsbench.battery.autoscore --run-dir RUN [--hours 16]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from dsbench.battery.items import by_id, read_jsonl


def complete_passes(run_dir: Path) -> list[tuple[str, str]]:
    todo = []
    for gen in sorted((run_dir / "gen").glob("*.jsonl")):
        bench, state = gen.stem.rsplit(".", 1)
        items = run_dir / "items" / f"{bench}.jsonl"
        if not items.exists():
            continue
        want = {row["id"] for row in read_jsonl(items)}
        if set(by_id(read_jsonl(gen))) < want:
            continue
        score = run_dir / "scores" / f"{bench}.{state}.jsonl"
        if not score.exists() or score.stat().st_mtime < gen.stat().st_mtime:
            todo.append((bench, state))
    return todo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--hours", type=float, default=16.0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    deadline = time.time() + 3600 * args.hours
    while time.time() < deadline:
        for bench, state in complete_passes(args.run_dir):
            print(f"[autoscore] {time.strftime('%H:%M:%S')} scoring {bench} {state}", flush=True)
            subprocess.run([sys.executable, "-m", "dsbench.battery.score", "--run-dir",
                            str(args.run_dir), "--bench", bench, "--states", state,
                            "--workers", str(args.workers)], check=False)
        time.sleep(60)


if __name__ == "__main__":
    main()
