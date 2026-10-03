"""The full battery with thinking on: ADR-004 decision 9, and action item 8's last point.

Gate 2 ran the battery's pinned items greedy, with thinking off, each under its own benchmark's
budget. Revision 2 is served with thinking on, so the battery runs again that way: the base once,
since every candidate compares with it, then the candidate the mini-battery passes. This writes
the items:
- **Every item asks for thinking within pi's reply cap** (`mini.thinking`): 12,288 tokens, sampled
  with Qwen's settings for thinking mode, with the same seed for an item in every state but the A/A
  pass (`generate.py`).
- **Stop strings are dropped** (`mini.thinking`): llama-server matches them against the reasoning
  too, so DS-1000's `</code>` would end a reply whose reasoning names the tag.
- **Each benchmark's items in a seeded random order.** The first N items are then a random sample:
  a window can answer only them (a plan step `bench:state:N`), which is how the benchmarks the
  mini-battery's calibration didn't run (MMLU-Pro, GPQA, LiveCodeBench, DS-1000) get measured
  before the rest is planned. A later step resumes the pass, and so does a window stopped by its
  time limit; either way the answered items stay a random sample.

`report.py` compares the states, with the mini-battery's rules for thinking-on passes: a reply whose
reasoning never closed fails whatever its scorer says, and a pass counts only the items it answered.

    python -m dsbench.battery.full --items PINNED_ITEMS --out RUN/items
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from dsbench.battery.generate import REP_SEED_OFFSET, THINKING_SAMPLING
from dsbench.battery.items import load_items, save_items
from dsbench.battery.mini import BUDGET, thinking

SEED = 20261002
# The pinned battery (prepare.py); the four the calibration didn't measure come first
BENCHES = ("mmlu_pro", "gpqa", "lcb", "ds1000", "ifeval", "bfcl", "bird", "humaneval_plus")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(items_dir: Path, out_dir: Path, benches: tuple[str, ...] = BENCHES,
            seed: int = SEED) -> dict:
    """Write every pinned item of `benches`, thinking on, in a seeded order per benchmark, and the
    manifest `report.py` reads (`manifest.json`)."""
    manifest: dict = {
        "thinking": {"budget": BUDGET, "sampling": THINKING_SAMPLING, "order_seed": seed,
                     "seeds": "per item, sha256(bench:id); base_rep adds "
                              f"{REP_SEED_OFFSET['base_rep']}",
                     "stop_strings": "dropped: llama-server matches them against the reasoning"},
        "benches": {}}
    for bench in benches:
        items = load_items(items_dir / f"{bench}.jsonl")
        random.Random(f"{seed}:{bench}").shuffle(items)
        path = out_dir / f"{bench}.jsonl"
        save_items(path, [thinking(item) for item in items])
        manifest["benches"][bench] = {"n": len(items), "sha256": _sha256(path),
                                      "stops_dropped": sum("stop" in it.gen for it in items)}
    source = items_dir / "manifest.json"
    if source.exists():
        manifest["source_manifest_sha256"] = _sha256(source)
        manifest["source"] = json.loads(source.read_text())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", required=True, type=Path, help="the battery's pinned items")
    ap.add_argument("--out", required=True, type=Path, help="RUN/items")
    ap.add_argument("--bench", default=",".join(BENCHES))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    manifest = prepare(args.items, args.out, tuple(args.bench.split(",")), args.seed)
    print(json.dumps({b: e["n"] for b, e in manifest["benches"].items()}, indent=1))


if __name__ == "__main__":
    main()
