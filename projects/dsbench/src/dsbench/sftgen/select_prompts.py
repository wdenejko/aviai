"""Pick Revision 2's replay and code prompts from the acquired pools (ADR-004 Revision 2, item 6).

The base answers every prompt with thinking on (`reasoning_pilot.py generate` runs these items as
they are), so the pools' own answers are dropped here. A prompt is a row's messages up to and
including its last user turn. Earlier turns stay as the dataset wrote them: an oasst1 conversation
keeps its history, and the base writes one new turn.

A row is eligible when its prompt
- fits: at most `--max-prompt-tokens`, estimated as chars/3.5 like the rest of the pipeline. The
  default, 3,072, leaves 5,120 of an 8,192-token row for the reply. That covers the pilot's p90
  reply in every pool but opencoder's (10,195), whose prompts are short: its tail is the reply's
  doing, and its quota allows for it (below);
- shares nothing with the battery: no 13-gram of any item, and no short item whole. A row below
  rule 4's line would pass the assembly gate, but skipping it here costs a few rows in thousands;
- carries what its check needs (opencoder: its tests, each one `assert` line), and no tools: a
  tool conversation needs a real rollout, not one reply. An opencoder prompt also shows its first
  test (`EXAMPLE_TEST`);
- is not a prompt already taken, from this pool or an earlier one.

Each pool's quota is its share of its bucket's kept-row target (ADR-004's table: replay 1,600,
code and SWE 800), divided by the share of its rows expected to survive generation: rows over
8,192 tokens and replies that never finish are dropped. Shares within a bucket are normalised, so
giving FLAN v2 0.1 takes it from the others. The defaults are proposals (decisions 5 and 6).

Aya is sampled evenly across its languages, SciRIFF across its tasks, the other pools uniformly.
Each pool has its own seeded generator, so changing one pool's quota leaves the others' picks alone.

    uv run python -m dsbench.sftgen.select_prompts --battery-items data/battery/items \\
        --out data/sft/rev2_prompts.jsonl --report data/sft/rev2_prompts_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from dsbench.battery.contamination import BatteryIndex, build_index, shared_units
from dsbench.battery.items import load_items

SEED = 20260930
MAX_PROMPT_TOKENS = 3072
BUCKET_ROWS = {"replay": 1600, "code": 800}


@dataclass(frozen=True)
class Pool:
    name: str
    bucket: str  # replay | code
    file: str  # in the data zone
    share: float  # of its bucket's rows (proposed: ADR-004 decision 6)
    keep_rate: float = 0.95  # expected share of its rows kept after generation
    source: str | None = None  # meta.source: the Tulu mixture's subsets share one file
    balance: str | None = None  # meta key to sample evenly across
    requires: str | None = None  # meta key a row needs for its check


# FLAN v2 is 0 until decision 5. The pilot lost 6 of its 32 opencoder rows to length (verbose
# re-checking) and none in the other pools here; 0.95 is ADR-004's allowance for those.
POOLS: tuple[Pool, ...] = (
    Pool("oasst1", "replay", "breadth_tulu3.jsonl", 0.4, source="ai2-adapt-dev/oasst1_converted"),
    Pool("flan_v2", "replay", "breadth_tulu3.jsonl", 0.0,
         source="ai2-adapt-dev/flan_v2_converted"),
    Pool("aya", "replay", "breadth_tulu3_aya.jsonl", 0.2, balance="language"),
    Pool("sciriff", "replay", "breadth_tulu3_sciriff.jsonl", 0.2, balance="task"),
    Pool("gsm8k", "replay", "breadth_gsm8k.jsonl", 0.2),
    Pool("opencoder_edu", "code", "breadth_opencoder_edu_rev2.jsonl", 0.75, keep_rate=0.8,
         requires="testcase"),
    Pool("swe_swiss", "code", "breadth_swe_swiss_rev2.jsonl", 0.25),
)

# Kept out of an item's meta: sizing that no longer applies, the acquisition's bucket (the item
# has its own), and what moves to `verify`.
_META_DROP = ("approx_tokens", "bucket", "gold_answer", "testcase", "has_testcase", "entry_point")


# OpenCoder's tests call the function by the name the dataset's own answer gave it, and expect that
# answer's return format, but the instruction rarely says either: 19 of the 750 prompts selected on
# 2026-09-30 named the function, so a right answer under another name would fail every test. The
# prompt therefore shows the first test, as MBPP's prompts show theirs, and the check runs them all:
# the others stay unseen. Added before the length and battery checks, so they see it; duplicates
# are still found by the instruction alone.
EXAMPLE_TEST = "\n\nYour code should pass this test, and others like it:\n```python\n{test}\n```"


def with_example_test(prompt: list[dict], tests: list[str]) -> list[dict]:
    last = {**prompt[-1], "content": prompt[-1]["content"] + EXAMPLE_TEST.format(test=tests[0])}
    return [*prompt[:-1], last]


def prompt_messages(messages: list[dict]) -> list[dict]:
    """The messages up to and including the last user turn, each as {role, content}."""
    last = max(i for i, m in enumerate(messages) if m.get("role") == "user")
    return [{"role": m["role"], "content": m.get("content") or ""} for m in messages[: last + 1]]


def prompt_tokens(prompt: list[dict]) -> int:
    return round(sum(len(m["content"]) for m in prompt) / 3.5)


def _digest(prompt: list[dict]) -> str:
    return hashlib.sha1(json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def even_allocation(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split n across groups as evenly as their sizes allow: the smallest group first, each taking
    an equal share of what is left, or all it has. Less than n in total only if the groups run out.
    """
    alloc: dict[str, int] = {}
    order = sorted(sizes, key=lambda g: (sizes[g], g))
    left = n
    for i, group in enumerate(order):
        alloc[group] = min(sizes[group], left // (len(order) - i))
        left -= alloc[group]
    return alloc


def quota(pool: Pool, pools: tuple[Pool, ...], bucket_rows: dict[str, int]) -> int:
    total = sum(p.share for p in pools if p.bucket == pool.bucket)
    if pool.share <= 0 or total <= 0:
        return 0
    return math.ceil(bucket_rows[pool.bucket] * pool.share / total / pool.keep_rate)


def make_item(pool: Pool, row: dict, prompt: list[dict]) -> dict:
    meta = {k: v for k, v in row["meta"].items() if k not in _META_DROP}
    row_id = str(row["meta"].get("id") or _digest(prompt)[:12])
    item = {"id": f"{pool.name}:{row_id}", "pool": pool.name, "bucket": pool.bucket,
            "messages": prompt, "prompt_tokens_est": prompt_tokens(prompt), "meta": meta}
    if "gold_answer" in row["meta"]:
        item["verify"] = {"gold_answer": row["meta"]["gold_answer"]}
    if row["meta"].get("testcase"):
        item["verify"] = {"entry_point": row["meta"].get("entry_point"),
                          "tests": row["meta"]["testcase"]}
    return item


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing pool {path}: acquire it first (sftgen/breadth/acquire.py; the "
                         "commands are in data/sft/rev2_breadth_manifest.json)")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _pct(values: list[int], q: float) -> int:
    return sorted(values)[min(len(values) - 1, int(q * len(values)))] if values else 0


def select(data_zone: Path, index: BatteryIndex | None, *, pools: tuple[Pool, ...] = POOLS,
           bucket_rows: dict[str, int] | None = None, max_prompt_tokens: int = MAX_PROMPT_TOKENS,
           seed: int = SEED, exclude: frozenset[str] = frozenset()) -> tuple[list[dict], dict]:
    """(items, report). `exclude`: item ids already selected in an earlier run (a top-up)."""
    bucket_rows = bucket_rows or BUCKET_ROWS
    taken: set[str] = set()  # digests of the prompts earlier pools selected
    items: list[dict] = []
    report: list[dict] = []
    for pool in pools:
        want = quota(pool, pools, bucket_rows)
        dropped: Counter[str] = Counter()
        candidates: dict[str, dict] = {}  # prompt digest -> item: one per prompt in a pool
        rows = [r for r in _load(data_zone / pool.file)
                if pool.source is None or r["meta"].get("source") == pool.source]
        for row in rows:
            if row.get("tools") or any(m.get("tool_calls") or m.get("role") == "tool"
                                       for m in row["messages"]):
                dropped["tools"] += 1
                continue
            if pool.requires and not row["meta"].get(pool.requires):
                dropped[f"no {pool.requires}"] += 1
                continue
            if not all(test.strip().startswith("assert ") for test in row["meta"].get("testcase",
                                                                                     ())):
                # 39 of OpenCoder's 10,000 rows: a test over several lines (building a tree),
                # stored with its lines out of order, so it fails on the dataset's own answer
                dropped["tests not one assert a line"] += 1
                continue
            if not any(m.get("role") == "user" for m in row["messages"]):
                dropped["no user turn"] += 1
                continue
            prompt = prompt_messages(row["messages"])
            # the instruction's digest: the same instruction with other tests is a duplicate
            digest = _digest(prompt)
            if row["meta"].get("testcase"):
                prompt = with_example_test(prompt, row["meta"]["testcase"])
            if prompt_tokens(prompt) > max_prompt_tokens:
                dropped["long"] += 1
                continue
            if digest in taken or digest in candidates:
                dropped["duplicate"] += 1
                continue
            if index is not None and shared_units("\n".join(m["content"] for m in prompt), index):
                dropped["battery"] += 1
                continue
            item = make_item(pool, row, prompt)
            if item["id"] in exclude:
                dropped["selected before"] += 1
                continue
            candidates[digest] = item

        groups: dict[str, list[dict]] = defaultdict(list)
        for item in sorted(candidates.values(), key=lambda it: it["id"]):  # stable before sampling
            groups[str(item["meta"].get(pool.balance)) if pool.balance else "all"].append(item)
        alloc = even_allocation({g: len(v) for g, v in groups.items()}, want)
        rng = random.Random(f"{seed}:{pool.name}")
        chosen = [item for g in sorted(groups) for item in rng.sample(groups[g], alloc[g])]
        digest_of = {item["id"]: digest for digest, item in candidates.items()}
        taken |= {digest_of[item["id"]] for item in chosen}
        items += chosen

        tokens = [item["prompt_tokens_est"] for item in chosen]
        entry = {
            "pool": pool.name, "bucket": pool.bucket, "file": pool.file, "share": pool.share,
            "keep_rate": pool.keep_rate, "quota": want, "rows": len(rows),
            "eligible": len(candidates), "dropped": dict(dropped.most_common()),
            "selected": len(chosen), "shortfall": want - len(chosen),
            "with_check": sum("verify" in item for item in chosen),
            "prompt_tokens_est": {"median": _pct(tokens, 0.5), "p90": _pct(tokens, 0.9),
                                  "max": max(tokens, default=0)},
        }
        if pool.balance:
            entry["groups"] = {"available": len(groups), "selected": sum(v > 0 for v in
                                                                         alloc.values())}
            entry["by_" + pool.balance] = dict(sorted(Counter(
                str(item["meta"].get(pool.balance)) for item in chosen).items()))
        report.append(entry)
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):  # generation resumes by id: a repeated id would be skipped
        raise ValueError("two selected prompts share an item id")
    return items, {"pools": report}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pairs(values: list[str], kind: type) -> dict:
    out = {}
    for value in values:
        name, _, number = value.partition("=")
        out[name] = kind(number)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-zone", type=Path, default=Path("data/sft"))
    ap.add_argument("--battery-items", required=True,
                    help="a battery run's items/ dir: prompts sharing anything with it are skipped")
    ap.add_argument("--out", type=Path, required=True, help="the items, one JSON object a line")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    ap.add_argument("--replay-rows", type=int, default=BUCKET_ROWS["replay"])
    ap.add_argument("--code-rows", type=int, default=BUCKET_ROWS["code"])
    ap.add_argument("--share", action="append", default=[], metavar="POOL=W",
                    help="a pool's share of its bucket (repeatable), e.g. flan_v2=0.1")
    ap.add_argument("--keep-rate", action="append", default=[], metavar="POOL=R")
    ap.add_argument("--exclude", type=Path, help="items selected before: not picked again")
    args = ap.parse_args()

    shares, keep = _pairs(args.share, float), _pairs(args.keep_rate, float)
    unknown = (set(shares) | set(keep)) - {p.name for p in POOLS}
    if unknown:
        raise SystemExit(f"unknown pools: {sorted(unknown)}")
    pools = tuple(replace(p, share=shares.get(p.name, p.share),
                          keep_rate=keep.get(p.name, p.keep_rate)) for p in POOLS)
    items_dir = Path(args.battery_items)
    battery = [it for path in sorted(items_dir.glob("*.jsonl")) for it in load_items(path)]
    if not battery:  # a mistyped path must not pass as "no overlap"
        raise SystemExit(f"no battery items under {items_dir}")
    exclude = frozenset()
    if args.exclude:
        exclude = frozenset(json.loads(line)["id"] for line in args.exclude.open())
    bucket_rows = {"replay": args.replay_rows, "code": args.code_rows}

    items, report = select(args.data_zone, build_index(battery), pools=pools,
                           bucket_rows=bucket_rows, max_prompt_tokens=args.max_prompt_tokens,
                           seed=args.seed, exclude=exclude)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    for entry in report["pools"]:
        entry["file_sha256"] = _sha256(args.data_zone / entry["file"])
    report = {
        "params": {"seed": args.seed, "max_prompt_tokens": args.max_prompt_tokens,
                   "bucket_rows": bucket_rows, "battery_items": str(items_dir),
                   "battery_items_n": len(battery), "exclude": str(args.exclude or "")},
        "items": len(items), "out": str(args.out), "out_sha256": _sha256(args.out),
        "by_bucket": dict(Counter(item["bucket"] for item in items)),
        **report,
    }
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    for e in report["pools"]:
        print(f"{e['pool']:14s} quota {e['quota']:4d}  eligible {e['eligible']:6d}  "
              f"selected {e['selected']:4d}  shortfall {e['shortfall']:3d}  dropped {e['dropped']}")
    print(f"{len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
