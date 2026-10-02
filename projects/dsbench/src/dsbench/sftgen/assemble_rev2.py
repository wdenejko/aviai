"""Assemble Revision 2's training mixture (ADR-004 Revision 2, action item 9): every pool's rows
that passed their check, as thinking-on training records, picked to the table's token budgets.

WHY a new assembler: Gate 2's (`assemble.py`, kept so that mixture can be rebuilt) took finished
rows from fixed files. Revision 2's rows are the base's own replies, each kept only where its
pool's check passed. So assembly joins each reply to the prompt it answered, measures the row
with the server's own counts, and picks rows to the budgets of ADR-004's table.

Inputs:
- the single-turn pools, each as an items file and its checker's output, joined by id:
  - the SQL pool: `synsql verify`; the tool rows (`tool_fit`, `tool_decline`): `tool_rows verify`;
  - code (`opencoder_edu`, `swe_swiss`) and replay (`oasst1`, `aya`, `sciriff`, `gsm8k`):
    `replay_verify`;
  - Target A: `target_a_hints verify`, whose rows carry their training prompt. Its plain and
    prefilled rows (`targetA`, `targetA_recall`, ...) are one pool;
- Target C: its trajectories, which are records already (`ml_delivery_trajectories --quota`).

A kept reply becomes a record in the shape every sftgen record has: the prompt the base
answered, then one assistant message with its reasoning (`reasoning_content`, a prefill
included), its answer and any tool calls. `tokenize_masked.thinking_record` renders it with the
model's template, thinking on, and labels the assistant turn.

Per pool, in order:
1. **The check.** `check.ok`, or Target A's `check.kept`. GSM8K keeps its replies that reach the
   gold number, as decision 5 proposes; `--gsm8k finished` keeps every finished one instead.
   Target A's ClickHouse weekday and weekend rows drop a trace that states Sunday = 1 again after
   the convention (`check.doubts`), as decision 2 proposes; `--target-a-doubts keep` keeps it.
2. **The block.** A row is the prompt, the reply and the newline after its `<|im_end|>`, and the
   separator follows it in its block: `prompt_tokens + completion_tokens + 2 <= 8,192` on the
   server's counts. Target C's rows are counted as their selection counted them, from the last
   request. The generation's budget already stopped the replies that couldn't fit; this drops
   the ones that finished just past it. `build_masked_dataset` counts again, exactly, on the box.
3. **The gate** that `decontaminate.py --battery-items` runs: dsbench's rules, and a fifth of a
   battery item. It scans the whole record, reasoning, answer and tool calls included. The
   prompts were gated when they were built, but the replies are new text. Overlaps below the
   line are counted in the manifest.
4. **The budget.** A seeded shuffle, then rows until the pool's tokens reach its budget. A pool
   short of its budget gives all it has, and the shortfall is reported, never padded.

Budgets are ADR-004's table (10M tokens) times `--scale` (decision 1), with the shares within
code and replay that decision 6 proposes (`--share POOL=W`). A budget counts whole rows,
prompts included, because training time is paid per token. The manifest also gives each pool's
reply tokens, the part that is trained on.

Decision 7, Target C's failed calls: a turn whose call failed (`agentic.tools.is_error`: 141 of
the kept rows' 1,158 turns, 72,885 of their 378,639 reply tokens) is marked `"loss": False` by
default, as proposed. It stays in the row, so the fix that follows trains with its cause in view,
but it isn't trained on. `--target-c-failed-turns train` trains every turn, as the loop happened.

    uv run python -m dsbench.sftgen.assemble_rev2 --battery-items data/battery/items \\
        --verified ITEMS VERIFIED [--verified ITEMS VERIFIED ...] \\
        --trajectories data/sft/rev2_target_c/trajectories.jsonl \\
        --out data/sft/rev2_mixture.jsonl --report data/sft/rev2_mixture_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from dsbench.agentic.tools import is_error
from dsbench.sftgen.decontaminate import DenyList, _row_text, build_denylist, scan_text
from dsbench.sftgen.reasoning_pilot import latest_rows

SEED = 20261002
BLOCK = 8192
# ADR-004 Revision 2's table: tokens per line, whole rows, 10M in all.
BUCKET_TOKENS = {"target_a": 800_000, "sql": 1_000_000, "target_c": 1_000_000,
                 "tool_fit": 400_000, "tool_decline": 300_000, "code": 2_500_000,
                 "replay": 4_000_000}


@dataclass(frozen=True)
class Pool:
    name: str  # the rows' `pool`
    bucket: str  # the table's line
    share: float = 1.0  # of the line's tokens, normalised over the line (decision 6)


POOLS: tuple[Pool, ...] = (
    Pool("targetA", "target_a"), Pool("synsql", "sql"), Pool("target_c", "target_c"),
    Pool("tool_fit", "tool_fit"), Pool("tool_decline", "tool_decline"),
    Pool("opencoder_edu", "code", 0.75), Pool("swe_swiss", "code", 0.25),
    Pool("oasst1", "replay", 0.4), Pool("aya", "replay", 0.2), Pool("sciriff", "replay", 0.2),
    Pool("gsm8k", "replay", 0.2), Pool("flan_v2", "replay", 0.0),
)


@dataclass
class Row:
    record: dict
    tokens: int | None  # the whole row, on the server's counts
    reply: int  # the base's own tokens: what is trained on


def pool_of(row_pool: str) -> str:
    """The mixture pool of a row: Target A's placements (plain, recall, ...) are one."""
    return "targetA" if row_pool.startswith("targetA") else row_pool


def budget(pool: Pool, pools: tuple[Pool, ...], scale: float = 1.0) -> int:
    total = sum(p.share for p in pools if p.bucket == pool.bucket)
    if pool.share <= 0 or total <= 0:
        return 0
    return round(BUCKET_TOKENS[pool.bucket] * scale * pool.share / total)


def passed(rec: dict, gsm8k: str = "gold", target_a_doubts: str = "drop") -> bool:
    """Whether the row's checker kept it, with decision 5 applied to GSM8K and decision 2's open
    question to Target A's doubting traces."""
    check = rec.get("check") or {}
    if rec.get("pool") == "gsm8k" and gsm8k == "finished":
        return bool(check.get("finished"))
    if target_a_doubts == "drop" and check.get("doubts"):
        return False
    return bool(check["ok"] if "ok" in check else check.get("kept"))


def single_turn(item: dict, rec: dict) -> Row:
    """The training record of one kept reply: the prompt it answered, then the base's turn."""
    turn = {"role": "assistant", "content": rec.get("answer") or "",
            "reasoning_content": rec.get("reasoning") or ""}
    if rec.get("tool_calls"):
        turn["tool_calls"] = rec["tool_calls"]
    # Target A's training prompt can differ from the one the base saw (a hint left out of it)
    prompt = rec.get("train_messages") or item.get("train_messages") or item["messages"]
    counted = rec.get("prompt_tokens") is not None and rec.get("completion_tokens") is not None
    tokens = rec["prompt_tokens"] + rec["completion_tokens"] + 1 if counted else None
    record = {"messages": [*prompt, turn], "loss_mask_roles": ["assistant"],
              "meta": {"id": item["id"], "pool": item["pool"], "source": item.get("meta", {}),
                       "answer": {"by": "the base, thinking on", "sampling": rec.get("sampling"),
                                  "finish_reason": rec.get("finish_reason")},
                       "verification": rec.get("check"),
                       "tokens": {"row": tokens, "prompt": rec.get("prompt_tokens"),
                                  "reply": rec.get("completion_tokens")}}}
    if item.get("tools"):
        record["tools"] = item["tools"]
    return Row(record, tokens, rec.get("completion_tokens") or 0)


def failed_turns(messages: list[dict]) -> list[int]:
    """The indices of the assistant turns whose tool calls were answered by an error: any of the
    tool messages that follow the turn, for parallel calls."""
    failed = []
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            if is_error(messages[j].get("content") or ""):
                failed.append(i)
                break
            j += 1
    return failed


def trajectory(record: dict, mask_failed: bool = True) -> Row:
    """A Target C loop, as its selection measured it: the last request's tokens, and the newline
    after the last turn. With `mask_failed` (decision 7), the turns whose call failed carry
    `"loss": False`, and the reply tokens count only the turns that train. The selection's
    per-turn counts follow the assistant turns in order."""
    meta = record["meta"]
    served = (meta.get("selection") or {}).get("served_tokens")
    messages = record["messages"]
    masked = set(failed_turns(messages)) if mask_failed else set()
    if masked:
        messages = [{**m, "loss": False} if i in masked else m for i, m in enumerate(messages)]
    assistant = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    reply = sum(t.get("completion_tokens") or 0
                for i, t in zip(assistant, meta.get("turns") or [], strict=False)
                if i not in masked)
    record = {**record, "messages": messages,
              "meta": {**meta, "pool": "target_c", "masked_turns": sorted(masked)}}
    return Row(record, served + 1 if served is not None else None, reply)


def load_verified(items_path: Path, verified_path: Path, gsm8k: str = "gold",
                  target_a_doubts: str = "drop") -> tuple[list[Row], Counter]:
    """(the kept rows, the rows read per pool) of one checker's output."""
    items = {item["id"]: item for item in map(json.loads, items_path.open())}
    rows, read = [], Counter()
    for rec in latest_rows(verified_path).values():
        item = items.get(rec["id"])
        if item is None:
            raise SystemExit(f"{verified_path}: {rec['id']} is not in {items_path}")
        read[pool_of(item["pool"])] += 1
        if passed(rec, gsm8k, target_a_doubts):
            rows.append(single_turn(item, rec))
    return rows, read


def select(rows: list[Row], target: int, rng: random.Random) -> list[Row]:
    """A seeded shuffle, then rows until the target is reached (the last may cross it)."""
    order = rows[:]
    rng.shuffle(order)
    kept, tokens = [], 0
    for row in order:
        if tokens >= target:
            break
        kept.append(row)
        tokens += row.tokens or 0
    return kept


def assemble(rows: list[Row], read: Counter, deny: DenyList | None, *,
             pools: tuple[Pool, ...] = POOLS, scale: float = 1.0, seed: int = SEED,
             block: int = BLOCK) -> tuple[list[dict], dict]:
    """(the mixture's records, shuffled; the manifest's pool and total sections)."""
    by_pool: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_pool[pool_of(row.record["meta"]["pool"])].append(row)
    unknown = set(by_pool) - {p.name for p in pools}
    if unknown:
        raise SystemExit(f"rows from pools the mixture has no budget for: {sorted(unknown)}")

    mixture: list[dict] = []
    report = []
    for pool in pools:
        target = budget(pool, pools, scale)
        candidates, over, uncounted = [], 0, 0
        contaminated: Counter[str] = Counter()
        below_line = 0
        for row in by_pool[pool.name]:
            if row.tokens is None:
                uncounted += 1
                continue
            if row.tokens + 1 > block:
                over += 1
                continue
            if deny is not None:
                audit: list[dict] = []
                reason = scan_text(_row_text(row.record), deny, audit=audit)
                if reason:
                    contaminated[reason["rule"]] += 1
                    continue
                below_line += bool(audit)
            candidates.append(row)
        chosen = select(candidates, target, random.Random(f"{seed}:{pool.name}"))
        tokens = sum(row.tokens or 0 for row in chosen)
        licences = Counter(str(row.record["meta"].get("source", {}).get("licence")
                               or (row.record["meta"].get("provenance") or {}).get("licence"))
                           for row in chosen)
        report.append({
            "pool": pool.name, "bucket": pool.bucket, "share": pool.share, "budget": target,
            "read": read[pool.name], "kept_by_check": len(by_pool[pool.name]),
            "uncounted": uncounted, "over_block": over,
            "contaminated": dict(contaminated), "battery_below_line": below_line,
            "eligible": len(candidates), "rows": len(chosen), "tokens": tokens,
            "reply_tokens": sum(row.reply for row in chosen),
            "shortfall": max(0, target - tokens), "left_over_rows": len(candidates) - len(chosen),
            "licences": dict(licences),
            "study_only": sum(row.record["meta"].get("source", {}).get("redistributable")
                              is False for row in chosen),
        })
        mixture += [{**row.record, "meta": {**row.record["meta"], "mix_pool": pool.name,
                                            "mix_bucket": pool.bucket}} for row in chosen]
    random.Random(seed).shuffle(mixture)

    total = sum(entry["tokens"] for entry in report)
    buckets: dict[str, dict] = {}
    for entry in report:
        b = buckets.setdefault(entry["bucket"], {"budget": 0, "tokens": 0, "reply_tokens": 0,
                                                 "rows": 0})
        for key in ("budget", "tokens", "reply_tokens", "rows"):
            b[key] += entry[key]
    for b in buckets.values():
        b["pct_of_mixture"] = round(100 * b["tokens"] / total, 1) if total else 0.0
    return mixture, {"pools": report, "buckets": buckets,
                     "total": {"rows": len(mixture), "tokens": total,
                               "reply_tokens": sum(e["reply_tokens"] for e in report),
                               "budget": sum(e["budget"] for e in report)}}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--battery-items", required=True,
                    help="a battery run's items/ dir: the gate's rule 4")
    ap.add_argument("--verified", nargs=2, action="append", default=[], type=Path,
                    metavar=("ITEMS", "VERIFIED"), help="an items file and its checker's output")
    ap.add_argument("--trajectories", action="append", default=[], type=Path,
                    help="Target C's kept trajectories")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="all budgets times this (decision 1)")
    ap.add_argument("--share", action="append", default=[], metavar="POOL=W",
                    help="a pool's share of its line (decision 6), e.g. flan_v2=0.1")
    ap.add_argument("--gsm8k", choices=("gold", "finished"), default="gold",
                    help="decision 5: GSM8K's replies that reach the gold number, or every "
                         "finished one")
    ap.add_argument("--target-c-failed-turns", choices=("mask", "train"), default="mask",
                    help="decision 7: Target C's turns whose call failed")
    ap.add_argument("--target-a-doubts", choices=("drop", "keep"), default="drop",
                    help="decision 2: Target A's traces that state Sunday = 1 again after the "
                         "convention")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    shares = {name: float(w) for name, _, w in (s.partition("=") for s in args.share)}
    unknown = set(shares) - {p.name for p in POOLS}
    if unknown:
        raise SystemExit(f"unknown pools: {sorted(unknown)}")
    pools = tuple(replace(p, share=shares.get(p.name, p.share)) for p in POOLS)

    rows: list[Row] = []
    read: Counter = Counter()
    for items_path, verified_path in args.verified:
        kept, seen = load_verified(items_path, verified_path, args.gsm8k, args.target_a_doubts)
        rows += kept
        read += seen
    for path in args.trajectories:
        loops = [json.loads(line) for line in path.open() if line.strip()]
        read["target_c"] += len(loops)
        mask = args.target_c_failed_turns == "mask"
        rows += [trajectory(loop, mask) for loop in loops]  # kept by their own selection

    deny = build_denylist(battery_items=args.battery_items)
    mixture, result = assemble(rows, read, deny, pools=pools, scale=args.scale, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for record in mixture:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "params": {"seed": args.seed, "block": BLOCK, "scale": args.scale,
                   "shares": {p.name: p.share for p in pools}, "gsm8k": args.gsm8k,
                   "target_a_doubts": args.target_a_doubts,
                   "target_c_failed_turns": args.target_c_failed_turns,
                   "battery_items": args.battery_items},
        "inputs": [{"items": str(i), "verified": str(v), "verified_sha256": _sha256(v)}
                   for i, v in args.verified]
                  + [{"trajectories": str(t), "sha256": _sha256(t)} for t in args.trajectories],
        **result,
        "out": str(args.out), "out_sha256": _sha256(args.out),
    }
    args.report.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    for e in result["pools"]:
        print(f"{e['pool']:14s} budget {e['budget']:>9,}  tokens {e['tokens']:>9,}  "
              f"rows {e['rows']:5d}  eligible {e['eligible']:5d}  over block {e['over_block']:3d}"
              f"  contaminated {sum(e['contaminated'].values()):3d}  shortfall {e['shortfall']:,}")
    t = result["total"]
    print(f"{t['rows']} rows, {t['tokens']:,} tokens of {t['budget']:,} -> {args.out}")


if __name__ == "__main__":
    main()
