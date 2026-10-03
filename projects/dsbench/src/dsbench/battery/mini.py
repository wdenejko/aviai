"""The thinking-on mini-battery: Gate 2's four failures, checked on a checkpoint against the base.

ADR-004 Revision 2 (Validation; action item 8). Gate 2's adapter failed four benchmarks, each
through a habit it had learned (reports/gate-evals/20260924-gate2-battery.md):
- IFEval: shorter replies;
- BFCL irrelevance: a tool call where none fits;
- BIRD: SQL that SQLite can't run (176 errors against the base's 47);
- HumanEval+: reasoning leaking into the reply.

The retrain is served with thinking on, so its checkpoints are checked that way:
- `prepare` writes subsets of the battery's pinned items (`prepare.py`), drawn with a fixed seed:
  - IFEval: 200 of 541;
  - BFCL: 120 of the 240 irrelevance items;
  - BIRD: 150 of 1,534, in proportion to its difficulty levels;
  - HumanEval+: all 164.

  Every item asks for thinking and gets one budget, 12,288 tokens, pi's reply cap: a reply that
  can't finish there fails, as it would when served. `generate.py` samples thinking items, with
  the same seed per item in every state but the A/A pass.
- `summary` compares a state with the base, item by item:
  - each benchmark's passes;
  - BIRD's SQL that SQLite can't run;
  - replies whose reasoning never closed (`unclosed`), and replies cut at the budget;
  - a second reasoning block opened inside the reply;
  - reasoning length: the medians, and the per-item ratio to the base. The ratio is a geometric
    mean over items where both closed their reasoning; with it comes a sign test of shorter
    against longer.

**Flags.**
- A flag is a significant change for the worse: an exact McNemar test at 5%, or the sign test
  for length, whose ratio must also fall below `BREVITY_FLOOR`.
- The A/A pass (`base_rep`, other seeds) gets the same comparison. A flag there means the noise
  is larger than the test assumes.

**Sizing.** The subsets are sized for the habits, which moved far past their noise in Gate 2,
not for small changes in pass rates:
- BFCL irrelevance fell 22 points;
- BIRD's errors nearly quadrupled;
- IFEval's 4.6 points needed all 541 items to reach p = 0.014. Here the brevity check carries the
  same habit.

The full battery, re-baselined with thinking on, stays the final check.

    python -m dsbench.battery.mini prepare --items ITEMS --out RUN/items
    python -m dsbench.battery.mini summary --run-dir RUN --state adapter --out-json J --out-md M
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from dsbench.battery import stats
from dsbench.battery.generate import REP_SEED_OFFSET, THINKING_SAMPLING
from dsbench.battery.items import Item, by_id, load_items, read_jsonl, save_items

SEED = 20261002
BUDGET = 12288  # pi's reply cap (ADR-004): what a served reply gets
SIZES = {"ifeval": 200, "bfcl": 120, "bird": 150, "humaneval_plus": None}  # None: every item
SELECT = {"bfcl": lambda item: item.meta.get("category") == "irrelevance"}
STRATUM = {"bird": "difficulty"}
# Gate 2: MMLU-Pro's reasoning ran 0.46 of the base's per item; IFEval's replies, 0.82 at the median
BREVITY_FLOOR = 0.8
# Per-item conditions that are failures, each compared with the base like a pass rate.
BAD = ("unclosed", "truncated", "second_think", "sql_error")


def proportional(items: list[Item], key: str, n: int, seed: int) -> list[Item]:
    """`n` items, each stratum in proportion to its size (largest remainder), order kept."""
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        groups[str(item.meta.get(key))].append(item)
    quotas = {g: n * len(members) / len(items) for g, members in groups.items()}
    counts = {g: int(q) for g, q in quotas.items()}
    for g in sorted(quotas, key=lambda g: (counts[g] - quotas[g], g))[: n - sum(counts.values())]:
        counts[g] += 1
    rng = random.Random(seed)
    chosen = {item.id for g in sorted(groups) for item in rng.sample(groups[g], counts[g])}
    return [item for item in items if item.id in chosen]


def thinking(item: Item) -> Item:
    """The item asking for thinking within the budget, and without stop strings.

    llama-server matches stop strings against everything it generates, the reasoning included
    (`process_token` in tools/server/server-context.cpp), so DS-1000's `</code>` would end a reply
    whose reasoning names the tag. DS-1000's extraction cuts at the same markers itself
    (`extract.ds1000_code`). The mini-battery's benchmarks have no stop strings.
    """
    gen = {key: value for key, value in item.gen.items() if key != "stop"}
    return Item(item.bench, item.id, item.messages,
                {**gen, "max_tokens": BUDGET, "thinking": True}, item.ref, item.meta)


def subset(bench: str, items: list[Item], seed: int = SEED) -> list[Item]:
    """The mini-battery's items for one benchmark, each asking for thinking within the budget."""
    pool = [item for item in items if SELECT.get(bench, lambda _: True)(item)]
    n = SIZES[bench]
    if n is None or n >= len(pool):
        chosen = pool
    elif bench in STRATUM:
        chosen = proportional(pool, STRATUM[bench], n, seed)
    else:
        ids = {item.id for item in random.Random(seed).sample(pool, n)}
        chosen = [item for item in pool if item.id in ids]
    return [thinking(item) for item in chosen]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(items_dir: Path, out_dir: Path, seed: int = SEED) -> dict:
    """Write the subsets and a manifest that records how they were drawn."""
    manifest: dict = {"seed": seed, "budget": BUDGET, "sampling": THINKING_SAMPLING,
                      "seeds": "per item, sha256(bench:id); base_rep adds "
                               f"{REP_SEED_OFFSET['base_rep']}", "benches": {}}
    source = items_dir / "manifest.json"
    if source.exists():
        manifest["source_manifest_sha256"] = _sha256(source)
    for bench in SIZES:
        items = subset(bench, load_items(items_dir / f"{bench}.jsonl"), seed)
        path = out_dir / f"{bench}.jsonl"
        save_items(path, items)
        entry = {"n": len(items), "sha256": _sha256(path)}
        if bench in STRATUM:
            entry["strata"] = dict(sorted(_count(str(it.meta.get(STRATUM[bench]))
                                                 for it in items).items()))
        manifest["benches"][bench] = entry
    (out_dir / "mini_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        counts[v] += 1
    return dict(counts)


def _rows(run_dir: Path, kind: str, bench: str, state: str) -> dict[str, dict] | None:
    path = run_dir / kind / f"{bench}.{state}.jsonl"
    return by_id(read_jsonl(path)) if path.exists() else None


def conditions(bench: str, score: dict, gen: dict) -> dict[str, bool]:
    """One item's outcome in one state: passed, and each failure condition.

    A pass needs the scorer's pass and closed reasoning. A reply that never closes gives the user
    nothing, yet BFCL scores irrelevance as "no call decoded", and some IFEval instructions hold on
    empty text ("no commas"). Counted as passes, a checkpoint that loops more would score better.
    The base itself loops: 7 of BFCL's 120 irrelevance items, on the calibration's first pass.
    """
    return {"passed": bool(score.get("passed")) and not gen.get("unclosed"),
            "unclosed": bool(gen.get("unclosed")),
            "truncated": gen.get("finish_reason") == "length",
            "second_think": "<think>" in (gen.get("content") or ""),
            # SQLite's own error, not a wrong answer: the habit Gate 2 learned from Gretel
            "sql_error": bench == "bird" and score.get("status") == "error"}


def compare(name: str, base: dict[str, bool], other: dict[str, bool], good: bool) -> dict:
    """Paired comparison of one condition; `good` says whether True is the better outcome."""
    flip = (lambda v: v) if good else (lambda v: not v)
    p = stats.paired(name, {i: flip(v) for i, v in base.items()},
                     {i: flip(v) for i, v in other.items()})
    share = (lambda acc: acc) if good else (lambda acc: 100 - acc)
    return {"base": round(share(p.acc_base), 2), "other": round(share(p.acc_adapter), 2),
            "worse": p.losses, "better": p.gains, "p": p.p,
            "flag": p.p < stats.ALPHA and p.delta < 0}


def brevity(base_gen: dict[str, dict], other_gen: dict[str, dict], keep: set[str]) -> dict:
    """Reasoning length against the base's, over items where both closed their reasoning."""
    both = [i for i in sorted(keep) if i in base_gen and i in other_gen
            and base_gen[i].get("reasoning_chars") and other_gen[i].get("reasoning_chars")
            and not base_gen[i].get("unclosed") and not other_gen[i].get("unclosed")]
    ratios = [other_gen[i]["reasoning_chars"] / base_gen[i]["reasoning_chars"] for i in both]
    if not ratios:
        return {"n": 0, "flag": False}
    shorter, longer = sum(r < 1 for r in ratios), sum(r > 1 for r in ratios)
    gmean = math.exp(statistics.fmean(math.log(r) for r in ratios))
    p = stats.mcnemar_exact(shorter, longer)  # the sign test
    return {"n": len(ratios), "ratio_gmean": round(gmean, 3), "shorter": shorter,
            "longer": longer, "p": p,
            "flag": gmean < BREVITY_FLOOR and p < stats.ALPHA and shorter > longer}


def cost(gen: dict[str, dict], keep: set[str]) -> dict:
    """What a pass spent: its tokens per item and in total, and the summed request latency. The
    window keeps 8 requests in flight, so a pass's wall time is about an eighth of that sum."""
    rows = [gen[i] for i in sorted(keep) if i in gen]
    if not rows:
        return {}
    tokens = sorted(r.get("completion_tokens") or 0 for r in rows)
    return {"items": len(rows), "errors": sum(bool(r.get("error")) for r in rows),
            "reasoning_chars_median": statistics.median(r.get("reasoning_chars") or 0
                                                        for r in rows),
            "completion_tokens": {"total": sum(tokens), "median": statistics.median(tokens),
                                  "p90": tokens[int(0.9 * (len(tokens) - 1))],
                                  "max": tokens[-1]},
            "prompt_tokens_total": sum(r.get("prompt_tokens") or 0 for r in rows),
            "latency_s_total": round(sum(r.get("elapsed_s") or 0 for r in rows), 1)}


def bench_summary(run_dir: Path, bench: str, state: str, base: str = "base",
                  aa: str = "base_rep") -> dict | None:
    """One benchmark: `state` and the A/A pass, each against the base."""
    path = run_dir / "items" / f"{bench}.jsonl"
    if not path.exists():
        return None
    items = {it.id for it in load_items(path)}
    gold = _rows(run_dir, "scores", bench, "gold") or {}
    # the gold scores cover the whole battery; only the subset's own items are listed
    unmeasurable = sorted(i for i, r in gold.items() if not r["passed"] and i in items)
    keep = items - set(unmeasurable)
    outcome: dict[str, dict[str, dict[str, bool]]] = {}
    gens: dict[str, dict[str, dict]] = {}
    for s in (base, state, aa):
        scores, gen = _rows(run_dir, "scores", bench, s), _rows(run_dir, "gen", bench, s)
        if scores is None or gen is None:
            continue
        gens[s] = gen
        # Only items the pass answered: a pass stopped by the window's time limit has score rows
        # (`no_generation`) for the rest, and they say nothing about the model.
        per_item = {i: conditions(bench, scores[i], gen[i]) for i in keep
                    if i in scores and i in gen and not gen[i].get("error")}
        outcome[s] = {c: {i: v[c] for i, v in per_item.items()} for c in ("passed", *BAD)}
    if base not in outcome or state not in outcome:
        return None
    out: dict = {"n": len(keep), "unmeasurable": unmeasurable, "states": {}}
    for s in (state, aa):
        if s not in outcome:
            continue
        checks = {"passed": compare(f"{bench}:passed", outcome[base]["passed"],
                                    outcome[s]["passed"], good=True)}
        for c in BAD:
            if c == "sql_error" and bench != "bird":
                continue
            checks[c] = compare(f"{bench}:{c}", outcome[base][c], outcome[s][c], good=False)
        checks["brevity"] = brevity(gens[base], gens[s], keep)
        out["states"][s] = {"checks": checks,
                            "flags": [c for c, v in checks.items() if v.get("flag")],
                            "cost": cost(gens[s], keep)}
    out["cost_base"] = cost(gens[base], keep)
    if aa in outcome:
        out["aa_flips"] = stats.aa_flip_rate(outcome[base]["passed"], outcome[aa]["passed"])[0]
    return out


def summary(run_dir: Path, state: str, base: str = "base", aa: str = "base_rep") -> dict:
    benches = {b: s for b in SIZES if (s := bench_summary(run_dir, b, state, base, aa))}
    flags = [f"{b}:{c}" for b, s in benches.items() for c in s["states"][state]["flags"]]
    aa_flags = [f"{b}:{c}" for b, s in benches.items()
                for c in s["states"].get(aa, {}).get("flags", [])]
    return {"state": state, "base": base, "aa": aa, "benches": benches, "flags": flags,
            "aa_flags": aa_flags, "screen": "pass" if not flags else "flagged",
            "missing": [b for b in SIZES if b not in benches]}


def markdown(result: dict) -> str:
    state = result["state"]
    lines = [f"Mini-battery, thinking on: `{state}` against `{result['base']}` "
             f"(screen: **{result['screen']}**)", "",
             "| Benchmark | Check | n | Base | State | Worse/better | p | A/A | Flag |",
             "|---|---|---:|---:|---:|---|---:|---|---|"]
    for bench, s in result["benches"].items():
        mine, aa = s["states"][state]["checks"], s["states"].get(result["aa"], {}).get("checks", {})
        for check, v in mine.items():
            if check == "brevity":
                cell = (f"ratio {v.get('ratio_gmean', '-')}", f"{v.get('shorter', 0)}/"
                        f"{v.get('longer', 0)}")
                lines.append(f"| {bench} | reasoning length | {v['n']} | 1 | {cell[0]} | "
                             f"{cell[1]} shorter/longer | {v.get('p', 1):.3g} | "
                             f"{aa.get(check, {}).get('ratio_gmean', '-')} | "
                             f"{'**flag**' if v['flag'] else ''} |")
                continue
            noise = aa.get(check)
            lines.append(f"| {bench} | {check} | {s['n']} | {v['base']} | {v['other']} | "
                         f"{v['worse']}/{v['better']} | {v['p']:.3g} | "
                         f"{'' if noise is None else noise['other']} | "
                         f"{'**flag**' if v['flag'] else ''} |")
    lines += ["", f"Flags: {', '.join(result['flags']) or 'none'}. A/A flags (noise beyond the "
              f"test's): {', '.join(result['aa_flags']) or 'none'}."]
    if result["missing"]:
        lines.append(f"Not scored: {', '.join(result['missing'])}.")
    lines += ["", "| Benchmark | Pass | Items | Errors | Tokens: median | p90 | max | total | "
              "Reasoning chars, median |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for bench, s in result["benches"].items():
        passes = {result["base"]: s["cost_base"],
                  **{name: st["cost"] for name, st in s["states"].items()}}
        for name, c in passes.items():
            if c:
                t = c["completion_tokens"]
                lines.append(f"| {bench} | {name} | {c['items']} | {c['errors']} | {t['median']} | "
                             f"{t['p90']} | {t['max']} | {t['total']} | "
                             f"{c['reasoning_chars_median']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="write the subsets")
    p.add_argument("--items", required=True, type=Path, help="the battery's items directory")
    p.add_argument("--out", required=True, type=Path, help="RUN/items")
    p.add_argument("--seed", type=int, default=SEED)
    s = sub.add_parser("summary", help="a state against the base")
    s.add_argument("--run-dir", required=True, type=Path)
    s.add_argument("--state", default="adapter")
    s.add_argument("--base", default="base")
    s.add_argument("--aa", default="base_rep")
    s.add_argument("--out-json", type=Path)
    s.add_argument("--out-md", type=Path)
    args = ap.parse_args()
    if args.cmd == "prepare":
        print(json.dumps(prepare(args.items, args.out, args.seed), indent=1))
        return
    result = summary(args.run_dir, args.state, args.base, args.aa)
    text = markdown(result)
    if args.out_json:
        args.out_json.write_text(json.dumps(result, indent=1) + "\n")
    if args.out_md:
        args.out_md.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
