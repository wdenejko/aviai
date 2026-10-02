"""The mini-battery's calibration: the base twice with thinking on, read for cost, budget and noise.

ADR-004 Revision 2, action item 8. The calibration window runs the base (`base`) and its A/A pass
(`base_rep`, other seeds) on the mini-battery's items. This reads, per benchmark:
- **Cost:** tokens per reply (median, p90, p99, max, total), the share spent in replies that
  never closed their reasoning, and each pass's wall time from the window's log.
- **Budget:** how long closed replies run, and what a smaller budget would cut. A reply that
  closes past it would no longer close; one that loops would stop sooner.
- **Passes**, once scored:
  - the scorer's, and served (the scorer's and closed, as `mini.conditions` counts them);
  - against Gate 2's base on the same items, which ran greedy with thinking off.
- **Noise:** the A/A pass's discordant items. From them, the smallest difference each subset
  detects at 80% power (exact McNemar at 5%, by the normal approximation). Its discordance is a
  floor for a checkpoint's, so the figure is optimistic.
- **Gate 2 in retrospect:** the mini-battery's checks on Gate 2's adapter over the same subsets,
  thinking off as it ran. It needs no generation.

On the box, from the battery's synced source:

    cd ~/benchlab/scripts/battery/src && PYTHONPATH=. ~/benchlab/batteryvenv/bin/python \
        PATH/TO/calibration.py RUN GATE2_RUN WINDOW_LOG[,WINDOW_LOG...] OUT.json
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

from dsbench.battery import mini, stats
from dsbench.battery.items import by_id, load_items, read_jsonl

STATES = ("base", "base_rep")
BUDGETS = (4096, 6144, 8192, 10240)  # the mini-battery's is 12,288
Z = 1.96 + 0.84  # two-sided 5%, 80% power


def quantile(values: list[int], q: float) -> int:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1)))]


def pass_times(logs: list[Path]) -> dict[str, float]:
    """Minutes per `bench:state` pass, from the windows' step lines (box clock). A pass stopped by
    one window's time limit and resumed by the next counts both parts."""
    minutes: dict[str, float] = {}
    for log in logs:
        marks = []
        for line in log.read_text().splitlines():
            m = re.match(r"\[window\] (\d\d:\d\d:\d\d) (generate (\S+) (\S+)|plan done|time limit)",
                         line)
            if m:
                name = f"{m.group(3)}:{m.group(4)}" if m.group(3) else "end"
                marks.append((datetime.strptime(m.group(1), "%H:%M:%S"), name))
        for (when, name), (later, _) in zip(marks, marks[1:], strict=False):
            if name != "end":
                minutes[name] = minutes.get(name, 0) + (later - when).total_seconds() / 60
    return {name: round(m, 1) for name, m in minutes.items()}


def budget_effect(rows: list[dict], budget: int) -> dict:
    """At a smaller budget: closed replies that ran past it, and the tokens saved."""
    closed_over = sum(1 for r in rows if not r.get("unclosed")
                      and (r.get("completion_tokens") or 0) > budget)
    saved = sum(max(0, (r.get("completion_tokens") or 0) - budget) for r in rows)
    total = sum(r.get("completion_tokens") or 0 for r in rows)
    return {"closed_cut": closed_over, "tokens_saved_pct": round(100 * saved / max(1, total), 1)}


def cost(rows: list[dict]) -> dict:
    tokens = [r.get("completion_tokens") or 0 for r in rows]
    closed = [t for r, t in zip(rows, tokens, strict=True) if not r.get("unclosed")]
    looping = sum(t for r, t in zip(rows, tokens, strict=True) if r.get("unclosed"))
    return {
        "n": len(rows), "errors": sum(bool(r.get("error")) for r in rows),
        "finish": dict(sorted({k: sum(r.get("finish_reason") == k for r in rows)
                               for k in {r.get("finish_reason") for r in rows}}.items(),
                              key=lambda kv: str(kv[0]))),
        "unclosed": sum(bool(r.get("unclosed")) for r in rows),
        "tokens": {"median": statistics.median(tokens), "p90": quantile(tokens, 0.9),
                   "p99": quantile(tokens, 0.99), "max": max(tokens), "total": sum(tokens)},
        "closed_tokens": {"p90": quantile(closed, 0.9), "p99": quantile(closed, 0.99),
                          "max": max(closed)} if closed else {},
        "looping_tokens_pct": round(100 * looping / max(1, sum(tokens)), 1),
        "budgets": {b: budget_effect(rows, b) for b in BUDGETS},
    }


def passes(bench: str, run: Path, gate2: Path, keep: set[str]) -> dict | None:
    """Scorer's and served passes per state, Gate 2's thinking-off base, and the A/A noise."""
    out: dict = {}
    served: dict[str, dict[str, bool]] = {}
    for state in STATES:
        scores = run / "scores" / f"{bench}.{state}.jsonl"
        if not scores.exists():
            continue
        s, g = by_id(read_jsonl(scores)), by_id(read_jsonl(run / "gen" / f"{bench}.{state}.jsonl"))
        # only items the pass answered: one stopped by the time limit has no generation for the rest
        ids = [i for i in sorted(keep) if i in s and i in g and not g[i].get("error")]
        served[state] = {i: mini.conditions(bench, s[i], g[i])["passed"] for i in ids}
        out[state] = {"n": len(ids),
                      "scorer_pct": round(100 * sum(s[i]["passed"] for i in ids) / len(ids), 1),
                      "served_pct": round(100 * sum(served[state].values()) / len(ids), 1)}
    if "base" not in served:
        return None
    off = gate2 / "scores" / f"{bench}.base.jsonl"
    if off.exists():
        s_off = by_id(read_jsonl(off))
        ids = [i for i in served["base"] if i in s_off]
        p = stats.paired(bench, {i: bool(s_off[i]["passed"]) for i in ids},
                         {i: served["base"][i] for i in ids})
        out["thinking_off_base"] = {"n": p.n, "off_pct": round(p.acc_base, 1),
                                    "on_pct": round(p.acc_adapter, 1), "losses": p.losses,
                                    "gains": p.gains, "p": p.p}
    if "base_rep" in served:
        flips, n = stats.aa_flip_rate(served["base"], served["base_rep"])
        d = flips / max(1, n)
        out["aa"] = {"discordant": flips, "n": n, "rate": round(d, 3),
                     "mde_pp": round(100 * Z * math.sqrt(d / n), 1) if n else None}
    return out


def gate2_flags(bench: str, gate2: Path, keep: set[str]) -> dict:
    """The mini-battery's checks run on Gate 2's adapter, and on it at half scale, against Gate 2's
    base over this subset: would the subset have flagged the habits it is sized for?

    Gate 2 ran greedy with thinking off, so this is a retrospective, not the same test: there is
    no reasoning whose length to compare, and none to leave unclosed. Its think-leak opened a
    reasoning block inside the reply, which is the `second_think` check."""
    rows: dict[str, dict[str, dict[str, bool]]] = {}
    for state in ("base", "adapter", "adapter_half"):
        s = gate2 / "scores" / f"{bench}.{state}.jsonl"
        g = gate2 / "gen" / f"{bench}.{state}.jsonl"
        if s.exists() and g.exists():
            scores, gen = by_id(read_jsonl(s)), by_id(read_jsonl(g))
            rows[state] = {i: mini.conditions(bench, scores[i], gen[i]) for i in keep
                           if i in scores and i in gen}
    if "base" not in rows:
        return {}
    out: dict = {}
    for state in ("adapter", "adapter_half"):
        if state not in rows:
            continue
        checks = {c: mini.compare(f"{bench}:{c}", {i: v[c] for i, v in rows["base"].items()},
                                  {i: v[c] for i, v in rows[state].items()}, good=c == "passed")
                  for c in ("passed", "truncated", "second_think", "sql_error")
                  if c != "sql_error" or bench == "bird"}
        out[state] = {"n": len(rows["base"].keys() & rows[state].keys()), "checks": checks,
                      "flags": [c for c, v in checks.items() if v["flag"]]}
    return out


def main() -> None:
    run, gate2, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[4])
    logs = [Path(log) for log in sys.argv[3].split(",")]
    result: dict = {"pass_minutes": pass_times(logs), "benches": {}}
    for bench in mini.SIZES:
        items = run / "items" / f"{bench}.jsonl"
        if not items.exists():
            continue
        gold = run / "scores" / f"{bench}.gold.jsonl"
        unmeasurable = {r["id"] for r in read_jsonl(gold) if not r["passed"]} if gold.exists() \
            else set()
        ids = {it.id for it in load_items(items)}
        keep = ids - unmeasurable
        entry: dict = {"items": len(keep), "unmeasurable": sorted(unmeasurable & ids)}
        for state in STATES:
            gen = run / "gen" / f"{bench}.{state}.jsonl"
            if gen.exists():
                rows = [r for i, r in by_id(read_jsonl(gen)).items() if i in keep]
                entry[state] = cost(rows)
        entry["passes"] = passes(bench, run, gate2, keep)
        entry["gate2_flags"] = gate2_flags(bench, gate2, keep)
        result["benches"][bench] = entry
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
