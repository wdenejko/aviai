"""Summarise a battery run: paired deltas per benchmark, the A/A noise floor, and the Gate-2 call.

Reads RUN/scores/<bench>.<state>.jsonl and writes one JSON file (everything) and one Markdown
table (the headline). Three things are applied before any delta is computed:

* Unmeasurable items are dropped: those whose published reference solution fails in our harness
  (RUN/scores/<bench>.gold.jsonl, from `score.py --gold`). An item nothing can pass tells us
  nothing about either state.
* The A/A control: base vs base_rep, same state twice. Its flip count is the harness's own noise
  on that benchmark; the adapter's discordant pairs are read against it.
* The contamination check (`contamination.py`), if given: each delta is recomputed without the
  items whose text overlaps the training data strongly, so a gain that exists only on seen items
  shows up as such.
* Think-leak accounting. The Gate-2 adapter opens its own `<think>` block on many prompts although
  thinking is off (6% of its training turns began with real reasoning). A reply that spends its
  budget reasoning fails for a FORMAT reason, so each delta is also recomputed on the items where
  neither state leaked. That subset is a diagnostic, not the verdict: the verdict is the model
  as served.

Run:  python -m dsbench.battery.report --run-dir RUN [--contamination C] --out-json J --out-md M
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dsbench.battery import stats
from dsbench.battery.items import by_id, load_items, read_jsonl

BENCH_ORDER = ("ds1000", "bird", "dsbench", "bfcl_ast", "ifeval", "mmlu_pro", "gpqa", "lcb",
               "humaneval_plus", "bfcl_irrelevance")
# Result rows that are a slice of one scored benchmark: {row: (source bench, item filter)}.
SPLITS = {
    "bfcl_ast": ("bfcl", lambda it: it.meta.get("category") != "irrelevance"),
    "bfcl_irrelevance": ("bfcl", lambda it: it.meta.get("category") == "irrelevance"),
}
STRATA = {"ds1000": "library", "lcb": "difficulty", "bird": "difficulty", "bfcl": "category",
          "mmlu_pro": "category"}


def _scores(run_dir: Path, bench: str, state: str) -> dict[str, dict] | None:
    path = run_dir / "scores" / f"{bench}.{state}.jsonl"
    return by_id(read_jsonl(path)) if path.exists() else None


def _outcomes(rows: dict[str, dict], keep: set[str]) -> dict[str, bool]:
    return {i: bool(r["passed"]) for i, r in rows.items() if i in keep}


def _ifeval_levels(rows: dict[str, dict], keep: set[str]) -> dict[str, float]:
    """IFEval's four published numbers: prompt- and instruction-level, strict and loose."""
    kept = [rows[i] for i in keep if i in rows and "extra" in rows[i]]
    strict = [x for r in kept for x in r["extra"]["strict_list"]]
    loose = [x for r in kept for x in r["extra"]["loose_list"]]
    n = max(1, len(kept))
    return {"prompt_strict": 100 * sum(r["passed"] for r in kept) / n,
            "prompt_loose": 100 * sum(r["extra"]["loose_all"] for r in kept) / n,
            "inst_strict": 100 * sum(strict) / max(1, len(strict)),
            "inst_loose": 100 * sum(loose) / max(1, len(loose))}


def _leaks(run_dir: Path, bench: str, state: str) -> dict[str, str] | None:
    """Per item: 'none', or a reasoning block the model opened itself: 'closed' / 'open'."""
    path = run_dir / "gen" / f"{bench}.{state}.jsonl"
    if not path.exists():
        return None
    out = {}
    for row in read_jsonl(path):
        text = (row.get("content") or "").lstrip()
        out[row["id"]] = ("none" if not text.startswith("<think>")
                          else "closed" if "</think>" in text else "open")
    return out


def bench_summary(run_dir: Path, bench: str, strong: set[str] | None,
                  row: str | None = None, only=None) -> dict | None:
    """Paired summary of one benchmark, or of the slice `only(item)` reported as `row`."""
    row = row or bench
    base, adapter = _scores(run_dir, bench, "base"), _scores(run_dir, bench, "adapter")
    if not base or not adapter:
        return None
    # The agentic suite has no items file (pi drives it); its problems are the ids it scored.
    items_path = run_dir / "items" / f"{bench}.jsonl"
    items = {it.id: it for it in load_items(items_path)} if items_path.exists() else {}
    gold = _scores(run_dir, bench, "gold")
    unmeasurable = sorted(i for i, r in (gold or {}).items() if not r["passed"])
    keep = (set(items) or set(base)) - set(unmeasurable)
    if only is not None:
        keep = {i for i in keep if only(items[i])}
    base_o, adapter_o = _outcomes(base, keep), _outcomes(adapter, keep)
    result = stats.paired(row, base_o, adapter_o)
    out: dict = {"paired": result.as_dict(), "verdict": stats.verdict(result),
                 "unmeasurable": unmeasurable}
    half = _scores(run_dir, bench, "adapter_half")
    if half:
        half_result = stats.paired(row, base_o, _outcomes(half, keep))
        out["half"] = {"paired": half_result.as_dict(), "verdict": stats.verdict(half_result)}
    rep = _scores(run_dir, bench, "base_rep")
    if rep:
        flips, n = stats.aa_flip_rate(base_o, _outcomes(rep, keep))
        out["aa"] = {"flips": flips, "n": n,
                     "acc_base_rep": 100 * sum(_outcomes(rep, keep).values()) / max(1, n)}
    if strong is not None:
        clean = keep - strong
        out["contaminated_strong"] = len(keep & strong)
        if clean and clean != keep:
            out["clean"] = stats.paired(row, _outcomes(base, clean),
                                        _outcomes(adapter, clean)).as_dict()
    leaks = {s: _leaks(run_dir, bench, s) for s in ("base", "adapter")}
    if leaks["base"] is not None and leaks["adapter"] is not None:
        out["think_leak"] = {s: dict(sorted(_count(v for i, v in lk.items() if i in keep).items()))
                             for s, lk in leaks.items()}
        clean_fmt = {i for i in keep
                     if leaks["base"].get(i) == "none" and leaks["adapter"].get(i) == "none"}
        if clean_fmt and clean_fmt != keep:
            out["no_leak"] = stats.paired(row, _outcomes(base, clean_fmt),
                                          _outcomes(adapter, clean_fmt)).as_dict()
    out["truncated"] = {s: sum(1 for i, r in rows.items() if i in keep
                               and r.get("finish_reason") == "length")
                        for s, rows in (("base", base), ("adapter", adapter))}
    out["status"] = {s: dict(sorted(_count(r.get("status") for i, r in rows.items()
                                           if i in keep).items()))
                     for s, rows in (("base", base), ("adapter", adapter))}
    if bench == "ifeval":
        out["ifeval_levels"] = {"base": _ifeval_levels(base, keep),
                                "adapter": _ifeval_levels(adapter, keep)}
    if bench == "bfcl":
        out["format_ok"] = {s: 100 * sum(bool((r.get("extra") or {}).get("format_ok"))
                                         for i, r in rows.items() if i in keep) / len(keep)
                            for s, rows in (("base", base), ("adapter", adapter))}
    if bench in STRATA:
        key = STRATA[bench]
        groups: dict[str, set[str]] = defaultdict(set)
        for i in keep:
            groups[str(items[i].meta.get(key))].add(i)
        out["strata"] = {g: {"n": len(ids),
                             "base": 100 * sum(base_o[i] for i in ids) / len(ids),
                             "adapter": 100 * sum(adapter_o[i] for i in ids) / len(ids)}
                         for g, ids in sorted(groups.items())}
    return out


def _count(values) -> dict:
    counts: dict = defaultdict(int)
    for v in values:
        counts[str(v)] += 1
    return counts


def markdown(summaries: dict[str, dict], decision: dict) -> str:
    lines = ["| Benchmark | n | Base | Adapter | Δ pp (95% CI) | +/− | McNemar p | A/A flips "
             "| Verdict |", "|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for bench in BENCH_ORDER:
        s = summaries.get(bench)
        if s is None:
            continue
        p = s["paired"]
        aa = f"{s['aa']['flips']}/{s['aa']['n']}" if "aa" in s else "–"
        lines.append(
            f"| {bench} | {p['n']} | {p['acc_base']:.1f} | {p['acc_adapter']:.1f} | "
            f"{p['delta']:+.1f} ({p['ci_low']:+.1f}, {p['ci_high']:+.1f}) | "
            f"+{p['gains']}/−{p['losses']} | {p['p']:.3g} | {aa} | {s['verdict']} |")
    lines += ["", f"Target gains (need ≥ 2): {', '.join(decision['target_gains']) or 'none'}. "
              f"Regressions past threshold: {', '.join(decision['regressions']) or 'none'}. "
              f"Not measured: {', '.join(decision['not_measured']) or 'none'}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--contamination", type=Path, default=None)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    args = ap.parse_args()
    contamination = (json.loads(args.contamination.read_text())["benches"]
                     if args.contamination else {})
    summaries = {}
    for row in BENCH_ORDER:
        bench, only = SPLITS.get(row, (row, None))
        strong = (set(contamination[bench]["strong"]) if bench in contamination
                  else (set() if args.contamination else None))
        s = bench_summary(args.run_dir, bench, strong, row=row, only=only)
        if s is not None:
            summaries[row] = s
    decision = stats.gate2_decision({b: stats.Paired(**s["paired"])
                                     for b, s in summaries.items()})
    parity_path = args.run_dir / "parity.json"
    args.out_json.write_text(json.dumps({
        "summaries": summaries, "decision": decision,
        "manifest": json.loads((args.run_dir / "items" / "manifest.json").read_text()),
        "contamination": {b: {"n": c["n"], "any": len(c["any"]), "strong": len(c["strong"])}
                          for b, c in contamination.items()},
        "parity_file_present": parity_path.exists(),
    }, indent=1))
    args.out_md.write_text(markdown(summaries, decision))
    print(markdown(summaries, decision))


if __name__ == "__main__":
    main()
