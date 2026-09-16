"""Aggregate per-problem results into the category x difficulty table everyone actually reads."""

from __future__ import annotations

from collections import defaultdict

from dsbench.schema import CATEGORIES, CATEGORY_NAMES, DIFFICULTIES, ProblemResult


def cells(results: list[ProblemResult]) -> dict[tuple[str, str], tuple[int, int]]:
    """(category, difficulty) -> (passed, total)."""
    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        agg[(r.category, r.difficulty)][0] += int(r.passed)
        agg[(r.category, r.difficulty)][1] += 1
    return {k: (v[0], v[1]) for k, v in agg.items()}


def _pct(p: int, t: int) -> str:
    return f"{p}/{t}" if t else "-"


def render_markdown(meta: dict, results: list[ProblemResult]) -> str:
    c = cells(results)
    passed = sum(r.passed for r in results)
    total = len(results)
    truncated = sum(1 for r in results if r.status == "truncated")
    think = "off" if meta.get("no_think") else "on"
    lines = [
        f"# dsbench run: {meta.get('label', 'run')}",
        "",
        f"- model: `{meta.get('model')}`  endpoint: `{meta.get('base_url')}`",
        f"- when: {meta.get('timestamp')} temp: {meta.get('temperature')} "
        f"pass@{meta.get('k', 1)} thinking: {think}",
        f"- score: **{passed}/{total}** ({100 * passed / total:.1f}%)" if total else "- score: -",
    ]
    if truncated:
        # Model-never-answered artifacts, not capability misses -- called out so the headline
        # score is read with them in mind (consider raising --max-tokens or --no-think).
        lines.append(f"- ⚠ {truncated} truncated (hit token cap, no answer; not a capability miss)")
    lines.append("")
    # Header follows DIFFICULTIES so adding a tier (e.g. `expert`) needs no edit here.
    lines.append("| category | " + " | ".join(DIFFICULTIES) + " | total |")
    lines.append("|" + "|".join(["---"] * (len(DIFFICULTIES) + 2)) + "|")
    for cat in CATEGORIES:
        row = [CATEGORY_NAMES[cat]]
        cp = ct = 0
        for diff in DIFFICULTIES:
            p, t = c.get((cat, diff), (0, 0))
            cp += p
            ct += t
            row.append(_pct(p, t))
        row.append(_pct(cp, ct))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Failures listed so a run is actionable, not just a number.
    fails = [r for r in results if not r.passed]
    if fails:
        lines += ["## failures", "", "| id | difficulty | status | reason |", "|---|---|---|---|"]
        for r in sorted(fails, key=lambda r: r.id):
            reason = r.reason.replace("|", "\\|")[:100]
            lines.append(f"| {r.id} | {r.difficulty} | {r.status} | {reason} |")
        lines.append("")
    return "\n".join(lines)
