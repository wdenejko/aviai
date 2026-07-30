"""Compare two runs on the same frozen eval — the paired "did B beat A?" report (Phase 4).

The runner (runner.py) measures one model in isolation. This is the other half the stats
layer was built for (see stats.py's `mcnemar` docstring): a *paired* comparison of a base
model against its finetune on the SAME 620 records, so a lift is tested for significance,
not just eyeballed. Reused for every rung of the size sweep — base vs LoRA at each size.

Pairing is exact and unit-honest:
- **Value accuracy** pairs on each *(record, field) where the reference value existed* —
  the same population `recall` is computed over. Whether a field "existed" depends only on
  the (fixed) reference, so the item set is identical for both runs → a clean McNemar pair.
- **Whole-record EM** pairs on each record.
- **Hallucination** is a rate over *truly-absent* fields (the safety axis), reported as a
  delta; McNemar on it too, over the (record, absent-field) population.

McNemar reads only the discordant pairs: b = base right & finetune wrong (regressions),
c = base wrong & finetune right (fixes). A finetune that helps has c >> b.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avtext.harness.stats import mcnemar

# Per-field outcomes partition into "reference present" (a value existed → recall population)
# and "reference absent" (→ hallucination population). Keep the two axes separate.
_PRESENT = {"hit", "wrong", "abstain"}  # the value existed; hit is the only success
_ABSENT = {"hallucinate", "true_abstain"}  # no value; hallucinate is the failure


def _load(run_dir: Path) -> dict[str, dict]:
    rows = [json.loads(x) for x in (run_dir / "scores.jsonl").read_text().splitlines()]
    return {r["id"]: r for r in rows}


def _value_hits(row: dict) -> dict[str, bool]:
    """field -> was it a HIT, over fields whose reference value existed."""
    return {f: (o == "hit") for f, o in row["outcomes"].items() if o in _PRESENT}


def _absent_ok(row: dict) -> dict[str, bool]:
    """field -> did the model correctly abstain, over fields whose reference was absent."""
    return {f: (o == "true_abstain") for f, o in row["outcomes"].items() if o in _ABSENT}


def compare(base_dir: Path, ft_dir: Path) -> str:
    a, b = _load(base_dir), _load(ft_dir)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit("no shared record ids between the two runs")

    # --- value accuracy (per present-field) ---
    va_base: list[bool] = []
    va_ft: list[bool] = []
    for i in ids:
        ha, hb = _value_hits(a[i]), _value_hits(b[i])
        for f in ha.keys() & hb.keys():  # identical set (reference-driven), but be defensive
            va_base.append(ha[f])
            va_ft.append(hb[f])
    m_va = mcnemar(va_base, va_ft)
    acc_base = sum(va_base) / len(va_base)
    acc_ft = sum(va_ft) / len(va_ft)

    # --- whole-record exact match (per record) ---
    em_base = [bool(a[i]["exact_match"]) for i in ids]
    em_ft = [bool(b[i]["exact_match"]) for i in ids]
    m_em = mcnemar(em_base, em_ft)

    # --- hallucination (per absent-field): correct-abstain rate; halluc = 1 - that ---
    ab_base: list[bool] = []
    ab_ft: list[bool] = []
    for i in ids:
        aa, ab = _absent_ok(a[i]), _absent_ok(b[i])
        for f in aa.keys() & ab.keys():
            ab_base.append(aa[f])
            ab_ft.append(ab[f])
    hr_base = 1 - sum(ab_base) / len(ab_base) if ab_base else 0.0
    hr_ft = 1 - sum(ab_ft) / len(ab_ft) if ab_ft else 0.0

    def _line(name: str, base: float, ft: float) -> str:
        return f"| {name} | {base:.1%} | {ft:.1%} | {ft - base:+.1%} |"

    out = [
        f"# Paired comparison — base vs finetuned ({len(ids)} records)",
        "",
        f"- **base:** `{base_dir.name}`",
        f"- **finetuned:** `{ft_dir.name}`",
        "",
        "| metric | base | finetuned | Δ |",
        "|---|--:|--:|--:|",
        _line("value accuracy (recall)", acc_base, acc_ft),
        _line("whole-record EM", sum(em_base) / len(ids), sum(em_ft) / len(ids)),
        _line("hallucination rate", hr_base, hr_ft),
        "",
        "## McNemar (paired significance)",
        "b = base-right/finetune-wrong (regressions) · c = base-wrong/finetune-right (fixes)",
        "",
        "| axis | b | c | statistic | p-value |",
        "|---|--:|--:|--:|--:|",
        f"| value accuracy | {m_va.b} | {m_va.c} | {m_va.statistic:.1f} | {m_va.p_value:.2e} |",
        f"| whole-record EM | {m_em.b} | {m_em.c} | {m_em.statistic:.1f} | {m_em.p_value:.2e} |",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired base-vs-finetuned comparison on one eval.")
    ap.add_argument("base", type=Path, help="reports/.../runs/<base run dir>")
    ap.add_argument("finetuned", type=Path, help="reports/.../runs/<finetuned run dir>")
    args = ap.parse_args()
    print(compare(args.base, args.finetuned))


if __name__ == "__main__":
    main()
