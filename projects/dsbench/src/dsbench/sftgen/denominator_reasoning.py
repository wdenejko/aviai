"""Target B generator -- conditional-population / denominator reasoning (ADR-004).

The measured gap: both base and Ornith divide by the WRONG base on ratio questions
(`da_delay_attribution` 0/5, two different stable wrong answers -> a real reasoning gap). This
generator teaches the transferable skill on NON-aviation data, as execution-filtered teacher traces:

  1. build a small synthetic table, INLINE it in the prompt (a teacher with only a schema recites
     "SUM/COUNT" instead of a number), and compute the truth two ways (pandas + a DuckDB reference
     SQL); if they disagree the problem is skipped as buggy;
  2. ask a licence-clean teacher (Ling-3.0-flash / DeepSeek / Qwen-class) to reason it out;
  3. KEEP the teacher's trace only if its answer matches the verified truth AND it exposed a trace
     -- the reasoning is the point of Target B. The data is small enough to reason over by hand, so
     the ONLY thing separating right from wrong is the denominator/population choice.

Trap families (the four ADR-004 denominator traps):
  * `conditional-null-share` -- cause columns populated only under a condition (null = N/A, not 0);
    the share denominator is the sum of the causes, not a grand total (da_delay_attribution analog).
  * `pooled-vs-mean-rate` -- the overall rate is pooled sum/sum, not the mean of per-group rates
    (da_weighted_ontime analog; uneven groups make the two answers diverge).
  * `share-of-subtotal` -- divide by the category SUBTOTAL, not the grand total.
  * `excluded-denominator` -- cancelled rows count as 0 in the numerator but stay in the denominator
    (da_all_flights_avg_delay analog).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dsbench.sftgen.engines import DuckDBEngine
from dsbench.sftgen.schema import Provenance, SFTRow, Turn, Verification, row_to_dict
from dsbench.sftgen.teacher import HTTPTeacher, Teacher

_CAUSES = ["cause_a", "cause_b", "cause_c", "cause_d", "cause_e"]
_REGIONS = ["us-east", "us-west", "eu-central", "ap-south"]

SOLVE_SYSTEM = (
    "You are a careful data analyst. Reason step by step about WHICH rows are in scope and WHAT "
    "the correct denominator is, then give the final numeric answer on its own line as "
    "'Answer: <n>'."
)


@dataclass(frozen=True)
class BProblem:
    family: str
    table: str
    df: pd.DataFrame
    question: str
    truth: float
    ref_sql: str  # independent SQL route, cross-checked against the pandas truth
    tol: float
    tags: tuple[str, ...]


def _md_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a markdown table (nulls -> NULL) for inlining in a prompt."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for rec in df.itertuples(index=False, name=None):
        cells = []
        for v in rec:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("NULL")
            elif isinstance(v, float):
                cells.append(f"{v:g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _conditional_null_share(seed: int, n: int = 12) -> BProblem:
    """Small INLINE table: the teacher must sum causes over the non-null rows and pick the base."""
    rng = np.random.default_rng(seed)
    breached = rng.random(n) < 0.5  # ~half breached; causes populated ONLY for those
    cols = {"incident_id": np.arange(1, n + 1, dtype="int64"),
            "region": rng.choice(_REGIONS, size=n), "breached": breached.astype("int64")}
    for c in _CAUSES:
        vals = np.round(rng.gamma(2.0, 30.0, size=n), 1)
        vals[~breached] = np.nan  # null where not applicable
        cols[c] = vals
    df = pd.DataFrame(cols)
    filled = df[_CAUSES].fillna(0.0)
    total = float(filled.to_numpy().sum())
    truth = round(100.0 * float(filled["cause_c"].sum()) / total, 1)
    den = " + ".join(f"sum({c})" for c in _CAUSES)
    ref_sql = f"SELECT 100.0 * sum(cause_c) / ({den}) FROM incidents"
    q = (
        "Here is the full `incidents` table. The five cause columns hold minutes and are populated "
        "ONLY for incidents that breached SLA (breached = 1); otherwise they are NULL.\n\n"
        f"{_md_table(df)}\n\n"
        "Of the total minutes summed across all FIVE cause columns, what percentage is "
        "attributable to cause_c? Reply rounded to 1 decimal."
    )
    return BProblem("conditional-null-share", "incidents", df, q, truth, ref_sql, 0.2,
                    ("ratio", "null", "denominator"))


def _pooled_vs_mean_rate(seed: int) -> BProblem:
    """Small INLINE per-store summary: pooled Sum/Sum vs the mean of per-store rates diverge."""
    rng = np.random.default_rng(seed)
    stores = [f"store_{i}" for i in range(6)]
    attempts = rng.integers(20, 600, size=len(stores)).astype("int64")  # uneven volumes
    rates = rng.uniform(0.55, 0.95, size=len(stores))
    successes = np.round(attempts * rates).astype("int64")
    df = pd.DataFrame({"store": stores, "successes": successes, "attempts": attempts})
    truth = round(float(df["successes"].sum()) / float(df["attempts"].sum()), 4)  # pooled
    ref_sql = "SELECT sum(successes) * 1.0 / sum(attempts) FROM attempts"
    q = (
        "Here is a per-store summary of checkout `attempts` (successes out of attempts).\n\n"
        f"{_md_table(df)}\n\n"
        "What is the OVERALL success rate across ALL attempts (total successes divided by total "
        "attempts, pooling every store together)? The stores have very different volumes. Reply as "
        "a fraction rounded to 4 decimals."
    )
    return BProblem("pooled-vs-mean-rate", "attempts", df, q, truth, ref_sql, 0.0005,
                    ("ratio", "pooled", "weighting"))


def _share_of_subtotal(seed: int) -> BProblem:
    """Denominator = the category SUBTOTAL, not the grand total (share-of-part vs share-of-all)."""
    rng = np.random.default_rng(seed)
    cats = {"electronics": ["phones", "laptops", "audio"],
            "apparel": ["shoes", "coats"], "grocery": ["produce", "dairy"]}
    rows = []
    for cat, subs in cats.items():
        for sub in subs:
            rows.append((cat, sub, float(np.round(rng.gamma(3.0, 400.0), 2))))
    df = pd.DataFrame(rows, columns=["category", "subcategory", "amount"])
    tcat = "electronics"
    tsub = str(rng.choice(cats[tcat]))
    part = float(df[(df.category == tcat) & (df.subcategory == tsub)]["amount"].sum())
    subtotal = float(df[df.category == tcat]["amount"].sum())
    truth = round(100.0 * part / subtotal, 1)
    ref_sql = (
        f"SELECT 100.0 * sum(CASE WHEN category='{tcat}' AND subcategory='{tsub}' THEN amount "
        f"ELSE 0 END) / sum(CASE WHEN category='{tcat}' THEN amount ELSE 0 END) FROM sales"
    )
    q = (
        "Here is the full `sales` table (amount per subcategory).\n\n"
        f"{_md_table(df)}\n\n"
        f"Within category '{tcat}' ONLY, what percentage of that category's total amount comes "
        f"from subcategory '{tsub}'? Reply rounded to 1 decimal."
    )
    return BProblem("share-of-subtotal", "sales", df, q, truth, ref_sql, 0.2,
                    ("ratio", "subtotal", "denominator"))


def _excluded_denominator(seed: int, n: int = 12) -> BProblem:
    """Cancelled rows count as 0 in the numerator but stay in the denominator (da_all_flights)."""
    rng = np.random.default_rng(seed)
    cancelled = rng.random(n) < 0.35
    delay = np.round(rng.gamma(2.0, 20.0, size=n), 1)
    delay[cancelled] = np.nan  # delay is NULL for cancelled orders
    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1, dtype="int64"),
        "cancelled": cancelled.astype("int64"),
        "delay_minutes": delay,
    })
    num = float(np.nan_to_num(delay).sum())  # cancelled -> 0 delay
    truth = round(num / float(n), 2)  # denominator = ALL orders, cancelled included
    ref_sql = "SELECT sum(coalesce(delay_minutes, 0)) * 1.0 / count(*) FROM orders"
    q = (
        "Here is the full `orders` table. delay_minutes is NULL for cancelled orders "
        "(cancelled = 1).\n\n"
        f"{_md_table(df)}\n\n"
        "What is the average delay PER ORDER across ALL orders, counting each cancelled order as 0 "
        "delay (they still count in the denominator)? Reply rounded to 2 decimals."
    )
    return BProblem("excluded-denominator", "orders", df, q, truth, ref_sql, 0.05,
                    ("ratio", "excluded", "denominator"))


_TRAPS = {"conditional-null-share": _conditional_null_share,
          "pooled-vs-mean-rate": _pooled_vs_mean_rate,
          "share-of-subtotal": _share_of_subtotal,
          "excluded-denominator": _excluded_denominator}


def _extract_number(text: str) -> float | None:
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m[-1]) if m else None  # the LAST number (the 'Answer: n' line)


def _self_verify(prob: BProblem) -> bool:
    """Confirm the pandas truth matches an independent SQL route before trusting the problem."""
    eng = DuckDBEngine()
    eng.setup()
    try:
        eng.load(prob.table, prob.df)
        got = eng.scalar(prob.ref_sql)
        return got is not None and abs(float(got) - prob.truth) <= max(prob.tol, 1e-6)
    finally:
        eng.teardown()


def process_problem(
    prob: BProblem, teacher: Teacher, pseed: int, teacher_id: str,
) -> tuple[SFTRow | None, str | None]:
    """Solve with the teacher and execution-filter. Returns (row, None) or (None, reject_reason)."""
    reply = teacher.answer(SOLVE_SYSTEM, prob.question)
    got = _extract_number(reply.content)
    if got is None or abs(got - prob.truth) > prob.tol:
        return None, "wrong-answer"
    if not reply.thinking:
        return None, "no-trace"  # the reasoning trace IS the Target B deliverable
    turns = (
        Turn(role="system", content=SOLVE_SYSTEM),
        Turn(role="user", content=prob.question),
        Turn(role="assistant", content=f"Answer: {prob.truth}", loss=True, thinking=reply.thinking),
    )
    row = SFTRow(
        id=f"B-{prob.family}-{pseed}", target="B", family=prob.family, dialect="n/a", turns=turns,
        provenance=Provenance(
            generator="denominator_reasoning", method="teacher-distilled",
            teacher=teacher_id, licence="Apache-2.0", seed=pseed,
        ),
        verification=Verification(
            engine="duckdb", truth=prob.truth, engine_result=prob.truth, agrees=True
        ),
        tags=prob.tags,
    )
    return row, None


def generate(
    *, teacher: Teacher, seed: int = 11, reps: int = 1, families: list[str] | None = None,
) -> tuple[list[SFTRow], dict]:
    fams = families or list(_TRAPS)
    teacher_id = getattr(teacher, "model", "stub")
    report: dict = {"emitted": 0, "rejected": 0, "by_family": {}, "reasons": {}, "buggy": 0}
    rows: list[SFTRow] = []
    for rep in range(reps):
        for fam in fams:
            # Stable seed (no hash() -- string hashing is randomized per process).
            pseed = seed + rep * 104729 + list(_TRAPS).index(fam) * 131
            prob = _TRAPS[fam](pseed)
            if not _self_verify(prob):
                report["buggy"] += 1
                continue
            row, reason = process_problem(prob, teacher, pseed, teacher_id)
            if row is None:
                report["rejected"] += 1
                report["reasons"][reason] = report["reasons"].get(reason, 0) + 1
                continue
            rows.append(row)
            report["emitted"] += 1
            report["by_family"][fam] = report["by_family"].get(fam, 0) + 1
    return rows, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Target B: denominator-reasoning SFT rows (teacher)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--families", default="")
    ap.add_argument("--base-url", default="http://localhost:18080/v1")
    ap.add_argument("--model", default="gpt-oss")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    teacher = HTTPTeacher(base_url=args.base_url, model=args.model)
    rows, report = generate(
        teacher=teacher, seed=args.seed, reps=args.reps,
        families=[f for f in args.families.split(",") if f] or None,
    )
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(row_to_dict(r)) + "\n")
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
    print(f"emitted: {report['emitted']}  rejected: {report['rejected']}  buggy: {report['buggy']}")
    print(f"by family: {report['by_family']}  reasons: {report['reasons']}")
    for r in rows[: args.sample]:
        print("\n---", r.id, "---")
        for t in r.turns:
            if t.thinking:
                print(f"[think] {t.thinking[:400]}")
            print(f"[{t.role}] {t.content}")


if __name__ == "__main__":
    main()
