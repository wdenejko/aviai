"""Target B generator -- conditional-population / denominator reasoning (ADR-004).

The measured gap: both base and Ornith divide by the WRONG base on ratio questions
(`da_delay_attribution` 0/5, two different stable wrong answers -> a real reasoning gap). This
generator teaches the transferable skill on NON-aviation data, as execution-filtered teacher traces:

  1. build a synthetic table whose ratio has ONE defensible answer, and compute that truth two ways
     (pandas + a reference SQL on DuckDB) -- if they disagree the problem is buggy and is skipped;
  2. ask a licence-clean teacher (gpt-oss / DeepSeek / Qwen-class) to reason it out, give a number;
  3. KEEP the teacher's trace only if its answer matches the verified truth AND it exposed a trace
     -- the reasoning is the point of Target B.

Trap families:
  * `conditional-null-share` -- cause columns populated only under a condition (null = N/A, not 0);
    the share denominator is the sum of the causes, not a grand total (da_delay_attribution analog).
  * `pooled-vs-mean-rate` -- the overall rate is pooled sum/sum, not the mean of per-group rates
    (da_weighted_ontime analog; uneven groups make the two answers diverge).
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


def _conditional_null_share(seed: int, n: int = 5000) -> BProblem:
    rng = np.random.default_rng(seed)
    breached = rng.random(n) < 0.30  # causes populated ONLY for breached incidents
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
        "Table `incidents(incident_id, region, breached, cause_a, cause_b, cause_c, cause_d, "
        "cause_e)`. The five cause columns hold minutes and are populated ONLY for incidents that "
        "breached SLA (breached = 1); otherwise they are null. Of the total minutes summed across "
        "all FIVE cause columns, what percentage is attributable to cause_c? Reply rounded to 1 "
        "decimal."
    )
    return BProblem("conditional-null-share", "incidents", df, q, truth, ref_sql, 0.2,
                    ("ratio", "null", "denominator"))


def _pooled_vs_mean_rate(seed: int, n: int = 6000) -> BProblem:
    rng = np.random.default_rng(seed)
    # Uneven group sizes + per-group success probabilities so pooled != mean-of-group-rates.
    stores = [f"store_{i}" for i in range(6)]
    weights = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
    probs = rng.uniform(0.55, 0.95, size=len(stores))
    store_idx = rng.choice(len(stores), size=n, p=weights)
    success = (rng.random(n) < probs[store_idx]).astype("int64")
    df = pd.DataFrame({
        "attempt_id": np.arange(1, n + 1, dtype="int64"),
        "store": np.array(stores)[store_idx],
        "success": success,
    })
    truth = round(float(df["success"].mean()), 4)  # pooled = sum(success)/count()
    ref_sql = "SELECT sum(success) * 1.0 / count(*) FROM attempts"
    q = (
        "Table `attempts(attempt_id, store, success)` where success is 1/0. What is the OVERALL "
        "success rate across ALL attempts (total successes divided by total attempts)? Note the "
        "stores have very different attempt volumes. Reply as a fraction rounded to 4 decimals."
    )
    return BProblem("pooled-vs-mean-rate", "attempts", df, q, truth, ref_sql, 0.0005,
                    ("ratio", "pooled", "weighting"))


_TRAPS = {"conditional-null-share": _conditional_null_share,
          "pooled-vs-mean-rate": _pooled_vs_mean_rate}


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
