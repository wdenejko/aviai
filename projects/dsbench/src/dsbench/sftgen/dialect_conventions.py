"""Target A generator -- execution-verified SQL-dialect date/time convention rows (ADR-004).

Composes synthetic non-aviation domains x convention traps x reachable dialect engines, executes
the candidate SQL against each REAL engine, and emits an SFT row ONLY when the engine's scalar
equals the independent pandas truth. Teacher-free, so every emitted row is Apache-2.0-clean.

Run (project env):
    uv run python -m dsbench.sftgen.dialect_conventions --reps 20 --out out.jsonl --report rep.json
    uv run python -m dsbench.sftgen.dialect_conventions --dialects duckdb --reps 3   # quick, no CH

A row that fails verification is never emitted; it is counted in the report's rejections so a wrong
convention mapping (or an engine quirk) is visible rather than silent.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from dsbench.sftgen import synth
from dsbench.sftgen.conventions import conventions_by_family
from dsbench.sftgen.engines import available_engines
from dsbench.sftgen.schema import (
    Provenance,
    SFTRow,
    Turn,
    Verification,
    row_to_dict,
)


def _assistant_content(sql: str, result: Any) -> str:
    return f"```sql\n{sql}\n```\n\nAnswer: {result}"


def _build_row(*, domain, dialect, conv, params, sql, truth, result, seed, thinking_on) -> SFTRow:
    trace = conv.thinking(dialect, params) if thinking_on else None
    turns = (
        Turn(role="system", content=conv.system(domain, dialect)),
        Turn(role="user", content=conv.question(domain, params)),
        Turn(
            role="assistant", content=_assistant_content(sql, result), loss=True, thinking=trace
        ),
    )
    return SFTRow(
        id=f"A-{conv.family}-{dialect}-{domain.name}-{seed}",
        target="A", family=conv.family, dialect=dialect, turns=turns,
        provenance=Provenance(
            generator="dialect_conventions", method="own-generated", teacher=None,
            licence="Apache-2.0", seed=seed,
        ),
        verification=Verification(
            engine=dialect, truth=truth, engine_result=result, agrees=True
        ),
        tags=conv.tags,
    )


def generate(
    *, seed: int = 7, reps: int = 1, n: int = 4000,
    dialects: list[str] | None = None, families: list[str] | None = None,
    thinking_frac: float = 0.4, sink=None,
) -> tuple[list[SFTRow], dict]:
    # `sink(row)`, if given, is called as each row is verified -- the caller writes+flushes it so a
    # long run is crash-safe (partial output survives). The returned list still powers the report;
    # rows are tiny, so keeping both is cheap.
    rng = np.random.default_rng(seed)
    conv_map = conventions_by_family()
    convs = [conv_map[f] for f in (families or list(conv_map))]
    domains = synth.domain_names()
    engines = available_engines(only=dialects)

    report: dict[str, Any] = {
        "engines": [e.name for e in engines], "emitted": 0, "rejected": 0,
        "by_dialect": {}, "by_family": {}, "thinking_rows": 0, "rejections": [],
    }
    rows: list[SFTRow] = []
    if not engines:
        return rows, report

    for rep in range(reps):
        for di, dname in enumerate(domains):
            dseed = seed + di * 100003 + rep * 7919
            domain = synth.build(dname, dseed, n)
            params = {c.family: c.params(rng) for c in convs}  # fixed across dialects
            for eng in engines:
                eng.setup()
                try:
                    eng.load(domain.name, domain.df)
                    for c in convs:
                        p = params[c.family]
                        truth = c.truth(domain, p)
                        sql = c.sql(domain, eng.name, p)
                        try:
                            result = eng.scalar(sql)
                        except Exception as ex:  # noqa: BLE001
                            report["rejected"] += 1
                            report["rejections"].append(
                                {"dialect": eng.name, "family": c.family,
                                 "reason": f"engine error: {str(ex)[:120]}"}
                            )
                            continue
                        if truth != result:
                            report["rejected"] += 1
                            report["rejections"].append(
                                {"dialect": eng.name, "family": c.family,
                                 "reason": f"mismatch: truth={truth} engine={result}"}
                            )
                            continue
                        thinking_on = bool(rng.random() < thinking_frac)
                        row = _build_row(
                            domain=domain, dialect=eng.name, conv=c, params=p, sql=sql,
                            truth=truth, result=result, seed=dseed, thinking_on=thinking_on,
                        )
                        rows.append(row)
                        if sink is not None:
                            sink(row)
                        report["emitted"] += 1
                        report["thinking_rows"] += int(thinking_on)
                        report["by_dialect"][eng.name] = report["by_dialect"].get(eng.name, 0) + 1
                        report["by_family"][c.family] = report["by_family"].get(c.family, 0) + 1
                finally:
                    eng.teardown()
    return rows, report


def _js(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def main() -> None:
    ap = argparse.ArgumentParser(description="Target A: dialect date/time convention SFT rows")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reps", type=int, default=1, help="synthetic tables per domain")
    ap.add_argument("--n", type=int, default=4000, help="rows per synthetic table")
    ap.add_argument("--dialects", default="", help="comma list; default = all reachable")
    ap.add_argument("--families", default="", help="comma list; default = all")
    ap.add_argument("--thinking-frac", type=float, default=0.4)
    ap.add_argument("--out", default="", help="raw SFTRow JSONL (source of truth)")
    ap.add_argument("--report", default="", help="run report JSON")
    ap.add_argument("--sample", type=int, default=0, help="print N rows to stdout")
    args = ap.parse_args()

    out_fh = open(args.out, "w") if args.out else None  # closed in the finally below

    def _sink(row) -> None:  # write+flush each row so a long run is crash-safe
        out_fh.write(json.dumps(row_to_dict(row), default=_js) + "\n")
        out_fh.flush()

    try:
        rows, report = generate(
            seed=args.seed, reps=args.reps, n=args.n,
            dialects=[d for d in args.dialects.split(",") if d] or None,
            families=[f for f in args.families.split(",") if f] or None,
            thinking_frac=args.thinking_frac, sink=_sink if out_fh else None,
        )
    finally:
        if out_fh:
            out_fh.close()

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2, default=_js)

    print(f"engines: {report['engines']}")
    print(f"emitted: {report['emitted']}  rejected: {report['rejected']}  "
          f"thinking: {report['thinking_rows']}")
    print(f"by dialect: {report['by_dialect']}")
    print(f"by family:  {report['by_family']}")
    for rej in report["rejections"][:8]:
        print(f"  REJECT [{rej['dialect']}/{rej['family']}] {rej['reason']}")
    for r in rows[: args.sample]:
        print("\n--- sample row", r.id, "---")
        for t in r.turns:
            if t.thinking:
                print(f"[{t.role} · think] {t.thinking}")
            print(f"[{t.role}] {t.content}")


if __name__ == "__main__":
    main()
