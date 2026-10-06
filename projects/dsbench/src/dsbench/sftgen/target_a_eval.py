"""Target A's direct test (ADR-004 Revision 2): the base and the Revision 2 adapter answer the same
held-out Target A prompts, paired, and every reply is checked on its dialect's sandboxed engine.
(The module isn't target_a_test.py: pytest collects *_test.py files as tests.)

WHY: the gate (the thinking-on mini-battery, 2026-10-06) found no regression and no gain, but it
measures IFEval, BFCL, BIRD and HumanEval+, and none of them asks for a SQL dialect's date
conventions, which is what Target A trains. This test asks for them directly:
- ClickHouse's weekdays (ISO: Monday 1 .. Sunday 7). The base gets them wrong: it believes Sunday
  is 1, as MySQL numbers it. The prefill pilot's plain sampling verified 7 of 96 ClickHouse weekday
  and weekend replies (reports/gate-evals/20261001-target-a-prefill-pilot.md). These are the target
  cells (TARGET_CELLS).
- The 14 other cells, which the base mostly gets right and the adapter must not break. The
  generation's plain replies verified 222 of 252 (ADR-004, "The generation"). Target A taught ISO
  for ClickHouse. A model that learned "ISO everywhere" would now fail DuckDB's, PostgreSQL's and
  MySQL's weekdays.

What is held out is the prompt. The model sees a table's schema and the question, never its rows.
So a new table in a training domain asks a training prompt again: the training drew 124 of
ClickHouse's 126 weekday prompts and all 18 of its weekend ones. The test's tables come from six
domains no training row names (synth.held_out_domain_names), with new table names, timestamp
columns and row nouns. `items` checks that against the mixture. The question wordings are the
training ones, so the test measures the conventions on unseen tables, not on unseen phrasing. The
held-out probe, on aviation data, measures the far transfer.

The test is paired. Each item is answered once by each state on one server: battery_server.sh with
the adapter loaded, every request naming its scale (reasoning_pilot generate --lora-scale). Both
states use the same sampling and the same per-item seed, so an item's two replies differ only in
the adapter. The sampling is the generation's (min_p 0.05), so the base state can be read against
the generation's own rates. The budget is the mini-battery's 12,288 tokens, which a served reply
gets.

    uv run python -m dsbench.sftgen.target_a_eval items \\
        --out data/sft/rev2_target_a_test/items.jsonl \\
        --report data/sft/rev2_target_a_test_manifest.json --mixture data/sft/rev2_mixture.jsonl
    # the box: one window, "ta_test:base ta_test:adapter" (patches/README.md, the Target A test)
    uv run python -m dsbench.sftgen.target_a_hints verify --items <items> \\
        --gen <run>/gen/ta_test.base.jsonl --out <run>/verified/ta_test.base.jsonl   # and .adapter
    uv run python -m dsbench.sftgen.target_a_eval compare --items <items> \\
        --base <run>/verified/ta_test.base.jsonl --adapter <run>/verified/ta_test.adapter.jsonl \\
        --out <run>/compare.json

Building the items needs the sandbox's four engines, since every item's gold SQL is checked against
its truth: docker compose -f sandbox/docker-compose.yml up -d --build clickhouse postgres mysql
duckdb.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

from dsbench.sftgen import synth
from dsbench.sftgen.reasoning_pilot import SAMPLING
from dsbench.sftgen.target_a_hints import DIALECTS, FAMILIES, ROWS_PER_TABLE, make_items

TEST_SEED = 20261006
# Every cell gets 2 tables per held-out domain, so 12 items a cell. Fourteen cells give 168 items,
# which is enough to see a cell family collapse: a model that took ISO for every dialect would lose
# most of DuckDB's, PostgreSQL's and MySQL's 72 weekday and weekend items.
TEST_CELL_REPS = 2
# ClickHouse's weekday and weekend cells get more tables on top: 36 and 24 items, which measure
# the adapter's rate there to about +-10 points. The base gets about 1 in 14 of them right.
TEST_EXTRA_REPS = {"weekday-numbering": 4, "weekend-flag": 2}
TEST_MAX_TOKENS = 12288  # the mini-battery's budget (battery/mini.BUDGET): what a served reply gets
TARGET_CELLS = ("clickhouse/weekday-numbering", "clickhouse/weekend-flag")


def _cell(item: dict) -> str:
    return f"{item['meta']['dialect']}/{item['meta']['family']}"


def eval_item(plain: dict) -> dict:
    """A test item from a plain Target A item (target_a_hints.make_items): its own id and pool, so
    no record of it can pass for a training reply, and the test's budget."""
    meta = {k: v for k, v in plain["meta"].items() if k not in ("hinted", "teacher")}
    return {**plain, "id": f"targetA_test:{plain['verify']['row_id']}", "pool": "targetA_test",
            "max_tokens": TEST_MAX_TOKENS,
            "meta": {**meta, "held_out": True,
                     "teacher": "none: a test item, answered by the base and by the adapter; "
                                "it never trains"}}


def build_test_items(*, seed: int = TEST_SEED, cell_reps: int = TEST_CELL_REPS,
                     extra_reps: dict[str, int] | None = None, n: int = ROWS_PER_TABLE
                     ) -> tuple[list[dict], dict]:
    """(items, report): every cell on the held-out domains, and ClickHouse's weekday and weekend
    cells again from their own draws. Each draw has its own seed, so no two draws share a table."""
    from dsbench.sftgen.dialect_conventions import generate
    from dsbench.sftgen.schema import row_to_dict

    extra_reps = TEST_EXTRA_REPS if extra_reps is None else extra_reps
    domains = synth.held_out_domain_names()
    draws = [("cells", seed, cell_reps, list(DIALECTS), list(FAMILIES))]
    draws += [(f"clickhouse/{family}", seed + 1 + k, reps, ["clickhouse"], [family])
              for k, (family, reps) in enumerate(extra_reps.items()) if reps]
    items: list[dict] = []
    report: dict = {"seed": seed, "rows_per_table": n, "domains": domains, "draws": []}
    for name, draw_seed, reps, dialects, families in draws:
        rows, generated = generate(seed=draw_seed, reps=reps, n=n, dialects=dialects,
                                   families=families, thinking_frac=0.0, domains=domains)
        missing = set(dialects) - set(generated["engines"])
        if missing:
            raise RuntimeError(f"no engine for {sorted(missing)}: bring the sandbox up")
        report["draws"].append({"name": name, "seed": draw_seed, "reps": reps,
                                "rows": len(rows), "generator_rejected": generated["rejected"]})
        items += [eval_item(make_items(row, n, hinted=False)[0]) for row in map(row_to_dict, rows)]
    ids = Counter(i["id"] for i in items)
    repeated = [i for i, k in ids.items() if k > 1]
    if repeated:
        raise RuntimeError(f"{len(repeated)} items share an id, e.g. {repeated[0]}")
    report["items"] = len(items)
    report["items_by_cell"] = dict(sorted(Counter(map(_cell, items)).items()))
    prompts = {(_cell(i), json.dumps(i["messages"])) for i in items}
    report["prompts_by_cell"] = dict(sorted(Counter(c for c, _ in prompts).items()))
    report["max_tokens"] = TEST_MAX_TOKENS
    report["sampling"] = SAMPLING
    return items, report


def _prompt(messages: list[dict]) -> str:
    return json.dumps([(m["role"], m["content"]) for m in messages if m["role"] != "assistant"])


def training_overlap(items: list[dict], mixture_path: Path) -> dict:
    """What the test shares with the training mixture: the test's prompts that a training row asks
    (none, by construction), and the training rows that name a held-out table or timestamp column
    anywhere. A SQL row may name one by chance; such a row doesn't teach a Target A answer."""
    names = set()
    for name in synth.held_out_domain_names():
        domain = synth.build(name, 0, 1)
        names |= {domain.name, domain.ts_col}
    asked: set[str] = set()
    naming: Counter[str] = Counter()
    rows = 0
    for line in mixture_path.open():
        rows += 1
        row = json.loads(line)
        asked.add(_prompt(row["messages"]))
        text = line.lower()
        naming.update(name for name in names if name in text)
    shared = sum(_prompt(i["messages"]) in asked for i in items)
    return {"mixture": str(mixture_path), "mixture_rows": rows, "prompts_in_training": shared,
            "rows_naming_a_held_out_name": {name: naming[name] for name in sorted(names)}}


# --- compare -------------------------------------------------------------------------------------


def mcnemar_p(b: int, c: int) -> float:
    """The exact two-sided McNemar test: of the b + c items on which the states disagree, how
    unlikely a split this uneven is if neither state is better (a fair coin per item)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _records(path: Path) -> dict[str, dict]:
    """A verified file's records by id, the last for an id winning. `target_a_hints.verify` checks
    every line of a generation file, and a request that failed stays there beside its retry, which
    a resumed run appends after it."""
    return {r["id"]: r for r in map(json.loads, path.open())}


def _pair_stats(pairs: list[tuple[bool, bool]]) -> dict:
    base_only = sum(b and not a for b, a in pairs)
    adapter_only = sum(a and not b for b, a in pairs)
    n = len(pairs)
    return {"n": n, "base": sum(b for b, _ in pairs), "adapter": sum(a for _, a in pairs),
            "base_only": base_only, "adapter_only": adapter_only,
            "p_mcnemar": round(mcnemar_p(base_only, adapter_only), 6)}


def _median(values: list[int]) -> float | None:
    return st.median(values) if values else None


def compare(items: dict[str, dict], base: dict[str, dict], adapter: dict[str, dict]) -> dict:
    """Base against adapter, item by item. An item counts when it is verified: it finished, with
    reasoning, as one ```sql block, and the block returns the truth (target_a_hints.check_reply).
    Only items both states answered are paired; an item whose table didn't rebuild to its truth
    (`rebuild_mismatch`) is the data's fault and counts for neither."""
    paired = [i for i in items if i in base and i in adapter]
    mismatched = [i for i in paired if "rebuild_mismatch" in (base[i]["check"]["status"],
                                                              adapter[i]["check"]["status"])]
    paired = [i for i in paired if i not in mismatched]
    ok = {i: (base[i]["check"]["status"] == "verified",
              adapter[i]["check"]["status"] == "verified") for i in paired}
    cells: dict[str, list[str]] = {}
    for i in paired:
        cells.setdefault(_cell(items[i]), []).append(i)
    groups = {"target": [i for c in TARGET_CELLS for i in cells.get(c, [])],
              "other cells": [i for c, ids in cells.items() if c not in TARGET_CELLS
                              for i in ids]}
    for family in FAMILIES:
        groups[f"other cells: {family}"] = [i for i in groups["other cells"]
                                            if items[i]["meta"]["family"] == family]
    for dialect in DIALECTS:
        groups[f"other cells: {dialect}"] = [i for i in groups["other cells"]
                                             if items[i]["meta"]["dialect"] == dialect]
    groups = {name: ids for name, ids in groups.items() if ids}
    states = {"base": base, "adapter": adapter}
    out: dict = {
        "items": len(items), "paired": len(paired), "rebuild_mismatch": len(mismatched),
        "unpaired": {s: sum(i in recs for i in items) - len(paired) - len(mismatched)
                     for s, recs in states.items()},
        "groups": {name: _pair_stats([ok[i] for i in ids]) for name, ids in groups.items()},
        "cells": {cell: _pair_stats([ok[i] for i in sorted(ids)])
                  for cell, ids in sorted(cells.items())},
        "statuses": {s: dict(Counter(recs[i]["check"]["status"] for i in paired))
                     for s, recs in states.items()},
    }
    # ClickHouse's Sunday = 1 (target_a_hints.doubts): the base's belief, and in the adapter's
    # training traces a doubt that the assembler dropped. Its rate in the target cells, per state.
    target = groups.get("target", [])
    out["target_states_sunday_1"] = {
        s: sum(recs[i]["check"].get("doubts", 0) > 0 for i in target) for s, recs in states.items()}
    out["reasoning_tokens_median"] = {
        s: {g: _median([recs[i]["reasoning_tokens"] for i in groups.get(g, [])
                        if recs[i].get("reasoning_tokens") is not None])
            for g in ("target", "other cells")}
        for s, recs in states.items()}
    return out


def markdown(result: dict) -> str:
    """The comparison as the report's tables."""
    lines = ["| group | n | base | adapter | base only | adapter only | McNemar p |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, g in [*result["groups"].items(), *result["cells"].items()]:
        lines.append(f"| {name} | {g['n']} | {g['base']} | {g['adapter']} | {g['base_only']} | "
                     f"{g['adapter_only']} | {g['p_mcnemar']:.3g} |")
    return "\n".join(lines) + "\n"


# --- CLI -----------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("items", help="the test's items, on the held-out domains")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--report", type=Path, required=True)
    b.add_argument("--mixture", type=Path, required=True,
                   help="the training mixture: no test prompt may be one of its prompts")
    c = sub.add_parser("compare", help="base against adapter, item by item")
    c.add_argument("--items", type=Path, required=True)
    c.add_argument("--base", type=Path, required=True, help="target_a_hints verify's records")
    c.add_argument("--adapter", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "compare":
        items = {i["id"]: i for i in map(json.loads, args.items.open())}
        result = compare(items, _records(args.base), _records(args.adapter))
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(markdown(result))
        print(json.dumps({k: result[k] for k in ("paired", "statuses", "target_states_sunday_1",
                                                 "reasoning_tokens_median")}, indent=1))
        return

    items, report = build_test_items()
    overlap = training_overlap(items, args.mixture)
    if overlap["prompts_in_training"]:
        raise SystemExit(f"{overlap['prompts_in_training']} test prompts are training prompts")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    args.report.write_text(json.dumps({
        **report, "training_overlap": overlap, "out": str(args.out),
        "out_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("items", "items_by_cell")}, indent=1))
    print(json.dumps(overlap, indent=1))


if __name__ == "__main__":
    main()
