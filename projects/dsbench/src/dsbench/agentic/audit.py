"""Audit saved dsbench and probe runs for access to the labels the grader holds back.

    uv run python -m dsbench.agentic.audit OUT.json reports/agentic-runs/*.json

A run since 2026-10-01 can't reach the withheld labels (`access`), and the guard fails one whose
tool calls try. This reads the runs from before: the native loop's and pi's result JSON. It
writes each flagged run to OUT.json with a summary, and prints both. The flags:
- `named_withheld`: a tool call names a table the problem's setup writes for the grader. The
  names are taken from the setup itself (`setup_tables`), so an agent's own `X_test_full` array
  is not one;
- `saw_withheld_listed`: a tool's output listed such a table, from SHOW TABLES say;
- `joined_test_to_flights`: a flight problem's test table joined to aviation.flights, whose rows
  hold the test flights' outcomes; `read_flights`: aviation.flights read without that join;
- `notam_test_labels`: a statement read categories from aviation.notam beyond the train split,
  as `aggregate` (counts) or `rows`.

reports/gate-evals/20261001-dsbench-withheld-labels.md has the audit of every run saved before the
fix.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dsbench.agentic.access import WITHHELD_TABLE, tool_call_texts
from dsbench.agentic.schema import GradeContext

_FLIGHT_PROBLEMS = ("ds_cancel_predict", "ds_delay_predict", "ds_taxi_regression")


class _RecordingClient:
    """Enough of clickhouse_connect for a problem's setup: the tables it creates."""

    def __init__(self) -> None:
        self.tables: list[str] = []

    def command(self, sql: str) -> None:
        words = sql.split()
        if words[:2] == ["CREATE", "TABLE"]:
            self.tables.append(words[2].split(".")[-1])

    def insert_df(self, table: str, df: Any) -> None:
        pass


def setup_tables(problem: Any) -> list[str]:
    """The tables a problem's setup creates, read without ClickHouse."""
    client = _RecordingClient()
    if problem.setup:
        problem.setup(GradeContext(client, "audit"))
    return client.tables


def _outputs(trajectory: list | None) -> str:
    out = []
    for message in trajectory or []:
        if message.get("role") == "tool":
            out.append(message.get("content") or "")
        elif message.get("role") == "toolResult":  # pi
            out.append(" ".join(b.get("text", "") for b in message.get("content") or []
                                if isinstance(b, dict)))
    return "\n".join(out)


def _notam_label_reads(text: str) -> list[str]:
    kinds = set()
    for stmt in re.findall(r"SELECT[^\"']*?FROM\s+aviation\.notam[^\"\n]*", text, re.I | re.S):
        if not re.search(r"category|label", stmt, re.I):
            continue
        if re.search(r"split\s*=\s*[\\'\"]*train", stmt):  # quotes may be escaped, twice
            continue
        kinds.add("aggregate" if re.search(r"count\(|group\s+by", stmt, re.I) else "rows")
    return sorted(kinds)


def audit_result(result: dict, tables: list[str]) -> dict:
    """The flags of one run of a problem whose setup creates `tables`."""
    pid = result["id"]
    calls = "\n".join(tool_call_texts(result.get("trajectory")))
    seen = _outputs(result.get("trajectory"))
    withheld = [t for t in tables if WITHHELD_TABLE.search(t)]
    flags: dict[str, Any] = {}
    named = [t for t in withheld if re.search(rf"\b{t}\b", calls)]
    if named:
        flags["named_withheld"] = named
    if any(re.search(rf"\b{t}\b", seen) for t in withheld):
        flags["saw_withheld_listed"] = True
    test = next((t for t in tables if t.endswith("_test")), None)
    if pid in _FLIGHT_PROBLEMS and test:
        if re.search(rf"\b{test}\b[^\n]*join[^\n]*aviation\.flights|"
                     rf"aviation\.flights[^\n]*join[^\n]*\b{test}\b", calls, re.I):
            flags["joined_test_to_flights"] = True
        elif "aviation.flights" in calls:
            flags["read_flights"] = True
    if pid == "ds_notam_classify":
        kinds = _notam_label_reads(calls)
        if kinds:
            flags["notam_test_labels"] = kinds
    return flags


def audit(paths: list[str], problems: dict[str, Any]) -> dict:
    tables = {pid: setup_tables(p) for pid, p in problems.items()}
    runs = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        meta = data.get("meta", {})
        for index, result in enumerate(data.get("results", [])):
            flags = audit_result(result, tables.get(result["id"], []))
            runs.append({"file": Path(path).name, "model": meta.get("model"),
                         "harness": meta.get("harness", "native"), "index": index,
                         "id": result["id"], "status": result["status"],
                         "passed": result["passed"], "flags": flags})
    summary: dict[str, Counter] = defaultdict(Counter)
    for run in runs:
        for flag in run["flags"]:
            summary[flag]["runs"] += 1
            summary[flag]["passed"] += run["passed"]
    return {"runs": len(runs), "files": len(paths),
            "summary": {k: dict(v) for k, v in sorted(summary.items())},
            "flagged": [r for r in runs if r["flags"]]}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    from dsbench.agentic.loader import load_problems
    from dsbench.sftgen.probe.tasks import PROBE_TASKS

    problems = {p.id: p for p in load_problems() + PROBE_TASKS}
    report = audit(sys.argv[2:], problems)
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1) + "\n")
    print(f"{report['runs']} runs in {report['files']} files")
    print(json.dumps(report["summary"], indent=1))
    for r in report["flagged"]:
        print(f"{r['file']:44} #{r['index']:<3} {r['id']:22} {r['model']!s:7} {r['status']:12} "
              f"passed={r['passed']!s:5} {r['flags']}")


if __name__ == "__main__":
    main()
