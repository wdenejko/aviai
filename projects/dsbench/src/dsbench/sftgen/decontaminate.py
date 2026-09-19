"""Decontamination gate -- keep the fine-tune data disjoint from the benchmark that scores it.

dsbench is the before/after metric (ADR-001/003). A training row that overlaps a problem prompt, the
aviation schema, or a problem's ground-truth answer would turn a post-fine-tune score gain into
memorisation. This gate (ADR-004 protocol) rejects a row on any of:

  1. 13-gram overlap with any of the live agentic problem prompts/titles (derived from the loader,
     so it tracks the problem set automatically);
  2. an aviation schema identifier (table/db/column names the targeted generators never emit -- a
     belt-and-braces check against leakage from the breadth/replay buckets);
  3. a DISTINCTIVE ground-truth answer (e.g. 44.1, 0.7212, 6538 -- decimals/large ints unlikely to
     occur by chance; generic small ints are deliberately NOT listed, to avoid false rejections).

Runs over a mixture JSONL (raw SFTRow rows OR rendered chat records -- both carry the same text) and
writes the clean rows plus a report (scanned, rejected by rule, examples) that is committed with the
run (ADR-003 reproducibility contract).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any

from dsbench.agentic.loader import load_problems
from dsbench.sftgen.schema import row_from_dict

# Distinctive aviation schema identifiers (matched case-insensitively as substrings). Generic words
# (cancelled, distance, origin) are excluded -- not dsbench-unique.
AVIATION_IDENTIFIERS: tuple[str, ...] = (
    "aviation.flights", "aviation.metar", "aviation.taf", "aviation.notam",
    "crsdeptime", "crsarrtime", "arrdelayminutes", "depdelayminutes",
    "lateaircraftdelay", "carrierdelay", "weatherdelay", "nasdelay", "securitydelay",
    "reporting_airline", "flight_number_reporting_airline", "arrdel15", "taxiout",
)
# Only DISTINCTIVE answers -- decimals / 4-digit ints unlikely to appear by chance. Generic small
# ints (13, 6, 18) are deliberately omitted to avoid false rejections.
AVIATION_ANSWERS: tuple[str, ...] = ("44.1", "0.7212", "6538")

_NGRAM = 13


@dataclass
class DenyList:
    grams: set[str]
    identifiers: tuple[str, ...]
    answers: tuple[str, ...]
    problem_ids: list[str] = field(default_factory=list)


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (numbers kept, so numeric grams count)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int = _NGRAM) -> set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def build_denylist(n: int = _NGRAM) -> DenyList:
    grams: set[str] = set()
    ids: list[str] = []
    for p in load_problems():
        ids.append(p.id)
        grams |= _ngrams(_tokens(f"{p.title}\n{p.prompt}"), n)
    return DenyList(grams=grams, identifiers=AVIATION_IDENTIFIERS,
                    answers=AVIATION_ANSWERS, problem_ids=ids)


def scan_text(text: str, deny: DenyList, n: int = _NGRAM) -> dict | None:
    """Return a rejection reason dict if the text is contaminated, else None."""
    low = text.lower()
    for ident in deny.identifiers:
        if ident in low:
            return {"rule": "schema-identifier", "hit": ident}
    for ans in deny.answers:
        # digit-boundary match so "44.1" hits "44.1" but not "144.1"/"44.12" (the tokenizer would
        # split the decimal, so match the raw text instead).
        if re.search(rf"(?<!\d){re.escape(ans)}(?!\d)", low):
            return {"rule": "numeric-answer", "hit": ans}
    toks = _tokens(text)
    overlap = _ngrams(toks, n) & deny.grams
    if overlap:
        return {"rule": "13-gram", "hit": next(iter(overlap))}
    return None


def _row_text(obj: dict[str, Any]) -> str:
    """Extract scannable text from either a raw SFTRow dict or a rendered chat record."""
    if "messages" in obj:
        return "\n".join(m.get("content", "") for m in obj["messages"])
    if "turns" in obj:
        return row_from_dict(obj).text_blob()
    return json.dumps(obj)


def scan_file(in_path: str, out_path: str | None = None) -> dict:
    deny = build_denylist()
    report: dict[str, Any] = {
        "problems": len(deny.problem_ids), "denylist_grams": len(deny.grams),
        "scanned": 0, "clean": 0, "rejected": 0,
        "by_rule": {}, "examples": [],
    }
    out = open(out_path, "w") if out_path else None
    try:
        with open(in_path) as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                report["scanned"] += 1
                reason = scan_text(_row_text(obj), deny)
                if reason is None:
                    report["clean"] += 1
                    if out:
                        out.write(json.dumps(obj) + "\n")
                else:
                    report["rejected"] += 1
                    report["by_rule"][reason["rule"]] = report["by_rule"].get(reason["rule"], 0) + 1
                    if len(report["examples"]) < 10:
                        report["examples"].append(reason)
    finally:
        if out:
            out.close()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Decontaminate an SFT mixture against dsbench")
    ap.add_argument("input", help="mixture JSONL (raw SFTRow rows or rendered chat records)")
    ap.add_argument("--out", default="", help="write clean rows here (omit to only report)")
    ap.add_argument("--report", default="", help="write the report JSON here")
    ap.add_argument("--fail-on-hit", action="store_true", help="exit 1 if any row is rejected")
    args = ap.parse_args()

    report = scan_file(args.input, args.out or None)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)

    print(f"problems: {report['problems']}  denylist 13-grams: {report['denylist_grams']}")
    print(f"scanned: {report['scanned']}  clean: {report['clean']}  rejected: {report['rejected']}")
    print(f"by rule: {report['by_rule']}")
    for ex in report["examples"][:8]:
        print(f"  REJECT [{ex['rule']}] {ex['hit'][:80]}")
    if args.fail_on_hit and report["rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
