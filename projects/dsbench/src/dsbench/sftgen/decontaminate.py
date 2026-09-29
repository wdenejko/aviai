"""Decontamination gate -- keep the fine-tune data disjoint from the benchmark that scores it.

dsbench is the before/after metric (ADR-001/003). A training row that overlaps a problem prompt, the
aviation schema, or a problem's ground-truth answer would turn a post-fine-tune score gain into
memorisation. This gate (ADR-004 protocol) rejects a row on any of:

  1. 13-gram overlap with any of the live agentic problem prompts/titles (derived from the loader,
     so it tracks the problem set automatically);
  2. an aviation schema identifier (table/db/column names the targeted generators never emit -- a
     belt-and-braces check against leakage from the breadth/replay buckets);
  3. a DISTINCTIVE ground-truth answer (e.g. 44.1, 0.7212, 6538 -- decimals/large ints unlikely to
     occur by chance; generic small ints are deliberately NOT listed, to avoid false rejections);
  4. with `--battery-items` (ADR-004 Revision 2): at least a fifth of a battery item's 13-grams, or
     the whole of a short item, the threshold `battery/contamination.py` calls "strong".

Why rule 4 rejects on a fifth and rule 1 on a single 13-gram: dsbench's prompts are distinctive
aviation text, so any shared 13-gram is a leak. The battery is eight public benchmarks full of
generic code and prose. Measured on the Gate-2 mixture, 44 items shared a 13-gram with it, all of
them generic (counting sequences, textbook Fibonacci, the definition of a subsequence), and one
reached the fifth (reports/gate-evals/20260924-gate2-battery.md). Rows below the line are kept but
listed in the report's audit, so the owner sees every overlap.

Runs over a mixture JSONL (raw SFTRow rows OR rendered chat records -- both carry the same text) and
writes the clean rows plus a report (scanned, rejected by rule, examples) that is committed with the
run (ADR-003 reproducibility contract).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dsbench.agentic.loader import load_problems
from dsbench.battery.contamination import STRONG_FRACTION, BatteryIndex, build_index, shared_units
from dsbench.battery.items import load_items
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
    battery: BatteryIndex | None = None


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (numbers kept, so numeric grams count)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int = _NGRAM) -> set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def build_denylist(n: int = _NGRAM, battery_items: str | None = None) -> DenyList:
    """dsbench's denylist; with `battery_items` (a battery run's `items/` dir), rule 4 as well."""
    grams: set[str] = set()
    ids: list[str] = []
    for p in load_problems():
        ids.append(p.id)
        grams |= _ngrams(_tokens(f"{p.title}\n{p.prompt}"), n)
    battery = None
    if battery_items:
        items = [it for path in sorted(Path(battery_items).glob("*.jsonl"))
                 for it in load_items(path)]
        if not items:  # a mistyped path must not pass as "no contamination"
            raise SystemExit(f"no battery items under {battery_items}")
        battery = build_index(items)
    return DenyList(grams=grams, identifiers=AVIATION_IDENTIFIERS,
                    answers=AVIATION_ANSWERS, problem_ids=ids, battery=battery)


def scan_text(text: str, deny: DenyList, n: int = _NGRAM,
              audit: list[dict] | None = None) -> dict | None:
    """Return a rejection reason dict if the text is contaminated, else None.

    `audit`, if given, collects the battery overlaps that stay below rule 4's line.
    """
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
    if deny.battery is not None:
        hits = [{"rule": "battery", "hit": f"{unit.bench}:{unit.id}", "shared": shared,
                 "of": unit.size} for unit, shared in shared_units(text, deny.battery)]
        if hits and hits[0]["shared"] >= STRONG_FRACTION * hits[0]["of"]:
            return hits[0]  # the largest share first: if any unit crosses the line, this one does
        if audit is not None:
            audit.extend(hits)
    return None


def _row_text(obj: dict[str, Any]) -> str:
    """Scannable text from a raw SFTRow dict OR a rendered chat record (incl. tool-call args)."""
    if "messages" in obj:
        parts: list[str] = []

        def _s(v: Any) -> str:
            # Content/arguments are usually strings, but some breadth sources use structured content
            # or a dict `arguments`; coerce so the scan never crashes on a non-str (and still sees
            # the text inside a JSON-dumped dict).
            if v is None:
                return ""
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

        for m in obj["messages"]:
            parts.append(_s(m.get("content")))
            # Revision 2's thinking-on rows keep the reasoning beside the content.
            parts.append(_s(m.get("reasoning_content")))
            # Target C trajectories carry the SQL/code in tool-call arguments, not in content.
            for tc in (m.get("tool_calls") or []):
                parts.append(_s((tc.get("function") or {}).get("arguments")))
        # The tool schemas a row offers: rule 4 checks them against BFCL's, which are indexed with
        # sorted keys, so these are dumped the same way (key order must not decide a match).
        if obj.get("tools"):
            parts.append(json.dumps(obj["tools"], ensure_ascii=False, sort_keys=True))
        return "\n".join(p for p in parts if p)
    if "turns" in obj:
        return row_from_dict(obj).text_blob()
    return json.dumps(obj)


def _row_id(obj: dict[str, Any], n: int) -> str:
    return str((obj.get("meta") or {}).get("id") or obj.get("id") or f"#{n}")


def scan_file(in_path: str, out_path: str | None = None,
              battery_items: str | None = None) -> dict:
    deny = build_denylist(battery_items=battery_items)
    report: dict[str, Any] = {
        "problems": len(deny.problem_ids), "denylist_grams": len(deny.grams),
        "scanned": 0, "clean": 0, "rejected": 0,
        "by_rule": {}, "examples": [],
    }
    battery_hits: Counter[str] = Counter()  # rejections per battery unit
    below: Counter[str] = Counter()  # kept rows' sub-threshold overlaps, per battery unit
    below_rows = 0
    below_examples: list[dict] = []
    out = open(out_path, "w") if out_path else None
    try:
        with open(in_path) as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                row = _row_id(obj, report["scanned"])
                report["scanned"] += 1
                audit: list[dict] = []
                reason = scan_text(_row_text(obj), deny, audit=audit)
                if reason is None:
                    report["clean"] += 1
                    if out:
                        out.write(json.dumps(obj) + "\n")
                    below_rows += bool(audit)
                    for hit in audit:
                        below[hit["hit"]] += 1
                        if len(below_examples) < 20:
                            below_examples.append({"row": row, **hit})
                else:
                    report["rejected"] += 1
                    report["by_rule"][reason["rule"]] = report["by_rule"].get(reason["rule"], 0) + 1
                    if reason["rule"] == "battery":
                        battery_hits[reason["hit"]] += 1
                    if len(report["examples"]) < 10:
                        report["examples"].append({"row": row, **reason})
    finally:
        if out:
            out.close()
    if deny.battery is not None:
        by_bench: Counter[str] = Counter()
        for unit in deny.battery.units:
            by_bench[unit.bench] += 1
        report["battery"] = {
            "units": dict(sorted(by_bench.items())),
            "unchecked_too_short": deny.battery.unchecked,
            "rejected_by_unit": dict(battery_hits.most_common()),
            "below_line": {"rows": below_rows, "units": len(below),
                           "by_unit": dict(below.most_common()), "examples": below_examples},
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Decontaminate an SFT mixture against dsbench and the "
                                             "acceptance battery")
    ap.add_argument("input", help="mixture JSONL (raw SFTRow rows or rendered chat records)")
    ap.add_argument("--out", default="", help="write clean rows here (omit to only report)")
    ap.add_argument("--report", default="", help="write the report JSON here")
    ap.add_argument("--battery-items", default="",
                    help="a battery run's items/ dir: also reject rows on the battery (rule 4)")
    ap.add_argument("--fail-on-hit", action="store_true", help="exit 1 if any row is rejected")
    args = ap.parse_args()

    report = scan_file(args.input, args.out or None, args.battery_items or None)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)

    print(f"problems: {report['problems']}  denylist 13-grams: {report['denylist_grams']}")
    print(f"scanned: {report['scanned']}  clean: {report['clean']}  rejected: {report['rejected']}")
    print(f"by rule: {report['by_rule']}")
    for ex in report["examples"][:8]:
        print(f"  REJECT [{ex['rule']}] {ex['row']}: {str(ex['hit'])[:80]}")
    if "battery" in report:
        b = report["battery"]
        print(f"battery units: {b['units']}  unchecked (too short): {b['unchecked_too_short']}")
        print(f"battery rejections by unit: {b['rejected_by_unit']}")
        print(f"below the line (kept, audit): {b['below_line']['rows']} rows "
              f"touching {b['below_line']['units']} units")
    if args.fail_on_hit and report["rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
