"""Reasoning prefill (ADR-004 Revision 2, action item 2): the convention written into the base's own
thinking, not into its prompt.

WHY: the hint pilot (reports/gate-evals/20260930-target-a-hints-pilot.md) found that stating
ClickHouse's weekday numbering in the generation prompt fixes the base's answers, but the base
reads it back. A thinking trace narrates its prompt, so most hinted traces cited the hint ("the
prompt explicitly states ..."), and a training row has no hint in its prompt to cite. A prefill
puts the convention where nothing can be cited: it opens, or continues, the base's own thinking
block, as if the base had recalled it, and the base goes on from there. Every token after the
prefill is the base's. The training row is the plain prompt, then the whole trace, prefill
included.

Two placements, piloted side by side:
- `start`: the thinking block opens with the convention. It is simple, but every such row then
  begins with the same sentence. And the base, given the fact before it starts, may reason less:
  the hinted traces ran less than half the base's usual length.
- `recall`: the base first answers plain, and its trace is cut where it first turns to the
  weekday function or states a numbering (`cut`). In the hint pilot's 24 plain ClickHouse traces,
  the cut falls before every claim the base makes about the numbering, a median of 469 characters
  in. The convention takes that sentence's place, and the base continues from it. The row keeps
  the base's own opening, and the one sentence it didn't write is the one it gets wrong. This
  needs the plain reply first, so it is a second phase of generation.

This module is the part that runs on the box, so it needs only the standard library, like
`reasoning_pilot generate`. It finds the cut and builds the `recall` items from the plain phase's
replies. The items, the convention's sentence and its checks are in `target_a_hints.py`.

    python -m dsbench.sftgen.prefill splice --items <phase-1 items> --gen <phase-1 replies> \\
        --out <recall items>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_DAY = r"(?:mon|tues|wednes|thurs|fri|satur|sun)day"
# Where the base turns to the weekday function. ClickHouse's `toDayOfWeek` and its alias
# `dayOfWeek` are what it writes; the other dialects' names are here in case it reaches for one.
_FUNCTION = re.compile(r"\b(?:to)?(?:iso)?day_?of_?week\b|\bisodow\b|\bdow\b|\bweekday\s*\(",
                       re.IGNORECASE)
# ...or where it states a numbering, with no function named: "Sunday=1", "1 for Sunday", "where
# 1 is Sunday", "Saturday is day 7", "Sun = 1".
_NUMBERING = re.compile(
    rf"\b{_DAY}s?\s*(?:==?|is|as|being|means|:|->|=>|→)\s*(?:day\s*)?\(?[0-7]\b"
    rf"|\b[0-7]\s*(?:==?|is|for|as|means|:|->|=>|→)\s*{_DAY}\b"
    r"|\b(?:mon|tue|wed|thu|fri|sat|sun)\s*=\s*[0-7]\b",
    re.IGNORECASE)
_FENCE = re.compile(r"^[ \t]*```", re.MULTILINE)
# A sentence starts after a line break, or after ". ", "? ", "! " or ": ".
_BOUNDARY = re.compile(r"\n|[.!?:](?=[ \t])")
# A list item's marker, kept with the base's text: the convention becomes the item.
_MARKER = re.compile(r"[ \t]*(?:(?:[-*•]|\d+[.)])[ \t]+)?")


def _fences(text: str) -> list[tuple[int, int]]:
    """(start, end) of each fenced code block; an unclosed one runs to the end."""
    marks = [m.start() for m in _FENCE.finditer(text)]
    return [(marks[i], marks[i + 1] if i + 1 < len(marks) else len(text))
            for i in range(0, len(marks), 2)]


def cut(reasoning: str) -> int | None:
    """Where a `recall` prefill cuts the base's plain trace: the start of the sentence in which it
    first turns to the weekday function or states a numbering. None if it never does.

    The cut goes to the sentence's start, not the function's name, so that the convention replaces
    a whole statement ("In ClickHouse, `dayOfWeek` returns ... where Sunday=1"), not the second
    half of one. A query drafted in a code block is cut before its fence, never inside the SQL.
    """
    hits = [m.start() for m in (_FUNCTION.search(reasoning), _NUMBERING.search(reasoning)) if m]
    if not hits:
        return None
    first = min(hits)
    for start, end in _fences(reasoning):
        if start <= first < end:
            return start
    bounds = [m.end() for m in _BOUNDARY.finditer(reasoning, 0, first)]
    at = bounds[-1] if bounds else 0
    if at == 0 or reasoning[at - 1] == "\n":
        return min(_MARKER.match(reasoning, at).end(), first)
    while at < first and reasoning[at] in " \t":
        at += 1
    return at


def splice(item: dict, record: dict) -> dict | None:
    """The `recall` item made from a plain item and its reply: the reply's reasoning up to the
    cut, then the convention, for the base to continue. None when the reply failed or never turns
    to the weekday function."""
    reasoning = record.get("reasoning") or ""
    at = None if record.get("error") else cut(reasoning)
    if at is None:
        return None
    meta = item["meta"]
    sentence = meta["recall"]
    return {"id": f"targetA_recall:{item['verify']['row_id']}#{meta['sample']}",
            "pool": "targetA_recall", "messages": item["messages"],
            "prefill": reasoning[:at] + sentence,
            "verify": {**item["verify"], "recall": sentence},
            "meta": {**meta, "placement": "recall", "cut_chars": at, "spliced_from": item["id"],
                     "teacher": "none: the base's own trace, cut where it first turns to the "
                                "weekday function, the convention, then the base again"}}


def splice_all(items: list[dict], records: list[dict]) -> tuple[list[dict], dict]:
    """(recall items, counts) for every plain item of the prefill pilot. A resumed generation
    appends a retried item's new record, so a reply without an error wins."""
    replies: dict[str, dict] = {}
    for rec in records:
        if rec["id"] not in replies or not rec.get("error"):
            replies[rec["id"]] = rec
    out: list[dict] = []
    counts: Counter[str] = Counter()
    for item in items:
        if item.get("meta", {}).get("placement") != "plain":
            continue
        counts["plain"] += 1
        rec = replies.get(item["id"])
        if rec is None or rec.get("error"):
            counts["no_reply"] += 1
            continue
        spliced = splice(item, rec)
        if spliced is None:
            counts["no_cut"] += 1
            continue
        counts["spliced"] += 1
        out.append(spliced)
    return out, dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("splice", help="build the recall items from the plain phase's replies")
    s.add_argument("--items", type=Path, required=True)
    s.add_argument("--gen", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    items = [json.loads(line) for line in args.items.open()]
    records = [json.loads(line) for line in args.gen.open()]
    spliced, counts = splice_all(items, records)
    args.out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in spliced))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
