"""Check the battery's items against the fine-tune's training data (13-gram overlap, ADR-004).

ADR-001 (Gate 2, item 1) asked for the mixture to be decontaminated against every eval set. It was
only decontaminated against dsbench's own prompts (`sftgen/decontaminate.py`). The public breadth
pools (OpenCoder, DataMind, Gretel SQL, Tulu 3, SWE) could carry benchmark text, and a "gain" on an
item the adapter saw in training measures memory, not skill. This module measures that exposure
after the fact, so the report can show every delta on the full set AND on the clean subset.

Only the item-specific text is indexed (the problem, question, reference solution), never the
benchmark's shared boilerplate. LiveCodeBench's "Read the inputs from stdin ..." format line, for
example, is copied into many SFT sets, and indexing it would flag every item.

Two levels, because 13-gram overlap on code has false positives (`import pandas as pd ...` is
common to many texts that are not the benchmark):
  any     at least one of the item's 13-grams occurs in the training data
  strong  at least STRONG_FRACTION of the item's 13-grams occur there
The clean subset for the robustness check excludes `strong` items.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from dsbench.battery.items import Item, load_items, read_jsonl

NGRAM = 13  # the ADR-004 protocol (same as sftgen/decontaminate.py)
STRONG_FRACTION = 0.2


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _gram_hashes(text: str, n: int = NGRAM) -> set[int]:
    toks = _tokens(text)
    return {hash(" ".join(toks[i:i + n])) for i in range(len(toks) - n + 1)}


def _between(text: str, start: str, end: str | None) -> str:
    i = text.find(start)
    if i < 0:
        return text
    text = text[i + len(start):]
    j = text.find(end) if end else -1
    return text[:j] if j >= 0 else text


def item_text(item: Item) -> str:
    """The text that identifies this item: problem and reference answer, never boilerplate."""
    user = next((m["content"] for m in reversed(item.messages) if m["role"] == "user"), "")
    ref = item.ref
    if item.bench == "humaneval_plus":
        return ref["prompt"] + "\n" + ref["canonical_solution"]
    if item.bench == "ds1000":
        return user + "\n" + ref["reference_code"]
    if item.bench == "ifeval":
        return ref["prompt"]
    if item.bench == "mmlu_pro":
        return _between(user, "Question: ", "\nAnswer: Let's think")
    if item.bench == "gpqa":
        return user.split("\n\n", 1)[-1]
    if item.bench == "lcb":
        return _between(user, "### Question:\n", "\n\n### Format")
    if item.bench == "bird":
        return _between(user, "-- Question: ", "\n\nReturn only") + "\n" + ref["gold_sql"]
    if item.bench == "bfcl":
        return user
    return user


def mixture_texts(path: Path):
    """Every message text of every training row (chat records or raw SFT rows)."""
    for row in read_jsonl(path):
        msgs = row.get("messages") or []
        yield "\n".join(str(m.get("content") or "") for m in msgs) or json.dumps(row)


def scan(items: list[Item], mixtures: list[Path]) -> dict[str, dict]:
    """{bench: {n, any: [ids], strong: [ids]}} for items overlapping any mixture file."""
    index: dict[int, list[int]] = defaultdict(list)
    grams_per_item = []
    for k, item in enumerate(items):
        grams = _gram_hashes(item_text(item))
        grams_per_item.append(len(grams))
        for g in grams:
            index[g].append(k)
    hits: list[set[int]] = [set() for _ in items]
    for path in mixtures:
        for text in mixture_texts(path):
            for g in _gram_hashes(text):
                for k in index.get(g, ()):
                    hits[k].add(g)
    out: dict[str, dict] = {}
    for k, item in enumerate(items):
        entry = out.setdefault(item.bench, {"n": 0, "any": [], "strong": []})
        entry["n"] += 1
        if hits[k]:
            entry["any"].append(item.id)
            if len(hits[k]) >= STRONG_FRACTION * max(1, grams_per_item[k]):
                entry["strong"].append(item.id)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items-dir", required=True, type=Path)
    ap.add_argument("--mixture", required=True, action="append", type=Path,
                    help="training JSONL; repeat for several (pilot + gate2)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    items = [it for path in sorted(args.items_dir.glob("*.jsonl")) for it in load_items(path)]
    result = scan(items, args.mixture)
    args.out.write_text(json.dumps({"ngram": NGRAM, "strong_fraction": STRONG_FRACTION,
                                    "mixtures": [str(p) for p in args.mixture],
                                    "benches": result}, indent=1))
    for bench, r in sorted(result.items()):
        print(f"{bench:15s} n={r['n']:5d}  any={len(r['any']):4d}  strong={len(r['strong']):4d}")


if __name__ == "__main__":
    main()
