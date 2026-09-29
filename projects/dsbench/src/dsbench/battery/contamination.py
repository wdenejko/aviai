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

`build_index` and `worst_overlap` serve the other direction: the training-side gate
(`sftgen/decontaminate.py`, ADR-004 Revision 2), which decides row by row before training. It uses
the same item text, 13-grams and threshold, plus two additions the measurement did not need (see
`build_index`).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dsbench.battery.items import Item, load_items, read_jsonl

NGRAM = 13  # the ADR-004 protocol (same as sftgen/decontaminate.py)
STRONG_FRACTION = 0.2
# The shortest item the gate matches whole (see `build_index`).
SHORT_MIN = 6


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _hashes(toks: list[str], n: int = NGRAM) -> set[int]:
    return {hash(" ".join(toks[i:i + n])) for i in range(len(toks) - n + 1)}


def _gram_hashes(text: str, n: int = NGRAM) -> set[int]:
    return _hashes(_tokens(text), n)


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


# ---- The training-side gate's index (ADR-004 Revision 2) ----


@dataclass
class Unit:
    """One indexed text: an item's identifying text, or a BFCL item's function schemas."""

    bench: str
    id: str
    size: int  # its distinct 13-grams, or 1 for a short text matched whole


@dataclass
class BatteryIndex:
    units: list[Unit] = field(default_factory=list)
    grams: dict[int, list[int]] = field(default_factory=dict)  # 13-gram hash -> units
    # Short texts, keyed by their first two tokens: (length, hash of the whole sequence, unit).
    short: dict[tuple[str, str], list[tuple[int, int, int]]] = field(default_factory=dict)
    unchecked: dict[str, int] = field(default_factory=dict)  # bench -> texts too short to match


def _units(item: Item) -> list[tuple[str, str]]:
    """(bench, text) pairs to index for one item."""
    units = [(item.bench, item_text(item))]
    tools = item.gen.get("tools")
    if item.bench == "bfcl" and tools:
        # Dumped exactly as the gate dumps a row's `tools` (decontaminate._row_text).
        units.append(("bfcl-schema", json.dumps(tools, ensure_ascii=False, sort_keys=True)))
    return units


def build_index(items: list[Item]) -> BatteryIndex:
    """Index the battery for the training-side gate.

    Two additions to what `scan` measures:
    - **Short items.** A 13-gram index cannot see a text shorter than 13 tokens, and 397 of BFCL's
      1,240 questions are that short ("Calculate the circumference of a circle with radius 3").
      Those are exactly the requests a generated tool row could reproduce. Down to SHORT_MIN
      tokens, such a text is matched as its whole token sequence, which counts as all of it. Below
      that it is too generic for a match to mean anything ("Who discovered radium?", 14 BFCL
      items), and it is counted as unchecked.
    - **BFCL's function schemas**, as units of their own (`bfcl-schema`): the retrain's tool rows
      must not reuse them (ADR-004 Revision 2).
    """
    index = BatteryIndex()
    grams: dict[int, list[int]] = defaultdict(list)
    short: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    unchecked: Counter[str] = Counter()
    for item in items:
        for bench, text in _units(item):
            toks = _tokens(text)
            unit = len(index.units)
            if len(toks) >= NGRAM:
                hashes = _hashes(toks)
                for g in hashes:
                    grams[g].append(unit)
                index.units.append(Unit(bench, item.id, len(hashes)))
            elif len(toks) >= SHORT_MIN:
                short[(toks[0], toks[1])].append((len(toks), hash(" ".join(toks)), unit))
                index.units.append(Unit(bench, item.id, 1))
            else:
                unchecked[bench] += 1
    index.grams, index.short, index.unchecked = dict(grams), dict(short), dict(unchecked)
    return index


def overlaps(text: str, index: BatteryIndex) -> Counter[int]:
    """{unit: distinct 13-grams shared with `text`}; a short unit found whole counts 1 of 1."""
    toks = _tokens(text)
    shared: Counter[int] = Counter()
    for g in _hashes(toks):
        for unit in index.grams.get(g, ()):
            shared[unit] += 1
    for i in range(len(toks) - 1):
        for length, whole, unit in index.short.get((toks[i], toks[i + 1]), ()):
            if hash(" ".join(toks[i:i + length])) == whole:
                shared[unit] = 1
    return shared


def shared_units(text: str, index: BatteryIndex) -> list[tuple[Unit, int]]:
    """Every unit `text` touches, with the grams shared, the largest share of a unit first."""
    shared = overlaps(text, index)
    order = sorted(shared, key=lambda u: (-shared[u] / index.units[u].size, -shared[u], u))
    return [(index.units[u], shared[u]) for u in order]


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
