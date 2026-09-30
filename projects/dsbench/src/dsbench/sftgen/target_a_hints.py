"""Target A's hint-conditioned generation (ADR-004 Revision 2, action item 2): STaR's
rationalisation for the SQL-dialect conventions.

WHY: the reasoning pilot found the base wrong on exactly the conventions Target A teaches, and for
one reason. It believes ClickHouse's `toDayOfWeek` numbers Sunday 1, as MySQL's `DAYOFWEEK` does,
where ClickHouse is ISO: Monday 1 to Sunday 7. ClickHouse weekdays verified 1 of 6 and weekends 1
of 13; DuckDB weekends 3 of 7 (reports/gate-evals/20260928-reasoning-pilot.md, addendum).
Rejection sampling alone keeps only the answers the base already gets right, so it can't teach
the fix. So the generation prompt states the convention the row tests (the hint), and a reply is
kept when its SQL verifies and it doesn't read as told: the training prompt has no hint, so the
kept trace has to reason its way to the convention.

The hint is one sentence appended to the system prompt, and every hint is checked on the engines
before any item is built (`check_hints`): a wrong hint would teach a wrong convention at full
weight. A reply is kept when:
- it finished, with reasoning;
- it is one ```sql block and nothing else, as the prompt asks (`reasoning_pilot.sql_only`);
- that SQL returns the row's truth on the dialect's sandboxed engine, on the row's own table,
  rebuilt from its seed after the gold SQL reproduces the truth there;
- for a hinted item, it doesn't cite the hint (`cites_hint`): no reference to having been told
  (the system prompt, a note, "we're told"), no "as stated" beside the hint's own terms, and no
  run of COPY_RUN tokens taken from the hint that the training prompt doesn't also contain.

The pilot pairs every item with a plain twin, the same prompt without the hint: 4 dialects x 4
families x 12 rows, 48 per dialect, as ADR-004 plans. Both run through `reasoning_pilot.py
generate` as they are (the hinted item's `messages` carry the hint, its `train_messages` don't).
The pairs show per cell what the hint changes: verified answers, citations, reasoning length. The
plain timezone rows also re-measure that family, which states its offsets now.

    uv run python -m dsbench.sftgen.target_a_hints items --battery-items data/battery/items \\
        --out data/sft/rev2_target_a_pilot.jsonl --report data/sft/rev2_target_a_pilot_manifest.json
    uv run python -m dsbench.sftgen.target_a_hints verify --items <items> --gen <gen> --out <out>

Both need the sandbox's four engines: docker compose -f sandbox/docker-compose.yml up -d --build
clickhouse postgres mysql duckdb.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics as st
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dsbench.sftgen.conventions import DIALECT_DISPLAY
from dsbench.sftgen.reasoning_pilot import SQL_BLOCK, sql_only

SEED = 20260930
ROWS_PER_TABLE = 1500  # rows in each synthetic table, as the Gate-2 slice had
PILOT_REPS = 2  # tables per domain: 6 domains x 2 = 12 rows per (dialect, family) cell
DIALECTS = ("clickhouse", "duckdb", "postgres", "mysql")
FAMILIES = ("weekday-numbering", "weekend-flag", "timezone-direction", "month-bucket")
COPY_RUN = 10  # a run this long taken from the hint reads as copied, not reasoned

# --- the hints -----------------------------------------------------------------------------------

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# Weekday numberings, from Python's weekday (Monday 0 .. Sunday 6) to the engine's number.
_SCHEMES: dict[str, Callable[[int], int]] = {
    "iso": lambda w: w + 1,  # Monday 1 .. Sunday 7
    "sun0": lambda w: (w + 1) % 7,  # Sunday 0 .. Saturday 6
    "sun1": lambda w: (w + 1) % 7 + 1,  # Sunday 1 .. Saturday 7
    "mon0": lambda w: w,  # Monday 0 .. Sunday 6
}
# 2024-01-01 was a Monday, so 2024-01-0{1..7} are Monday .. Sunday.
_WEEK = tuple(f"2024-01-0{d}" for d in range(1, 8))


@dataclass(frozen=True)
class Function:
    shown: str  # as the hint writes it
    check: str  # a query on a date literal, with {d} for the date
    scheme: str  # a key of _SCHEMES, or "month"
    term: str  # the name a citation would use beside "as stated"


WEEKDAY_FUNCTIONS: dict[str, tuple[Function, ...]] = {
    "clickhouse": (Function("toDayOfWeek(date)", "SELECT toDayOfWeek(toDate('{d}'))", "iso",
                            "todayofweek"),),
    "duckdb": (Function("dayofweek(date)", "SELECT dayofweek(DATE '{d}')", "sun0", "dayofweek"),
               Function("isodow(date)", "SELECT isodow(DATE '{d}')", "iso", "isodow")),
    "postgres": (Function("EXTRACT(DOW FROM date)", "SELECT EXTRACT(DOW FROM DATE '{d}')", "sun0",
                          "dow"),
                 Function("EXTRACT(ISODOW FROM date)", "SELECT EXTRACT(ISODOW FROM DATE '{d}')",
                          "iso", "isodow")),
    "mysql": (Function("DAYOFWEEK(date)", "SELECT DAYOFWEEK(DATE '{d}')", "sun1", "dayofweek"),
              Function("WEEKDAY(date)", "SELECT WEEKDAY(DATE '{d}')", "mon0", "weekday")),
}
MONTH_FUNCTIONS: dict[str, Function] = {
    "clickhouse": Function("toMonth(date)", "SELECT toMonth(toDate('{d}'))", "month", "tomonth"),
    "duckdb": Function("month(date)", "SELECT month(DATE '{d}')", "month", ""),
    "postgres": Function("EXTRACT(MONTH FROM date)", "SELECT EXTRACT(MONTH FROM DATE '{d}')",
                         "month", ""),
    "mysql": Function("MONTH(date)", "SELECT MONTH(DATE '{d}')", "month", ""),
}
TIMEZONE_HINT = ("UTC = local time - UTC offset: a clock at UTC-4 is 4 hours behind UTC, so its "
                 "UTC time is the local time plus 4 hours.")


def _numbering(fn: Function) -> str:
    number = _SCHEMES[fn.scheme]
    order = sorted(range(7), key=number)
    first, last = order[0], order[-1]
    return f"{fn.shown} numbers {_DAYS[first]} {number(first)} through {_DAYS[last]} {number(last)}"


def hint(family: str, dialect: str) -> str:
    """The convention a row of this family tests, stated for the generation prompt."""
    if family in ("weekday-numbering", "weekend-flag"):
        # A weekend row gets the numbering, not the answer: it still has to find Saturday and
        # Sunday's numbers.
        numberings = "; ".join(_numbering(fn) for fn in WEEKDAY_FUNCTIONS[dialect])
        return f"In {DIALECT_DISPLAY[dialect]}, {numberings}."
    if family == "month-bucket":
        return (f"In {DIALECT_DISPLAY[dialect]}, {MONTH_FUNCTIONS[dialect].shown} returns the "
                f"month number, January 1 through December 12.")
    if family == "timezone-direction":
        return TIMEZONE_HINT
    raise ValueError(f"no hint for {family}")


def hint_terms(family: str, dialect: str) -> set[str]:
    """Words that tie an "as stated" to the hint rather than to the question."""
    if family in ("weekday-numbering", "weekend-flag"):
        return {fn.term for fn in WEEKDAY_FUNCTIONS[dialect]} | {"iso"}
    if family == "month-bucket":
        return {MONTH_FUNCTIONS[dialect].term} - {""}
    return {"behind"}  # the question states the offsets; the hint adds the direction


def hint_checks(dialect: str) -> list[tuple[str, int]]:
    """(query, expected) pairs that every weekday and month hint of the dialect must pass."""
    checks = [(fn.check.format(d=day), _SCHEMES[fn.scheme](w))
              for fn in WEEKDAY_FUNCTIONS[dialect] for w, day in enumerate(_WEEK)]
    month = MONTH_FUNCTIONS[dialect]
    checks += [(month.check.format(d=f"2024-{m:02d}-15"), m) for m in (1, 7, 12)]
    return checks


def check_hints(engines: dict) -> dict[str, int]:
    """Run every dialect's hint checks on its engine; raise on the first that fails. The timezone
    hint is arithmetic, checked against the family's truth by the tests."""
    passed = {}
    for dialect in DIALECTS:
        engine = engines[dialect]
        engine.setup()
        try:
            # The sandboxed DuckDB runs a query against a loaded table; the others don't need one.
            engine.load("hint_check", pd.DataFrame({"d": pd.to_datetime(["2024-01-01"])}))
            for sql, expected in hint_checks(dialect):
                got = engine.scalar(sql)
                if got != expected:
                    raise RuntimeError(f"{dialect}: {sql} returned {got}, the hint says "
                                       f"{expected}")
            passed[dialect] = len(hint_checks(dialect))
        finally:
            engine.teardown()
    return passed


# --- the citation filter -------------------------------------------------------------------------

# Always a citation: the reply names a hint or a note. Nothing in the training prompt is one.
_HINT_WORD = re.compile(
    r"\b(?:the|this|that)\s+(?:hint|note|tip|reminder)\b"
    r"|\b(?:provided|given)\s+(?:fact|information|info|note|hint)\b",
    re.IGNORECASE)
# A citation when the hint's own terms (a function name, "ISO") are what the phrase attributes.
# The training prompt has a system prompt too, and the timezone family's question states its
# offsets, so "the system prompt asks for one query" and "we're told New York is UTC-4" are fine;
# "we're told toDayOfWeek is ISO" isn't. The base writes "the prompt says" all the time, quoting
# the question (211 times in the pilot's 80 Target A traces): `The prompt says "month of April",
# so toMonth(order_ts) = 4` names the function in its own SQL, after the quote. So:
# - "as stated", "according to the prompt" attribute their whole sentence;
_AS_STATED = re.compile(
    r"\bas\s+(?:stated|noted|mentioned|given|specified|provided|indicated|described)\b"
    r"|\b(?:stated|noted|mentioned|given|provided|specified|indicated)\s+(?:above|earlier|before)\b"
    r"|\baccording\s+to\s+the\s+(?:prompt|instructions?|system|context|problem|question)\b"
    r"|\b(?:from|in|per)\s+the\s+system\s+(?:prompt|message|instructions?)\b",
    re.IGNORECASE)
# - "the prompt says", "we're told" attribute what follows, up to the end of the clause.
_SAYS = re.compile(
    r"\b(?:we're|we\s+are|we\s+were|i'm|i\s+am|i\s+was|(?:have|has|'ve)\s+been)\s+told\b"
    r"|\byou\s+(?:told|said|say|mentioned|noted)\b"
    r"|\b(?:prompt|message|instructions?|problem|question|task|context)\s+"
    r"(?:says|states|tells\s+us|mentions|notes|specifies|gives)\b",
    re.IGNORECASE)
_CLAUSE_END = re.compile(r"[,;]|\s(?:so|then|which|but|because|and\s+so)\s|\s[-\u2014]\s")
_SENTENCE = re.compile(r"[^.!?\n]+")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _grams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def cites_hint(text: str, hint_text: str, terms: set[str], prompt: str) -> str | None:
    """Why a reply reads as told the hint, or None. `prompt` is the training prompt: what the
    reply shares with it was never the hint's to give."""
    text = text.replace("\u2019", "'")
    named = _HINT_WORD.search(text)
    if named:
        return f"names a hint: {named.group(0)!r}"
    for sentence in _SENTENCE.findall(text):
        stated = _AS_STATED.search(sentence)
        if stated and terms & set(_tokens(sentence)):
            return f"{stated.group(0)!r} in a sentence with the hint's terms"
        for says in _SAYS.finditer(sentence):
            clause = _CLAUSE_END.split(sentence[says.end():], maxsplit=1)[0]
            if terms & set(_tokens(clause)):
                return f"{says.group(0)!r} followed by the hint's terms"
    copied = (_grams(_tokens(text), COPY_RUN) & _grams(_tokens(hint_text), COPY_RUN)
              - _grams(_tokens(prompt), COPY_RUN))
    if copied:
        return f"copies the hint: {' '.join(sorted(copied)[0])!r}"
    return None


def soft_flags(text: str) -> list[str]:
    """The "told" phrases of a reply, citations or not: the pilot reviews them by hand."""
    text = text.replace("\u2019", "'")
    return sorted([m.group(0) for m in _AS_STATED.finditer(text)]
                  + [m.group(0) for m in _SAYS.finditer(text)], key=text.find)


# --- items ---------------------------------------------------------------------------------------

def parse_row_id(row_id: str) -> tuple[str, str, int]:
    """(dialect, domain, seed) from 'A-<family>-<dialect>-<domain>-<seed>'."""
    parts = row_id.split("-")
    return parts[-3], parts[-2], int(parts[-1])


def make_items(row: dict, n: int = ROWS_PER_TABLE) -> list[dict]:
    """The plain item and its hinted twin for one Target A row (dialect_conventions' raw form)."""
    system, user, gold = (t["content"] for t in row["turns"])
    family, dialect = row["family"], row["dialect"]
    _, domain, seed = parse_row_id(row["id"])
    train = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    verify = {"kind": "target_a", "dialect": dialect, "family": family,
              "truth": row["verification"]["truth"], "gold_sql": sql_only(gold),
              "row_id": row["id"], "n": n}
    meta = {"family": family, "dialect": dialect, "domain": domain, "seed": seed,
            "source_key": "targetA", "licence": "Apache-2.0"}
    plain = {"id": f"targetA:{row['id']}", "pool": "targetA", "messages": train,
             "verify": verify,
             "meta": {**meta, "hinted": False, "teacher": "none: the base answers"}}
    said = hint(family, dialect)
    head, answer_with, tail = system.partition(" Answer with")  # the hint goes before it
    hinted = {"id": f"targetA_hint:{row['id']}", "pool": "targetA_hint",
              "messages": [{"role": "system", "content": f"{head} {said}{answer_with}{tail}"},
                           {"role": "user", "content": user}],
              "train_messages": train,
              "verify": {**verify, "hint": said},
              "meta": {**meta, "hinted": True,
                       "teacher": "none: the base answers with the convention stated; the "
                                  "training prompt drops it"}}
    return [plain, hinted]


def build_items(*, seed: int = SEED, reps: int = PILOT_REPS, n: int = ROWS_PER_TABLE,
                gate: Callable[[dict], str | None] | None = None) -> tuple[list[dict], dict]:
    """(items, report): each Target A row the generator verifies, as a plain and a hinted item."""
    from dsbench.sftgen.dialect_conventions import generate
    from dsbench.sftgen.schema import row_to_dict

    rows, generated = generate(seed=seed, reps=reps, n=n, dialects=list(DIALECTS),
                               families=list(FAMILIES), thinking_frac=0.0)
    missing = set(DIALECTS) - set(generated["engines"])
    if missing:
        raise RuntimeError(f"no engine for {sorted(missing)}: bring the sandbox up")
    items: list[dict] = []
    rejected: Counter[str] = Counter()
    for row in map(row_to_dict, rows):
        pair = make_items(row, n)
        reason = gate(pair[0]) if gate else None  # the training text is the plain item's
        if reason:
            rejected[reason.split(" ", 1)[0]] += 1
            continue
        items += pair
    cells = Counter(f"{i['meta']['dialect']}/{i['meta']['family']}" for i in items
                    if not i["meta"]["hinted"])
    report = {"seed": seed, "reps": reps, "rows_per_table": n, "rows": len(rows),
              "generator_rejected": generated["rejected"], "items": len(items),
              "pairs_by_cell": dict(sorted(cells.items())), "gate_rejected": dict(rejected),
              "hints": {f"{d}/{f}": hint(f, d) for d in DIALECTS for f in FAMILIES}}
    return items, report


# --- verify --------------------------------------------------------------------------------------

def check_reply(item: dict, rec: dict) -> tuple[str, str | None, str | None]:
    """(status before execution, the SQL to run, the citation) for one generated reply. A status
    of "" means the SQL decides."""
    answer = rec.get("answer") or ""
    reasoning = rec.get("reasoning") or ""
    cites = None
    if "hint" in item["verify"]:
        prompt = "\n".join(m["content"] for m in item["train_messages"])
        cites = cites_hint(f"{reasoning}\n{answer}", item["verify"]["hint"],
                           hint_terms(item["verify"]["family"], item["verify"]["dialect"]), prompt)
    if rec.get("error") or rec.get("finish_reason") != "stop":
        return "unfinished", None, cites
    if not reasoning.strip():
        return "no_reasoning", None, cites
    if not SQL_BLOCK.search(answer):
        return "no_sql", None, cites
    sql = sql_only(answer)
    if sql is None:
        return "not_sql_only", None, cites
    return "", sql, cites


def verify(items_path: Path, gen_path: Path, out_path: Path,
           engines: dict | None = None) -> dict:
    """Check every generated reply on its dialect's sandboxed engine; write one record per reply
    and return the per-cell summary. `engines` ({dialect: Engine}) is for tests."""
    from dsbench.sftgen import synth

    items = {i["id"]: i for i in map(json.loads, items_path.open())}
    gens = [g for g in map(json.loads, gen_path.open()) if g["id"] in items]
    if engines is None:
        from dsbench.sftgen.engines import available_engines
        engines = {e.name: e for e in available_engines(sandboxed=True)}  # model SQL
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in gens:
        v = items[rec["id"]]["verify"]
        _, domain, seed = parse_row_id(v["row_id"])
        groups[(v["dialect"], domain, seed, v["n"])].append(rec)
    records = []
    for (dialect, domain_name, seed, n), recs in sorted(groups.items()):
        if dialect not in engines:
            raise SystemExit(f"no sandboxed {dialect} engine (is the dsbench sandbox up?)")
        engine = engines[dialect]
        domain = synth.build(domain_name, seed, n)
        engine.setup()
        try:
            engine.load(domain.name, domain.df)
            for rec in recs:
                item = items[rec["id"]]
                truth = item["verify"]["truth"]
                status, sql, cites = check_reply(item, rec)
                got = None
                if engine.scalar(item["verify"]["gold_sql"]) != truth:
                    status = "rebuild_mismatch"  # the data, not the model, is wrong
                elif not status:
                    try:
                        got = engine.scalar(sql)
                        status = "verified" if got == truth else "wrong"
                    except Exception as exc:  # noqa: BLE001 - an engine error is a wrong answer
                        status, got = "error", str(exc)[:200]
                kept = status == "verified" and cites is None
                soft = soft_flags(f"{rec.get('reasoning') or ''}\n{rec.get('answer') or ''}")
                records.append({**rec, "sql": sql, "train_messages": item.get(
                    "train_messages", item["messages"]), "check": {
                    "status": status, "truth": truth, "got": got, "cites": cites,
                    "soft_flags": soft, "kept": kept}})
        finally:
            engine.teardown()
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n"
                                for r in records))
    return summarize(records, items)


def summarize(records: list[dict], items: dict) -> dict:
    """Per (dialect, family) cell, plain against hinted: replies, verified, citing, kept, and the
    median reasoning length where the generation recorded it."""
    cells: dict[str, dict] = defaultdict(lambda: {"plain": Counter(), "hinted": Counter()})
    lengths: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in records:
        meta = items[r["id"]]["meta"]
        cell = f"{meta['dialect']}/{meta['family']}"
        side = "hinted" if meta["hinted"] else "plain"
        c = cells[cell][side]
        c["replies"] += 1
        c[r["check"]["status"]] += 1
        c["cites"] += r["check"]["cites"] is not None
        c["soft_flagged"] += bool(r["check"]["soft_flags"])
        c["kept"] += r["check"]["kept"]
        if r.get("reasoning_tokens") is not None:
            lengths[(cell, side)].append(r["reasoning_tokens"])
    out = {}
    for cell, sides in sorted(cells.items()):
        out[cell] = {side: {**dict(counts), "reasoning_tokens_median":
                            st.median(lengths[(cell, side)]) if lengths[(cell, side)] else None}
                     for side, counts in sides.items() if counts}
    return out


# --- CLI -----------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("items")
    b.add_argument("--battery-items", required=True, help="a battery run's items/ dir")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--report", type=Path, required=True)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--reps", type=int, default=PILOT_REPS)
    v = sub.add_parser("verify")
    v.add_argument("--items", type=Path, required=True)
    v.add_argument("--gen", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "verify":
        print(json.dumps(verify(args.items, args.gen, args.out), indent=1))
        return

    from dsbench.sftgen.decontaminate import strict_gate
    from dsbench.sftgen.engines import available_engines

    engines = {e.name: e for e in available_engines(sandboxed=True)}
    missing = set(DIALECTS) - set(engines)
    if missing:
        raise SystemExit(f"no sandboxed engine for {sorted(missing)}: docker compose -f "
                         "sandbox/docker-compose.yml up -d --build clickhouse postgres mysql "
                         "duckdb")
    checked = check_hints(engines)
    items, report = build_items(seed=args.seed, reps=args.reps,
                                gate=strict_gate(args.battery_items))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.report.write_text(json.dumps({"params": {"copy_run": COPY_RUN,
                                                  "battery_items": args.battery_items},
                                       "hint_checks_passed": checked, **report,
                                       "out": str(args.out), "out_sha256": digest},
                                      indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "items", "gate_rejected")}))


if __name__ == "__main__":
    main()
