"""SQL on real schemas for Revision 2 (ADR-004 Revision 2, action item 3): prompts built from
SynSQL-2.5M's own SQLite databases, and the execution-match check.

WHY: Gate 2's SQL came from Gretel, and one answer in five failed in SQLite against its own schema;
BIRD fell 9.1 points. This pool is the base's own SQL, kept only where it returns what the
dataset's gold SQL returns on the real database: BIRD's execution accuracy, one set of rows
against the other.

Source: SynSQL-2.5M (OmniSQL; Apache-2.0; per its card, generated entirely with open-source
LLMs). 2.5M questions over 16,583 synthetic SQLite databases (databases.zip, 937 MB unpacked). The
questions sit in one 9.4 GB JSON array, so they are read at seeded offsets rather than downloaded
whole: an offset takes the first whole item after it, and each item takes a database no other item
has. The revision is pinned.

SynSQL's databases hold about two rows a table: the prompt's example rows are the whole table,
and on them a wrong query often returns the gold's rows (AVG and SUM agree on one row). So every
query also runs on VARIANTS bigger variants of its database, the same tables with VARIANT_ROWS
rows each (sandbox/sqlite/runner.py), and a reply must match the gold on all of them: test-suite
accuracy (Zhong, Yu & Klein, 2020). With an aggregate swapped in the gold SQL (AVG for SUM, MIN
for MAX), 60% of the wrong queries passed on the database alone (300 items of a first build) and
15% with the variants (the pool's 1,112), most of those equivalent to the gold (an aggregate the
query doesn't return, or one row a group by construction).

A candidate becomes an item when:
- its style isn't "Multi-turn Dialogue", a transcript pasted into one question;
- its prompt fits in MAX_PROMPT_TOKENS (chars/3.5). The prompt is the schema in the battery's
  BIRD rendering (each CREATE TABLE with three example rows), any background the dataset gives,
  and the question. SQL replies ran to 2,604 tokens at the pilot's p90, so 4,096 leaves room;
- its gold SQL runs on the database and every variant, returns at most MAX_ROWS rows on each, and
  returns some rows, not all of them NULL, 0 or '', on at least one. A gold SQL that returns
  nothing anywhere, or only blanks (a COUNT or an AVG over no rows), would pass any query that
  matches nothing, so it checks nothing;
- it shares nothing with dsbench, nor any 13-gram with the battery (`strict_gate`).

The prompt names SQLite and asks for the query only, in a ```sql block. Its wording varies over a
few phrasings, so the pool teaches SQL rather than one prompt. A reply passes when it is that one
block and nothing else, and its rows are the gold SQL's on the database and every variant. Every
query runs in the sandbox's `sqlite` container (sandbox/sqlite), the gold SQL included: a model
wrote it too.

    uv run --with huggingface_hub python -m dsbench.sftgen.synsql build \\
        --battery-items data/battery/items --out data/sft/rev2_sql_prompts.jsonl \\
        --report data/sft/rev2_sql_prompts_manifest.json
    uv run --with huggingface_hub python -m dsbench.sftgen.synsql verify \\
        --items <items> --gen <gen> --out <verified>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import statistics as st
import subprocess
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from dsbench.battery.prepare import _bird_schema
from dsbench.sftgen.reasoning_pilot import SQL_BLOCK, latest_rows, sql_only

REPO = "seeklhy/SynSQL-2.5M"
REVISION = "cca2c84cc3b41afa6b51534762a6a3a4a420baca"
SEED = 20260930
# ADR-004's table: 500 rows. The keep rate is a guess until the base's replies are checked: the
# verified pools were planned at 30-60%.
SQL_ROWS, SQL_KEEP = 500, 0.45
MAX_PROMPT_TOKENS = 4096
MAX_ROWS = 1000
# Three variants of 100 rows a table: on 300 items' mutated gold SQL, a swapped aggregate passed
# 16% of the time with one variant and 12% with three; a flipped sort before a LIMIT 47% with
# 30-row variants and 20% with 100-row ones.
VARIANTS, VARIANT_ROWS = 3, 100
EXCLUDED_STYLES = ("Multi-turn Dialogue",)
CHUNK = 1 << 16  # an item is a few KB; 64 KB always holds a whole one after the offset
_ITEM_START = "\n  {\n"  # data.json is indented by 2: every item, and only an item, starts so
_BLANK = (None, 0, "")  # 0 == 0.0, so a float zero is blank too

INSTRUCTIONS = (
    "Here is a SQLite database, as its CREATE TABLE statements, each with three example rows."
    "\n\n{schema}\n\n{knowledge}Question: {question}\n\nWrite one SQLite query that answers the "
    "question. Reply with the query only, in a ```sql block.",
    "{schema}\n\nThe tables above live in a SQLite database. {knowledge}Write a SQLite query for "
    "this request: {question}\n\nGive only the SQL, in a ```sql code block.",
    "I'm working with this SQLite schema (sample rows included):\n\n{schema}\n\n{knowledge}"
    "{question}\n\nAnswer with a single SQLite query in a ```sql block and nothing else.",
)


def first_item(chunk: str) -> dict | None:
    """The first whole item of data.json that starts inside `chunk`, if the chunk holds one."""
    start = chunk.find(_ITEM_START)
    if start < 0:
        return None
    try:
        item, _ = json.JSONDecoder().raw_decode(chunk, start + 3)
    except json.JSONDecodeError:  # the chunk ends inside the item
        return None
    return item


def build_prompt(row: dict, schema: str, rng: random.Random) -> tuple[str, int]:
    """(prompt, which of INSTRUCTIONS worded it)."""
    k = rng.randrange(len(INSTRUCTIONS))
    knowledge = (f"Background: {row['external_knowledge'].strip()}\n\n"
                 if (row.get("external_knowledge") or "").strip() else "")
    return INSTRUCTIONS[k].format(schema=schema, knowledge=knowledge,
                                  question=row["question"].strip()), k


def make_item(row: dict, prompt: str, instruction: int) -> dict:
    qid = hashlib.sha1(row["question"].encode()).hexdigest()[:10]
    return {
        "id": f"synsql:{row['db_id']}:{qid}", "pool": "synsql", "bucket": "sql",
        "messages": [{"role": "user", "content": prompt}],
        "prompt_tokens_est": round(len(prompt) / 3.5),
        "verify": {"kind": "sqlite_ex", "db_id": row["db_id"], "gold_sql": row["sql"],
                   "variants": VARIANTS, "variant_rows": VARIANT_ROWS},
        "meta": {"db_id": row["db_id"], "sql_complexity": row.get("sql_complexity"),
                 "question_style": row.get("question_style"), "instruction": instruction,
                 "source_key": "synsql",
                 "hf_id": REPO, "revision": REVISION, "licence": "Apache-2.0",
                 "teacher": "none: the base answers (the question and gold SQL came from "
                            "SynSQL's open-source LLMs)"},
    }


class Databases:
    """SynSQL's databases, taken out of databases.zip one at a time as items need them."""

    def __init__(self, zip_path: Path, zone: Path) -> None:
        self.zip = zipfile.ZipFile(zip_path)
        self.dir = zone / "databases"

    def path(self, db_id: str) -> Path:
        target = self.dir / f"{db_id}.sqlite"
        if not target.exists():
            data = self.zip.read(f"databases/{db_id}/{db_id}.sqlite")
            self.dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return target


class SQLiteSandbox:
    """The sandbox's `sqlite` container: each call runs its queries in a fresh process there, on a
    read-only copy of the database sent on stdin, and on variants of it built there
    (sandbox/sqlite/runner.py)."""

    container = "avbench-sqlite"

    def available(self) -> bool:
        try:
            out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}",
                                  self.container], capture_output=True, text=True, timeout=10)
            return out.stdout.strip() == "true"
        except Exception:  # noqa: BLE001
            return False

    def run(self, db: Path, queries: list[str], timeout_s: int = 30,
            max_rows: int = MAX_ROWS) -> list[dict]:
        """Each query's result on the database."""
        return self.run_suite(db, "", queries, variants=0, timeout_s=timeout_s,
                              max_rows=max_rows)[0]

    def run_suite(self, db: Path, db_id: str, queries: list[str], variants: int = VARIANTS,
                  rows: int = VARIANT_ROWS, timeout_s: int = 30,
                  max_rows: int = MAX_ROWS) -> list[list[dict]]:
        """Each query's result on the database and on each variant: [[database], [variant 1],
        ...]. Variant k is seeded by "<db_id>:<k>", so it is the same database every time."""
        request = {"db": base64.b64encode(db.read_bytes()).decode(), "queries": queries,
                   "timeout_s": timeout_s, "max_rows": max_rows,
                   "variants": [{"seed": f"{db_id}:{k}", "rows": rows}
                                for k in range(1, variants + 1)]}
        out = self._request(request, budget=timeout_s * len(queries) * (1 + variants) + 30)
        return [out["results"], *out["variants"]]

    def _request(self, request: dict, budget: int) -> dict:
        proc = subprocess.run(["docker", "exec", "-i", self.container, "timeout", str(budget),
                               "python", "/runner.py"], input=json.dumps(request),
                              capture_output=True, text=True, timeout=budget + 30)
        if proc.returncode != 0:  # the sandbox broke, which no query result may hide
            raise RuntimeError(f"sqlite sandbox exited {proc.returncode}: {proc.stderr[-300:]}")
        return json.loads(proc.stdout)


def gold_check(results: list[dict]) -> str | None:
    """Why the gold SQL can't check a reply, or None. `results`: its result on the database and
    on each variant."""
    if not results[0]["ok"]:
        return "gold fails"
    if not all(r["ok"] for r in results):
        return "gold fails on a variant"
    if any(r["more"] for r in results):
        return f"gold returns over {MAX_ROWS} rows"
    rows = [row for r in results for row in r["rows"]]
    if not rows:
        return "gold returns no rows"
    if all(v in _BLANK for row in rows for v in row):
        return "gold returns only NULL, 0 or ''"
    return None


def build(*, reader: Callable[[int], dict | None], size: int, databases: Databases,
          sandbox: SQLiteSandbox, gate: Callable[[dict], str | None] | None, n: int,
          seed: int = SEED) -> tuple[list[dict], dict]:
    """(items, report). `reader(offset)` returns the first whole item after a byte offset of
    data.json; `size` is the file's length."""
    rng = random.Random(seed)
    items: list[dict] = []
    used: set[str] = set()
    rejected: Counter[str] = Counter()
    # Every candidate read, kept or not: next to the kept items' mix, it shows what the filters
    # take out, and `rejected_by_complexity` says which filter.
    read: dict[str, Counter[str]] = {"complexity": Counter(), "style": Counter()}
    by_complexity: dict[str, Counter[str]] = {}
    gate_hits: list[dict] = []  # which battery item or dsbench rule each rejection touched

    def reject(why: str, complexity: str | None) -> None:
        rejected[why] += 1
        by_complexity.setdefault(complexity, Counter())[why] += 1

    offsets = 0
    while len(items) < n:
        offsets += 1
        if offsets > 20 * n + 100:
            raise RuntimeError(f"too few candidates pass: {dict(rejected)}")
        row = reader(rng.randrange(size - CHUNK))
        if row is None:
            rejected["no whole item at the offset"] += 1
            continue
        complexity = row.get("sql_complexity")
        read["complexity"][complexity] += 1
        read["style"][row.get("question_style")] += 1
        if row["db_id"] in used:
            reject("database already used", complexity)
            continue
        if row.get("question_style") in EXCLUDED_STYLES:
            reject(f"style {row['question_style']}", complexity)
            continue
        db = databases.path(row["db_id"])
        prompt, instruction = build_prompt(row, _bird_schema(db), rng)
        if len(prompt) / 3.5 > MAX_PROMPT_TOKENS:
            reject("prompt too long", complexity)
            continue
        item = make_item(row, prompt, instruction)
        reason = gate(item) if gate else None
        if reason:
            reject(f"gate: {reason.split(' ', 1)[0]}", complexity)
            gate_hits.append({"id": item["id"], "reason": reason})
            continue
        gold = [results[0] for results in sandbox.run_suite(db, row["db_id"], [row["sql"]])]
        why = gold_check(gold)
        if why:
            reject(why, complexity)
            continue
        item["verify"]["gold_rows"] = [len(r["rows"]) for r in gold]
        used.add(row["db_id"])
        items.append(item)
        if len(items) % 50 == 0:
            print(f"{len(items)}/{n} items from {offsets} offsets", file=sys.stderr, flush=True)

    tokens = sorted(item["prompt_tokens_est"] for item in items)
    report = {
        "seed": seed, "revision": REVISION, "items": len(items), "offsets_read": offsets,
        "rejected": dict(rejected.most_common()),
        "by_complexity": dict(Counter(i["meta"]["sql_complexity"] for i in items).most_common()),
        "read_by_complexity": dict(read["complexity"].most_common()),
        "rejected_by_complexity": {k: dict(v.most_common())
                                   for k, v in sorted(by_complexity.items(), key=str)},
        "by_style": dict(Counter(i["meta"]["question_style"] for i in items).most_common()),
        "read_by_style": dict(read["style"].most_common()),
        "by_instruction": dict(sorted(Counter(i["meta"]["instruction"] for i in items).items())),
        "prompt_tokens_est": {"median": st.median(tokens) if tokens else 0,
                              "p90": tokens[int(0.9 * (len(tokens) - 1))] if tokens else 0,
                              "max": tokens[-1] if tokens else 0},
        "gold_rows_median": {
            "database": st.median(i["verify"]["gold_rows"][0] for i in items) if items else 0,
            "variants": st.median(n for i in items for n in i["verify"]["gold_rows"][1:])
            if items and VARIANTS else 0},
        "gate_hits": gate_hits,
    }
    return items, report


def same_rows(pred: list[list], gold: list[list]) -> bool:
    """BIRD's execution accuracy: the two results hold the same set of rows."""
    return set(map(tuple, pred)) == set(map(tuple, gold))


def verify(items_path: Path, gen_path: Path, out_path: Path, databases: Databases,
           sandbox: SQLiteSandbox) -> dict:
    """Check each generated reply; a row also needs a finished reply with reasoning. The SQL is
    taken as reasoning_pilot takes Target A's, from a ```sql block, never from bare text.
    `caught_by_variants` counts replies that match on the database but not on a variant."""
    items = {i["id"]: i for i in map(json.loads, items_path.open())}
    passed: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    caught = 0
    with out_path.open("w") as fh:
        for rec in latest_rows(gen_path).values():  # one row an item, retries resolved
            item = items[rec["id"]]
            answer = rec.get("answer") or ""
            sql = sql_only(answer)
            if rec.get("error") or rec.get("finish_reason") != "stop":
                ok, why = False, f"unfinished: {rec.get('error') or rec.get('finish_reason')}"
            elif not (rec.get("reasoning") or "").strip():
                ok, why = False, "no reasoning"
            elif not SQL_BLOCK.search(answer):
                ok, why = False, "no SQL block"
            elif sql is None:
                ok, why = False, "not the SQL only"
            else:
                check = item["verify"]
                suite = sandbox.run_suite(databases.path(check["db_id"]), check["db_id"],
                                          [sql, check["gold_sql"]],
                                          variants=check.get("variants", 0),
                                          rows=check.get("variant_rows", VARIANT_ROWS))
                ok, why = True, ""
                for k, (pred, gold) in enumerate(suite):
                    if not gold["ok"]:
                        raise RuntimeError(f"{item['id']}: the gold SQL failed at verification, "
                                           f"though it ran when the item was built: "
                                           f"{gold['error']}")
                    where = f"variant {k}: " if k else ""
                    if not pred["ok"]:
                        ok, why = False, f"{where}error: {pred['error']}"
                    elif pred["more"]:
                        ok, why = False, f"{where}over {MAX_ROWS} rows"
                    elif not same_rows(pred["rows"], gold["rows"]):
                        n, m = len(pred["rows"]), len(gold["rows"])
                        ok, why = False, where + (f"{n} rows, gold {m}" if n != m
                                                  else "other values")
                    if not ok:
                        caught += k > 0
                        break
            (passed if ok else failed)[item["pool"]] += 1
            fh.write(json.dumps({**rec, "sql": sql, "check": {"ok": ok, "why": why}},
                                ensure_ascii=False) + "\n")
    return {"passed": dict(passed), "failed": dict(failed), "caught_by_variants": caught}


# --- CLI -------------------------------------------------------------------------------------

def _remote_reader() -> tuple[Callable[[int], dict | None], int, Path]:
    """(reader, size of data.json, local path of databases.zip), all at the pinned revision."""
    from huggingface_hub import HfFileSystem, hf_hub_download

    fs = HfFileSystem()
    path = f"datasets/{REPO}@{REVISION}/data.json"
    handle = fs.open(path, block_size=CHUNK)

    def reader(offset: int) -> dict | None:
        nonlocal handle
        for attempt in range(4):
            try:
                handle.seek(offset)
                return first_item(handle.read(CHUNK).decode("utf-8", errors="ignore"))
            except Exception:  # noqa: BLE001 - a dropped connection: the same offset, reopened
                if attempt == 3:
                    raise
                time.sleep(5 * 2 ** attempt)
                handle = fs.open(path, block_size=CHUNK)
        return None  # unreachable: the last attempt returns or raises

    zip_path = Path(hf_hub_download(REPO, "databases.zip", repo_type="dataset",
                                    revision=REVISION))
    return reader, fs.size(path), zip_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--battery-items", required=True, help="a battery run's items/ dir")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--report", type=Path, required=True)
    b.add_argument("--zone", type=Path, default=Path("data/synsql"),
                   help="where the databases the items use are unpacked (git-ignored)")
    b.add_argument("--n", type=int, default=math.ceil(SQL_ROWS / SQL_KEEP))
    b.add_argument("--seed", type=int, default=SEED)
    v = sub.add_parser("verify")
    v.add_argument("--items", type=Path, required=True)
    v.add_argument("--gen", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--zone", type=Path, default=Path("data/synsql"))
    v.add_argument("--zip", type=Path, help="databases.zip (default: the pinned download)")
    args = ap.parse_args()

    sandbox = SQLiteSandbox()
    if not sandbox.available():
        raise SystemExit("the sandbox's sqlite container isn't running: "
                         "docker compose -f sandbox/docker-compose.yml up -d --build sqlite")
    if args.cmd == "verify":
        zip_path = args.zip
        if zip_path is None:
            from huggingface_hub import hf_hub_download
            zip_path = Path(hf_hub_download(REPO, "databases.zip", repo_type="dataset",
                                            revision=REVISION))
        print(json.dumps(verify(args.items, args.gen, args.out,
                                Databases(zip_path, args.zone), sandbox)))
        return

    from dsbench.sftgen.decontaminate import strict_gate

    reader, size, zip_path = _remote_reader()
    items, report = build(reader=reader, size=size, databases=Databases(zip_path, args.zone),
                          sandbox=sandbox, gate=strict_gate(args.battery_items), n=args.n,
                          seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    params = {"sql_rows": SQL_ROWS, "sql_keep": SQL_KEEP, "max_prompt_tokens": MAX_PROMPT_TOKENS,
              "max_rows": MAX_ROWS, "variants": VARIANTS, "variant_rows": VARIANT_ROWS,
              "excluded_styles": list(EXCLUDED_STYLES), "battery_items": args.battery_items}
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.report.write_text(json.dumps({"params": params, **report, "out": str(args.out),
                                       "out_sha256": digest}, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("items", "offsets_read", "rejected")}))


if __name__ == "__main__":
    main()
