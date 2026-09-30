"""Run SQL on one SQLite database, read-only, in a fresh process, and on bigger variants of it.

    docker exec -i avbench-sqlite timeout 120 python /runner.py < request.json

stdin:  {"db": base64 SQLite bytes, "queries": [sql, ...], "timeout_s": 30, "max_rows": 1000,
         "variants": [{"seed": "shop:1", "rows": 30}, ...]}     ("variants" is optional)
stdout: {"results": [result, ...], "variants": [[result, ...], ...]}
        result: {"ok": true, "rows": [[...], ...], "more": false} | {"ok": false, "error": msg}

The database is written to the container's tmpfs and opened read-only and immutable, and an
authorizer refuses ATTACH, so a query can read this database and nothing else. That is the second
wall; the first is the container itself (no network, a read-only root, no host mounts). Each query
gets `timeout_s` seconds (a progress handler interrupts it) and returns at most `max_rows` rows;
`more` says whether it had others.

A variant is a bigger copy of the database: its tables and rows, plus new rows drawn from `seed`
up to `rows` per table. SynSQL's databases hold about two rows a table, and on them a wrong query
often returns the right rows: AVG and SUM agree on one row. Comparing results on bigger databases
too is test-suite accuracy (Zhong, Yu & Klein, 2020). Variants are built here, not on the host,
because building one runs the dataset's CREATE TABLE statements. Called by sftgen/synsql.py.
"""

import base64
import json
import os
import random
import re
import sqlite3
import sys
import tempfile
import time
from datetime import date, timedelta

_REFUSED = (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH)
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})(.*)", re.DOTALL)
# 0/1 columns named as flags. Other 0/1 columns are ids or counts: SynSQL numbers rows from 0.
_FLAG = re.compile(r"^(is|has|can|should|allow|allows|enable|enabled|active|visible)(_|$)"
                   r"|_(flag|active|enabled)$", re.IGNORECASE)
KEEP_SHARE = 0.5  # a new row repeats a value its column already holds this often
NULL_SHARE = 0.1  # a nullable column is NULL in a new row this often
FK_SKEW = 0.3  # the rate of the exponential that picks a parent: most children go to a few


def _plain(value):
    """A JSON-safe cell: BLOBs become hex text, everything else is already JSON."""
    return "0x" + value.hex() if isinstance(value, bytes) else value


def _authorize(action, *_):
    return sqlite3.SQLITE_DENY if action in _REFUSED else sqlite3.SQLITE_OK


def _connect(path: str, *, read_only: bool) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro&immutable=1" if read_only else f"file:{path}"
    con = sqlite3.connect(uri, uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    con.set_authorizer(_authorize)
    return con


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def run_queries(path: str, queries: list[str], timeout_s: float, max_rows: int) -> list[dict]:
    results = []
    for sql in queries:
        con = _connect(path, read_only=True)
        deadline = time.monotonic() + timeout_s
        con.set_progress_handler(lambda end=deadline: time.monotonic() > end, 10_000)
        try:
            rows = con.execute(sql).fetchmany(max_rows + 1)
            results.append({"ok": True, "rows": [[_plain(v) for v in row]
                                                 for row in rows[:max_rows]],
                            "more": len(rows) > max_rows})
        except Exception as exc:  # noqa: BLE001 - any failure of the query is the answer
            results.append({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
        finally:
            con.close()
    return results


# --- variants ----------------------------------------------------------------------------------

def make_variant(src: str, dst: str, seed: str, rows: int) -> None:
    """Build a bigger copy of the database at `dst`: its schema and rows, and new rows drawn from
    `seed` up to `rows` per table. The same seed always builds the same database."""
    rng = random.Random(seed)
    source = _connect(src, read_only=True)
    target = _connect(dst, read_only=False)
    tables = source.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table' "
                            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    for _, ddl in tables:
        target.execute(ddl)
    keys = _keys(source, [name for name, _ in tables])
    for name in _parents_first([name for name, _ in tables], keys):
        _fill(source, target, name, keys[name.lower()], rng, rows)
    target.commit()
    target.close()
    source.close()


def _keys(source, tables: list[str]) -> dict[str, dict[str, tuple[str, str | None]]]:
    """{table: {column: (parent table, parent column)}}: the declared foreign keys, and a column
    named as another table's one-column primary key (`game_id` in `game_genres`), which SynSQL's
    junction tables often leave undeclared."""
    single_pk: dict[str, list[tuple[str, str]]] = {}
    columns = {}
    for table in tables:
        columns[table] = source.execute(f"PRAGMA table_info({_q(table)})").fetchall()
        pk = [c[1] for c in columns[table] if c[5]]
        if len(pk) == 1:
            single_pk.setdefault(pk[0].lower(), []).append((table, pk[0]))
    keys = {}
    for table in tables:
        declared = {r[3]: (r[2], r[4]) for r in
                    source.execute(f"PRAGMA foreign_key_list({_q(table)})")}
        own_pk = [c[1] for c in columns[table] if c[5]]
        for col in columns[table]:
            owners = single_pk.get(col[1].lower(), [])
            if col[1] not in declared and own_pk != [col[1]] and len(owners) == 1:
                declared[col[1]] = owners[0]
        keys[table.lower()] = declared
    return keys


def _parents_first(names: list[str], keys: dict) -> list[str]:
    """Each table after the tables its foreign keys point to; a cycle is cut where it is met."""
    by_lower = {n.lower(): n for n in names}
    order: list[str] = []

    def visit(name: str, path: set[str]) -> None:
        if name in order or name in path:
            return
        path.add(name)
        for parent, _ in keys[name.lower()].values():
            found = by_lower.get(str(parent).lower())
            if found and found != name:
                visit(found, path)
        order.append(name)

    for name in names:
        visit(name, set())
    return order


def _fill(source, target, table: str, fks: dict, rng: random.Random, rows: int) -> None:
    cols = source.execute(f"PRAGMA table_info({_q(table)})").fetchall()  # cid name type nn dflt pk
    original = source.execute(f"SELECT * FROM {_q(table)}").fetchall()
    insert = f"INSERT INTO {_q(table)} VALUES ({', '.join('?' * len(cols))})"
    target.executemany(insert, original)
    pk = [c for c in cols if c[5]]
    pools = [[row[c[0]] for row in original if row[c[0]] is not None] for c in cols]
    for n in range(len(original), rows):
        for attempt in range(10):  # a CHECK or a key clash: draw the row again
            values = [_draw(target, table, c, len(pk), pools[c[0]], fks.get(c[1]), n, rng,
                            attempt) for c in cols]
            try:
                target.execute(insert, values)
                break
            except sqlite3.IntegrityError:
                continue


def _draw(target, table: str, col: tuple, pk_width: int, pool: list, fk: tuple | None, n: int,
          rng: random.Random, attempt: int):
    _, name, declared, notnull, _, in_pk = col
    if in_pk and pk_width == 1:
        return _new_key(target, table, name, declared, pool, n, rng)
    if fk:
        parents = _values(target, *fk)
        if parents and attempt:  # the skewed pick clashed with a key: any parent
            return rng.choice(parents)
        if parents:
            # A few parents take most children and some take none: groups of several rows,
            # where AVG and SUM differ, and parents with no children, where INNER and LEFT JOIN do.
            return parents[min(int(rng.expovariate(FK_SKEW)), len(parents) - 1)]
    if not notnull and not in_pk and rng.random() < NULL_SHARE:
        return None
    return _new_value(declared.upper(), name, pool, rng)


def _values(target, table: str, column: str | None) -> list:
    """The values a foreign key can point at, in row order ([] if the parent doesn't exist)."""
    try:
        if column is None:  # REFERENCES parent: its primary key
            pks = [c[1] for c in target.execute(f"PRAGMA table_info({_q(table)})") if c[5]]
            if len(pks) != 1:
                return []
            column = pks[0]
        return [r[0] for r in target.execute(f"SELECT {_q(column)} FROM {_q(table)} "
                                             f"WHERE {_q(column)} IS NOT NULL")]
    except sqlite3.OperationalError:
        return []


def _new_key(target, table: str, name: str, declared: str, pool: list, n: int,
             rng: random.Random):
    if "INT" in declared.upper():
        top = target.execute(f"SELECT max({_q(name)}) FROM {_q(table)}").fetchone()[0]
        return (top if isinstance(top, int) else n) + 1
    return f"{rng.choice(pool) if pool else name}-{n}"


def _new_value(declared: str, name: str, pool: list, rng: random.Random):
    """Often a value the column already holds, so the question's literals still match rows;
    otherwise a new one of the same kind."""
    if pool and rng.random() < KEEP_SHARE:
        return rng.choice(pool)
    sample = rng.choice(pool) if pool else None
    if isinstance(sample, int) or (sample is None and "INT" in declared):
        ints = [v for v in pool if isinstance(v, int)] or [0, 100]
        lo, hi = min(ints), max(ints)
        if lo >= 0 and hi <= 1 and _FLAG.search(name):
            return rng.randint(0, 1)
        span = max(10, hi - lo)
        return rng.randint(max(0, lo - span) if lo >= 0 else lo - span, hi + span)
    if isinstance(sample, float) or (sample is None and any(t in declared for t in
                                                            ("REAL", "FLOA", "DOUB"))):
        nums = [v for v in pool if isinstance(v, (int, float))] or [0.0, 100.0]
        lo, hi = min(nums), max(nums)
        span = max(1.0, hi - lo)
        return round(rng.uniform(max(0.0, lo - span) if lo >= 0 else lo - span, hi + span), 2)
    if isinstance(sample, str):
        day = _ISO_DATE.match(sample)
        if day:
            try:
                moved = date(int(day[1]), int(day[2]), int(day[3])) + timedelta(
                    days=rng.randint(-365, 365))
            except ValueError:  # not a real date after all
                return sample
            return moved.isoformat() + day[4]
        return f"{sample} {rng.randint(2, 99)}"
    if sample is not None:  # a BLOB
        return sample
    return f"{name} {rng.randint(1, 99)}"


# --- the request -------------------------------------------------------------------------------

def run(request: dict) -> dict:
    timeout_s = request.get("timeout_s", 30)
    max_rows = request.get("max_rows", 1000)
    queries = request["queries"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "db.sqlite")
        with open(path, "wb") as handle:
            handle.write(base64.b64decode(request["db"]))
        out = {"results": run_queries(path, queries, timeout_s, max_rows)}
        if "variants" in request:
            out["variants"] = []
            for k, spec in enumerate(request["variants"]):
                variant = os.path.join(tmp, f"variant{k}.sqlite")
                try:
                    make_variant(path, variant, spec["seed"], spec["rows"])
                except Exception as exc:  # noqa: BLE001 - reported per query, like a failure
                    error = f"variant: {type(exc).__name__}: {exc}"[:300]
                    out["variants"].append([{"ok": False, "error": error} for _ in queries])
                    continue
                out["variants"].append(run_queries(variant, queries, timeout_s, max_rows))
        return out


def main() -> None:
    print(json.dumps(run(json.load(sys.stdin))))


if __name__ == "__main__":
    main()
