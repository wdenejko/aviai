"""Reasoning pilot (ADR-001 Gate 2 item 5): the base model's own thinking-mode traces, measured.

WHY: the retrain trains with thinking on, and the Gate-2 mixture's reasoning is too short to teach
a thinking mode (Target A's blocks are one templated sentence, ~46 tokens). The plan is to
regenerate the reasoning with the base itself in thinking mode and keep a row only where its answer
verifies: rejection sampling on the model's own distribution, which also keeps the base's reasoning
LENGTH, the thing whose loss cost the Gate-2 adapter MMLU-Pro, GPQA and LiveCodeBench points.
Before that runs at scale, a pilot measures what shapes it:

  - how long the traces are, per pool: they set the training sequence length (4096 or 8192; both
    are measured in reports/gate-evals/20260928-seq4096-enablement.md);
  - how often an answer verifies (Target A on ClickHouse, the owner's engine): the price of
    rejection sampling;
  - the box's generation throughput with thinking on: how long a full regeneration takes.

Subcommands, each run where its dependencies live:

    items     (Mac)  pick the pilot prompts from the Gate-2 mixture
    generate  (box)  query a llama-server in thinking mode; resumable; stdlib + httpx only, so it
                     runs in the box's lean batteryvenv
    verify    (Mac)  execute Target A's ClickHouse answers in the dsbench sandbox against the truth
    report    (Mac)  lengths, fits, throughput and yield, as JSON and a Markdown summary
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import re
import statistics as st
import threading
import time
from collections import defaultdict
from pathlib import Path

# Pilot size per pool: single-turn rows only (a multi-turn agent trajectory needs a real rollout
# with tools, not one completion). Target A is split by dialect with ClickHouse doubled: it is the
# only dialect verified here, since its sandbox is where model-written SQL already runs.
POOLS = {"targetA": 80, "gretel_sql": 32, "tulu3": 40, "opencoder_edu": 32, "swe_swiss": 16}
TARGET_A_DIALECTS = {"clickhouse": 32, "postgres": 16, "mysql": 16, "duckdb": 16}
SEED = 20260928
# Target A's prompt asks for "a single SQL query ..., then the numeric result". The model never
# sees the data, so any number it writes is invented, and the Gate-2 adapter learned to invent one
# (reports/gate-evals/20260924-gguf-export.md). The retrain asks for the SQL only, and so does
# the pilot.
NUMERIC_TAIL = ", then the numeric result."
# Qwen's recommended thinking-mode sampling: greedy decoding loops in thinking mode, and the point
# is the model's own distribution of traces, not its single most likely one.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
# The Gate-2 Target A slice was generated with `--n 1500` rows per synthetic table, not the CLI's
# default 4000: found by reproducing the stored truths (every probed row matched at 1500 only).
TARGET_A_ROWS = 1500
_SQL_BLOCK = re.compile(r"```[ \t]*sql[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


# --- items -------------------------------------------------------------------------------------


def _row_id(row: dict) -> str:
    """The mixture's own id, else a hash of the prompt (gretel, opencoder, swe rows have none)."""
    if row["meta"].get("id"):
        return str(row["meta"]["id"])
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    return hashlib.sha1(user.encode()).hexdigest()[:12]


def _single_turn(row: dict) -> bool:
    roles = [m["role"] for m in row["messages"]]
    return roles.count("user") == 1 and roles.count("assistant") == 1 and not row.get("tools")


def last_sql(text: str) -> str | None:
    """The last ```sql block of a reply (a model may draft one and then correct it)."""
    blocks = _SQL_BLOCK.findall(text or "")
    return blocks[-1].strip() if blocks else None


def make_item(pool: str, row: dict) -> dict:
    prompt = [{"role": m["role"], "content": m["content"]}
              for m in row["messages"] if m["role"] != "assistant"]
    item = {"id": f"{pool}:{_row_id(row)}", "pool": pool, "messages": prompt}
    if pool == "targetA":
        system = prompt[0]
        if system["role"] != "system" or not system["content"].endswith(NUMERIC_TAIL):
            raise ValueError(f"unexpected Target A system prompt in {item['id']}")
        system["content"] = system["content"][: -len(NUMERIC_TAIL)] + "."
        gold = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        item["verify"] = {"dialect": row["meta"]["dialect"],
                          "truth": row["meta"]["verification"]["truth"],
                          "gold_sql": last_sql(gold), "row_id": row["meta"]["id"]}
    return item


def build_items(rows: list[dict], *, seed: int = SEED) -> list[dict]:
    """A seeded, per-pool sample of single-turn prompts; the same rows every run."""
    rng = random.Random(seed)
    by_pool: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if _single_turn(row):
            by_pool[row["meta"]["mix_pool"]].append(row)
    items = []
    for pool, n in POOLS.items():
        candidates = sorted(by_pool[pool], key=_row_id)  # a stable order before sampling
        if pool == "targetA":
            chosen = []
            for dialect, k in TARGET_A_DIALECTS.items():
                chosen += rng.sample([r for r in candidates if r["meta"]["dialect"] == dialect], k)
        else:
            chosen = rng.sample(candidates, n)
        items += [make_item(pool, row) for row in chosen]
    return items


# --- generate ----------------------------------------------------------------------------------


def split_reasoning(message: dict) -> tuple[str, str]:
    """(reasoning, answer). llama-server puts the trace in `reasoning_content`; if a build leaves
    it inline, the answer starts after the first </think>."""
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    if not reasoning and "</think>" in content:
        reasoning, _, content = content.partition("</think>")
        reasoning = reasoning.replace("<think>", "", 1)
    return reasoning.strip(), content.strip()


def request_body(item: dict, max_tokens: int) -> dict:
    # A fixed per-item seed makes a rerun reproduce the same sample; `enable_thinking` overrides
    # the server's thinking-off default (battery_server.sh) for these requests only.
    seed = int(hashlib.sha1(item["id"].encode()).hexdigest()[:8], 16)
    return {"model": "base", "messages": item["messages"], **SAMPLING, "seed": seed,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": True}}


def _count_tokens(client, text: str) -> int:
    if not text:
        return 0
    r = client.post("/tokenize", json={"content": text})
    r.raise_for_status()
    return len(r.json()["tokens"])


def run_one(client, item: dict, max_tokens: int) -> dict:
    record = {"id": item["id"], "pool": item["pool"], "error": ""}
    started = time.time()
    try:
        r = client.post("/v1/chat/completions", json=request_body(item, max_tokens))
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        reasoning, answer = split_reasoning(choice.get("message") or {})
        usage = data.get("usage") or {}
        record.update(
            reasoning=reasoning, answer=answer, finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=_count_tokens(client, reasoning),
            answer_tokens=_count_tokens(client, answer),
            decode_tok_s=(data.get("timings") or {}).get("predicted_per_second"),
        )
    except Exception as exc:  # noqa: BLE001 - recorded, and the item is retried on resume
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
    record.update(started=round(started, 3), finished=round(time.time(), 3))
    return record


def generate(items_path: Path, out_path: Path, base_url: str, workers: int,
             max_tokens: int) -> None:
    import httpx  # the box's batteryvenv has it; imported here so `items` needs nothing extra

    items = [json.loads(line) for line in items_path.open()]
    done = set()
    if out_path.exists():
        done = {r["id"] for r in map(json.loads, out_path.open()) if not r.get("error")}
    todo = [item for item in items if item["id"] not in done]
    print(f"{len(items)} items, {len(done)} already done, {len(todo)} to generate", flush=True)
    local = threading.local()
    lock = threading.Lock()

    def one(item: dict) -> dict:
        if not hasattr(local, "client"):  # one connection per worker thread
            local.client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(3600.0))
        return run_one(local.client, item, max_tokens)

    with out_path.open("a") as fh, cf.ThreadPoolExecutor(workers) as pool:
        for n, record in enumerate(pool.map(one, todo), 1):
            with lock:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
            print(f"[{n}/{len(todo)}] {record['id']} {record.get('finish_reason')} "
                  f"reasoning={record.get('reasoning_tokens')} {record['error']}", flush=True)


# --- verify ------------------------------------------------------------------------------------


def parse_target_a_id(row_id: str) -> tuple[str, str, int]:
    """(dialect, domain, seed) from 'A-<family>-<dialect>-<domain>-<seed>'; the family has
    hyphens, the dialect and the domain don't."""
    parts = row_id.split("-")
    return parts[-3], parts[-2], int(parts[-1])


def verify(items_path: Path, gen_path: Path, out_path: Path, dialect: str = "clickhouse") -> None:
    """Run each Target A answer in its dialect's sandbox engine on the row's own synthetic data.

    The data is rebuilt from the row's seed, and the gold SQL must reproduce the stored truth on
    it before a model answer is judged, so a mismatch in the rebuild can't pass as a model error.
    """
    from dsbench.sftgen import synth
    from dsbench.sftgen.engines import available_engines

    items = {i["id"]: i for i in map(json.loads, items_path.open()) if i["pool"] == "targetA"}
    gens = {g["id"]: g for g in map(json.loads, gen_path.open()) if not g.get("error")}
    engines = available_engines(only=[dialect])
    if not engines:
        raise SystemExit(f"no {dialect} engine reachable (is the dsbench sandbox up?)")
    engine = engines[0]
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for item in items.values():
        if item["verify"]["dialect"] == dialect and item["id"] in gens:
            _, domain, seed = parse_target_a_id(item["verify"]["row_id"])
            groups[(domain, seed)].append(item)
    results = []
    for (domain_name, seed), group in sorted(groups.items()):
        domain = synth.build(domain_name, seed, TARGET_A_ROWS)
        engine.setup()
        try:
            engine.load(domain.name, domain.df)
            for item in group:
                truth = item["verify"]["truth"]
                result = {"id": item["id"], "truth": truth}
                if engine.scalar(item["verify"]["gold_sql"]) != truth:
                    result["status"] = "rebuild_mismatch"  # the data, not the model, is wrong
                else:
                    sql = last_sql(gens[item["id"]]["answer"])
                    if sql is None:
                        result["status"] = "no_sql"
                    else:
                        try:
                            got = engine.scalar(sql)
                            result.update(got=got,
                                          status="verified" if got == truth else "wrong")
                        except Exception as exc:  # noqa: BLE001 - an engine error is a wrong answer
                            result.update(status="error", detail=str(exc)[:200])
                results.append(result)
        finally:
            engine.teardown()
    out_path.write_text("".join(json.dumps(r, default=str) + "\n" for r in results))
    counts = defaultdict(int)
    for r in results:
        counts[r["status"]] += 1
    print(dict(counts))


# --- report ------------------------------------------------------------------------------------


def _pct(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(gens: list[dict], verified: list[dict] | None = None) -> dict:
    """Per pool: trace and sequence lengths, how many fit 4096 and 8192 tokens, the limit hits;
    overall: generation throughput; Target A: the verification yield."""
    ok = [g for g in gens if not g.get("error")]
    pools: dict[str, dict] = {}
    for pool in sorted({g["pool"] for g in ok}):
        rows = [g for g in ok if g["pool"] == pool]
        reasoning = [g["reasoning_tokens"] for g in rows]
        # a training row = prompt (with the template's generation prompt) + completion + <|im_end|>
        sequence = [g["prompt_tokens"] + g["completion_tokens"] + 1 for g in rows]
        pools[pool] = {
            "n": len(rows),
            "hit_limit": sum(g["finish_reason"] == "length" for g in rows),
            "reasoning_tokens": {"median": st.median(reasoning), "p90": _pct(reasoning, 0.9),
                                 "max": max(reasoning)},
            "answer_tokens_median": st.median(g["answer_tokens"] for g in rows),
            "sequence_tokens": {"median": st.median(sequence), "p90": _pct(sequence, 0.9),
                                "max": max(sequence)},
            "fits_4096": sum(s <= 4096 for s in sequence),
            "fits_8192": sum(s <= 8192 for s in sequence),
        }
    wall = max(g["finished"] for g in ok) - min(g["started"] for g in ok) if ok else 0
    out = {"pools": pools, "errors": len(gens) - len(ok),
           "generation": {"completion_tokens": sum(g["completion_tokens"] for g in ok),
                          "wall_seconds": round(wall, 1),
                          "tokens_per_second": round(sum(g["completion_tokens"] for g in ok)
                                                     / wall, 1) if wall else None,
                          "decode_tok_s_per_slot_median": st.median(
                              g["decode_tok_s"] for g in ok if g.get("decode_tok_s"))}}
    if verified is not None:
        counts: dict[str, int] = defaultdict(int)
        for v in verified:
            counts[v["status"]] += 1
        judged = sum(n for s, n in counts.items() if s != "rebuild_mismatch")
        out["target_a_clickhouse"] = {"counts": dict(counts), "judged": judged,
                                      "yield": round(counts["verified"] / judged, 3) if judged
                                      else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("items")
    s.add_argument("--mixture", type=Path, default=Path("data/sft/gate2_mixture.jsonl"))
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("generate")
    s.add_argument("--items", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--base-url", default="http://127.0.0.1:8093")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--max-tokens", type=int, default=16384)
    s = sub.add_parser("verify")
    s.add_argument("--items", type=Path, required=True)
    s.add_argument("--gen", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("report")
    s.add_argument("--gen", type=Path, required=True)
    s.add_argument("--verified", type=Path)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "items":
        rows = [json.loads(line) for line in args.mixture.open()]
        items = build_items(rows)
        args.out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
        print(f"{len(items)} items -> {args.out}")
    elif args.cmd == "generate":
        generate(args.items, args.out, args.base_url, args.workers, args.max_tokens)
    elif args.cmd == "verify":
        verify(args.items, args.gen, args.out)
    else:
        gens = [json.loads(line) for line in args.gen.open()]
        verified = ([json.loads(line) for line in args.verified.open()]
                    if args.verified else None)
        summary = summarize(gens, verified)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
