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
                     runs in the box's lean batteryvenv. An item with a `prefill` has the model
                     continue its thinking from it (sftgen/prefill.py). Revision 2's volume runs
                     use it too, with `--block`: no reply runs past what its training row can hold
    verify    (Mac)  execute Target A's ClickHouse answers in the dsbench sandbox against the truth
    report    (Mac)  lengths, fits, throughput and yield, as JSON and a Markdown summary
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
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
# is the model's own distribution of traces, not its single most likely one. min_p is llama-server's
# default, 0.05, which every Revision 2 generation so far sampled with because none sent one (the
# battery server's /props, 2026-10-02): the pilots, the prefill pilot and Target C's 154 rows.
# Qwen recommends 0, and the mini-battery measures the base and its checkpoints at 0
# (battery/generate.py). Sent explicitly now, so a server whose default differs can't change the
# data unnoticed.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.05}
# The Gate-2 Target A slice was generated with `--n 1500` rows per synthetic table, not the CLI's
# default 4000: found by reproducing the stored truths (every probed row matched at 1500 only).
TARGET_A_ROWS = 1500
SQL_BLOCK = re.compile(r"```[ \t]*sql[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


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
    blocks = SQL_BLOCK.findall(text or "")
    return blocks[-1].strip() if blocks else None


def sql_only(answer: str) -> str | None:
    """The reply's SQL if the reply is one ```sql block and nothing else, as Revision 2's SQL
    prompts ask. A row whose answer adds prose, or drafts two queries, would teach the model to
    ignore "only"."""
    blocks = SQL_BLOCK.findall(answer or "")
    if len(blocks) != 1 or SQL_BLOCK.sub("", answer).strip():
        return None
    return blocks[0].strip()


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


def _seed(item: dict) -> int:
    # A fixed per-item seed makes a rerun reproduce the same sample.
    return int(hashlib.sha1(item["id"].encode()).hexdigest()[:8], 16)


def request_body(item: dict, max_tokens: int) -> dict:
    # `enable_thinking` overrides the server's thinking-off default (battery_server.sh) for these
    # requests only.
    body = {"model": "base", "messages": item["messages"], **SAMPLING, "seed": _seed(item),
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": True}}
    if item.get("tools"):
        # Tool rows (sftgen/tool_rows.py) stream, as the battery's BFCL items do: the fork's
        # non-stream endpoint 500s on a malformed call, and a streamed one comes back as text.
        body.update(tools=item["tools"], parallel_tool_calls=True, stream=True,
                    stream_options={"include_usage": True})
    return body


# A prefilled item (sftgen/prefill.py) can't go through the chat endpoint: the chat template closes
# an assistant message's thinking block, and a prefill has to leave it open for the base to
# continue. So the server's own template renders the prompt (/apply-template, thinking on, the
# same rendering the chat endpoint does), the prefill follows the open <think>, and /completion
# continues the text. llama-server prints a special token only when the request preserves it, as
# the chat endpoint does for the template's own tokens. So the request preserves <think> and
# </think>, and the reply splits at </think>, as the chat endpoint's parser would split it.
THINK_TOKENS = ["<think>", "</think>"]
OPEN_THINKING = "<think>\n"


def render_prompt(client, item: dict) -> str:
    """The item's prompt as the server's chat template renders it with thinking on, ending in the
    open thinking block."""
    r = client.post("/apply-template", json={"messages": item["messages"],
                                              "chat_template_kwargs": {"enable_thinking": True}})
    r.raise_for_status()
    prompt = r.json()["prompt"]
    if not prompt.endswith(OPEN_THINKING):
        raise ValueError(f"the template didn't open a thinking block: ...{prompt[-40:]!r}")
    return prompt


def completion_body(item: dict, prompt: str, max_tokens: int) -> dict:
    return {"prompt": prompt + item["prefill"], "n_predict": max_tokens, **SAMPLING,
            "seed": _seed(item), "cache_prompt": True, "preserved_tokens": THINK_TOKENS}


def split_completion(prefill: str, text: str) -> tuple[str, str]:
    """(reasoning, answer) of a prefilled completion. The reasoning is the prefill and what the
    model wrote before </think>; the answer is what it wrote after. A reply that never closed its
    thinking (it ran out of tokens) is all reasoning."""
    reasoning, _, answer = (prefill + text).partition("</think>")
    return reasoning.strip(), answer.strip()


def finish_reason(stop_type: str | None) -> str:
    """/completion's stop type as the chat endpoint's finish reason: the end of the turn or a stop
    word is "stop"; the token limit, or anything else, counts as unfinished."""
    return "stop" if stop_type in ("eos", "word") else "length"


def _complete_prefilled(client, item: dict, max_tokens: int) -> tuple[str, str, str, dict, dict]:
    """(reasoning, answer, finish_reason, usage, timings) of a prefilled item."""
    body = completion_body(item, render_prompt(client, item), max_tokens)
    r = client.post("/completion", json=body)
    r.raise_for_status()
    data = r.json()
    reasoning, answer = split_completion(item["prefill"], data.get("content") or "")
    # The prompt's tokens include the prefill's, so prompt + completion is still the whole row.
    usage = {"prompt_tokens": data.get("tokens_evaluated"),
             "completion_tokens": data.get("tokens_predicted")}
    return reasoning, answer, finish_reason(data.get("stop_type")), usage, data.get("timings") or {}


def _count_tokens(client, text: str) -> int:
    if not text:
        return 0
    r = client.post("/tokenize", json={"content": text})
    r.raise_for_status()
    return len(r.json()["tokens"])


# A training row holds the prompt, the reply, a newline after the reply's <|im_end|> and the
# separator that follows the row in its block: 2 tokens past the reply. The reply is rendered again
# for training, its whitespace trimmed, so its token count can differ from the server's by a few.
# The slack covers that: a reply the budget stops could not have fit. Assembly decides the fit
# (tokenize_masked.thinking_record); the budget only saves the tokens a reply spends past it.
BLOCK_SLACK = 16


def rendered_prompt_tokens(client, item: dict) -> int:
    """The tokens before the model's own text in the item's training row: its prompt as the server
    renders it, thinking on and its tools listed, plus any prefill."""
    body = {"messages": item["messages"], "chat_template_kwargs": {"enable_thinking": True}}
    if item.get("tools"):
        body["tools"] = item["tools"]
    r = client.post("/apply-template", json=body)
    r.raise_for_status()
    return _count_tokens(client, r.json()["prompt"] + item.get("prefill", ""))


def reply_budget(client, item: dict, max_tokens: int, block: int) -> tuple[int, int | None]:
    """(the reply's token limit, the prompt's tokens or None). With a block, a reply may run only
    as far as its row could still fit: the volume runs drop every row over the block, and a reply
    that loops would otherwise run to `max_tokens` for nothing (the mini-battery's calibration:
    7 of 120 BFCL replies looped to 12,288 tokens and took 43.5% of the pass's tokens). If the
    prompt can't be counted, the reply keeps `max_tokens`: a wasted reply, never a lost row."""
    if not block:
        return max_tokens, None
    try:
        used = rendered_prompt_tokens(client, item)
    except Exception:  # noqa: BLE001 - counted as unmeasured in the record, and generated anyway
        return max_tokens, None
    return min(max_tokens, block - used + BLOCK_SLACK), used


def _complete(client, body: dict) -> tuple[dict, str | None, dict, dict]:
    """(message, finish_reason, usage, timings), whether the request streams or not."""
    if body.get("stream"):
        from dsbench.streaming import reassemble_stream  # only tool items stream

        with client.stream("POST", "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            out = reassemble_stream(r)
        return out["message"], out["finish_reason"], out["usage"], out["timings"]
    r = client.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    return (choice.get("message") or {}, choice.get("finish_reason"), data.get("usage") or {},
            data.get("timings") or {})


def run_one(client, item: dict, max_tokens: int, block: int = 0) -> dict:
    record = {"id": item["id"], "pool": item["pool"], "error": ""}
    started = time.time()
    try:
        budget, rendered = reply_budget(client, item, max_tokens, block)
        if block:
            record.update(max_tokens=budget, rendered_prompt_tokens=rendered)
            if rendered is not None and rendered >= block:  # no reply can fit after the prompt
                record.update(finish_reason="prompt_over_block", started=round(started, 3),
                              finished=round(time.time(), 3))
                return record
        if "prefill" in item:  # an empty prefill too: a plain reply, rendered the same way
            message = {}
            reasoning, answer, finish, usage, timings = _complete_prefilled(client, item, budget)
        else:
            message, finish, usage, timings = _complete(client, request_body(item, budget))
            reasoning, answer = split_reasoning(message)
        record.update(
            reasoning=reasoning, answer=answer, finish_reason=finish,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=_count_tokens(client, reasoning),
            answer_tokens=_count_tokens(client, answer),
            decode_tok_s=timings.get("predicted_per_second"),
        )
        if message.get("tool_calls"):
            record["tool_calls"] = message["tool_calls"]
    except Exception as exc:  # noqa: BLE001 - recorded, and the item is retried on resume
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
    record.update(started=round(started, 3), finished=round(time.time(), 3))
    return record


def latest_rows(out_path: Path) -> dict[str, dict]:
    """Each item's row in a generation file: its last without an error, else its last. An item
    that failed is tried again when a run resumes, so it can have several rows; a checker reads
    this, never the raw lines. A line that doesn't parse (a row torn by a power cut: a signal
    can't tear one, since each row is one write) is skipped, and its item runs again."""
    rows: dict[str, dict] = {}
    torn = 0
    if out_path.exists():
        for line in out_path.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                torn += 1
                continue
            if row["id"] not in rows or not row.get("error") or rows[row["id"]].get("error"):
                rows[row["id"]] = row
    if torn:
        print(f"skipped {torn} torn line(s) in {out_path}: those items run again", flush=True)
    return rows


def answered(out_path: Path) -> set[str]:
    """Ids already answered without an error: a resumed run skips them."""
    return {i for i, row in latest_rows(out_path).items() if not row.get("error")}


def generate(items_path: Path, out_path: Path, base_url: str, workers: int,
             max_tokens: int, block: int = 0) -> None:
    import httpx  # the box's batteryvenv has it; imported here so `items` needs nothing extra

    from dsbench.battery.items import append_row

    items = [json.loads(line) for line in items_path.open()]
    done = answered(out_path)
    todo = [item for item in items if item["id"] not in done]
    print(f"{len(items)} items, {len(done)} already done, {len(todo)} to generate", flush=True)
    local = threading.local()

    def one(item: dict) -> dict:
        if not hasattr(local, "client"):  # one connection per worker thread
            local.client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(3600.0))
        return run_one(local.client, item, max_tokens, block)

    fd = os.open(out_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size:
        with out_path.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                os.write(fd, b"\n")  # a torn last row stays a line of its own, which is skipped
    try:
        # Written as each reply finishes, not in item order: a window's time limit stops the run,
        # and a finished reply held back behind a slow one would be lost with it.
        with cf.ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(one, item) for item in todo]
            for n, future in enumerate(cf.as_completed(futures), 1):
                record = future.result()
                append_row(fd, record)
                print(f"[{n}/{len(todo)}] {record['id']} {record.get('finish_reason')} "
                      f"reasoning={record.get('reasoning_tokens')} "
                      f"budget={record.get('max_tokens', max_tokens)} {record['error']}",
                      flush=True)
    finally:
        os.close(fd)


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
    engines = available_engines(only=[dialect], sandboxed=True)  # model SQL
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
    s.add_argument("--block", type=int, default=0,
                   help="the training block (8192): a reply stops where its row would outgrow it")
    s = sub.add_parser("verify")
    s.add_argument("--items", type=Path, required=True)
    s.add_argument("--gen", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--dialect", default="clickhouse",
                   choices=["clickhouse", "postgres", "mysql", "duckdb"])
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
        generate(args.items, args.out, args.base_url, args.workers, args.max_tokens, args.block)
    elif args.cmd == "verify":
        verify(args.items, args.gen, args.out, args.dialect)
    else:
        gens = [json.loads(line) for line in args.gen.open()]
        verified = ([json.loads(line) for line in args.verified.open()]
                    if args.verified else None)
        summary = summarize(gens, verified)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
