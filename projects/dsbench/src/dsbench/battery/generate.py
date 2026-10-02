"""Run one benchmark's items through the server in one state, and save every reply.

A pass = (items file, state). The state picks the LoRA scale per request, so base and adapter hit
the same loaded model. The output is append-only JSONL with one row per item, so a crashed or
interrupted pass resumes where it stopped: ids already answered without error are skipped.

Requests are greedy (temperature 0, fixed seed) with thinking off, unless the item asks for
thinking (`gen["thinking"]`, the mini-battery's items, `mini.py`):
- Thinking on, the request samples with Qwen's settings for thinking mode, as the box generates
  Revision 2's data. Greedy decoding with thinking on falls into repetition (Qwen's model card).
- Every state draws the same seed for an item, so base and adapter start from the same random
  numbers: where the two models agree, so do their samples. The A/A pass (`base_rep`) draws
  another seed, so its flips are the full sampling noise, the most a comparison can face once
  long reasoning has diverged.
- The row keeps the reasoning, and whether it ever closed (`unclosed`).

Tool-calling items stream and are reassembled client-side (`dsbench.streaming`), because the
fork's non-stream endpoint 500s on a malformed tool call, and for BFCL a malformed call is a result
to score, not a transport error.

Run (on the box, next to the server):
    python -m dsbench.battery.generate --items ITEMS/ifeval.jsonl --state base \
        --out GEN/ifeval.base.jsonl --base-url http://127.0.0.1:8093 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from dsbench.battery.items import STATE_SCALE, Item, append_row, by_id, load_items, read_jsonl
from dsbench.streaming import reassemble_stream

SEED = 0
GREEDY = {"temperature": 0.0, "top_p": 1.0}
THINKING_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
REP_SEED_OFFSET = {"base_rep": 1}  # the A/A pass samples anew; every other state shares seeds


def item_seed(item: Item, state: str) -> int:
    """A thinking pass's seed: fixed per item, the same in every state but the A/A pass."""
    digest = hashlib.sha256(f"{item.bench}:{item.id}".encode()).digest()
    return (int.from_bytes(digest[:4], "big") + REP_SEED_OFFSET.get(state, 0)) % 2**31


def request_body(item: Item, state: str, model: str) -> dict:
    """The exact JSON sent for one item. Only `messages` and the per-item limits cross over."""
    thinking = bool(item.gen.get("thinking"))
    body = {
        "model": model,
        "messages": item.messages,
        **(THINKING_SAMPLING if thinking else GREEDY),
        "seed": item_seed(item, state) if thinking else SEED,
        "max_tokens": item.gen.get("max_tokens", 2048),
        "chat_template_kwargs": {"enable_thinking": thinking},
        "lora": [{"id": 0, "scale": STATE_SCALE[state]}],
        "cache_prompt": True,
    }
    if item.gen.get("stop"):
        body["stop"] = item.gen["stop"]
    if item.gen.get("tools"):
        body["tools"] = item.gen["tools"]
        body["parallel_tool_calls"] = True
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def call(client: httpx.Client, body: dict) -> dict:
    if body.get("stream"):
        with client.stream("POST", "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            return reassemble_stream(r)
    r = client.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    return {"message": choice.get("message") or {}, "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage") or {}, "timings": data.get("timings") or {}}


def run_item(client: httpx.Client, item: Item, state: str, model: str, attempts: int) -> dict:
    body = request_body(item, state, model)
    error = ""
    for attempt in range(attempts):
        t0 = time.monotonic()
        try:
            out = call(client, body)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            error = f"{type(e).__name__}: {e}"[:300]
            time.sleep(2 * (attempt + 1))
            continue
        msg = out["message"]
        row = {
            "bench": item.bench, "id": item.id, "state": state,
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "reasoning_chars": len(msg.get("reasoning_content") or ""),
            "finish_reason": out.get("finish_reason"),
            "prompt_tokens": (out.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (out.get("usage") or {}).get("completion_tokens"),
            "elapsed_s": round(time.monotonic() - t0, 2),
            "error": "",
        }
        if item.gen.get("thinking"):
            row.update(thinking_fields(msg))
        return row
    return {"bench": item.bench, "id": item.id, "state": state, "content": "", "tool_calls": [],
            "finish_reason": None, "error": error}


def thinking_fields(msg: dict) -> dict:
    """The reasoning, and whether it ever closed.

    The prompt opens the reasoning (`<think>\\n`), and the server moves text into `content` only
    after `</think>`. So a reply with reasoning and nothing after it never closed it: its budget
    ran out, or it stopped inside. That is thinking on's form of Gate 2's think-leak.
    """
    reasoning = msg.get("reasoning_content") or ""
    answered = (msg.get("content") or "").strip() or msg.get("tool_calls")
    return {"reasoning": reasoning, "unclosed": bool(reasoning.strip()) and not answered}


def erase_slots(base_url: str) -> int:
    """Empty every slot's prompt cache before a pass.

    llama-server reuses a slot's cached prefix for the next request. At a pass boundary that cache
    was computed under the OTHER state, so reusing it would leak the previous state's KV into this
    one. `parity.py` checks whether the server already guards against this; erasing makes the
    battery independent of that answer.
    """
    with httpx.Client(base_url=base_url, timeout=60) as client:
        slots = client.get("/slots").json()
        for slot in slots:
            client.post(f"/slots/{slot['id']}", params={"action": "erase"}).raise_for_status()
    return len(slots)


def try_erase_slots(base_url: str) -> str:
    """Erase if the server allows it (llama-server needs --slot-save-path for slot actions)."""
    try:
        return f"erased {erase_slots(base_url)} slot caches"
    except httpx.HTTPError as e:
        return f"WARNING slot erase unavailable ({e}); relying on the parity check"


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["id"] for row in read_jsonl(path) if not row.get("error")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", required=True, type=Path)
    ap.add_argument("--state", required=True, choices=sorted(STATE_SCALE))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:8093")
    ap.add_argument("--model", default="qwen36-battery")
    ap.add_argument("--workers", type=int, default=8, help="match the server's --parallel slots")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke tests)")
    ap.add_argument("--no-erase", action="store_true", help="keep the slots' prompt caches")
    args = ap.parse_args()

    items = load_items(args.items)[: args.limit]
    skip = done_ids(args.out)
    todo = [it for it in items if it.id not in skip]
    print(f"[gen] {args.items.name} state={args.state}: {len(items)} items, {len(skip)} done, "
          f"{len(todo)} to run", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if todo and not args.no_erase:
        print(f"[gen] {try_erase_slots(args.base_url)}", flush=True)
    lock = threading.Lock()
    t0, n_done, tokens = time.monotonic(), 0, 0
    limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
    fd = os.open(args.out, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    with httpx.Client(base_url=args.base_url, timeout=args.timeout, limits=limits) as client, \
            ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_item, client, it, args.state, args.model, args.attempts)
                   for it in todo]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                append_row(fd, row)
                n_done += 1
                tokens += row.get("completion_tokens") or 0
                if n_done % 25 == 0 or n_done == len(todo):
                    dt = time.monotonic() - t0
                    print(f"[gen] {n_done}/{len(todo)}  {tokens / dt:.0f} tok/s  "
                          f"{dt / 60:.1f} min  last={row['id']} {row['finish_reason']} "
                          f"{row['error'][:60]}", flush=True)
    os.close(fd)
    latest = by_id(read_jsonl(args.out))
    failed = [i for i, r in latest.items() if r.get("error")]
    print(f"[gen] finished; items still failing after retries: {len(failed)}", flush=True)


if __name__ == "__main__":
    main()
