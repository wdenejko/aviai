"""MTP draft acceptance with and without the adapter (ADR-001 Gate 2: drop <= 5 points).

Qwen3.6 ships one multi-token-prediction (MTP) layer that drafts the next tokens from the trunk's
last hidden state; llama-server verifies the drafts (`--spec-type draft-mtp`). The adapter changes
the trunk but not the MTP layer, which stays frozen at its original weights. So the question is
how often the unchanged drafter still guesses what the adapted trunk would say.

Same server, same prompts, greedy, run once at LoRA scale 0 and once at scale 1. Per request,
llama-server reports `timings.draft_n` (tokens proposed) and `timings.draft_n_accepted` (tokens
the target confirmed). Acceptance = accepted / proposed, pooled over all prompts. The prompts are a
fixed slice of the battery's own items, so the text resembles what the battery measures.

    python -m dsbench.battery.mtp --items-dir ITEMS --out MTP.json [--per-bench 50]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from dsbench.battery.items import load_items

BENCHES = ("ifeval", "humaneval_plus", "ds1000", "mmlu_pro")


def prompts(items_dir: Path, per_bench: int) -> list[list[dict]]:
    """The first `per_bench` items of each bench: deterministic, and mixed prose/code."""
    out = []
    for bench in BENCHES:
        path = items_dir / f"{bench}.jsonl"
        if path.exists():
            out.extend(it.messages for it in load_items(path)[:per_bench])
    return out


def one(client: httpx.Client, messages: list[dict], scale: float, max_tokens: int) -> dict:
    body = {"messages": messages, "temperature": 0.0, "top_p": 1.0, "seed": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "lora": [{"id": 0, "scale": scale}]}
    r = client.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    t = r.json().get("timings") or {}
    return {"draft_n": t.get("draft_n", 0), "accepted": t.get("draft_n_accepted", 0),
            "predicted_n": t.get("predicted_n", 0),
            "predicted_per_second": t.get("predicted_per_second", 0.0)}


def measure(base_url: str, batch: list[list[dict]], scale: float, workers: int,
            max_tokens: int) -> dict:
    with httpx.Client(base_url=base_url, timeout=1800) as client, \
            ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(lambda m: one(client, m, scale, max_tokens), batch))
    drafted = sum(r["draft_n"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    return {"scale": scale, "requests": len(rows), "drafted": drafted, "accepted": accepted,
            "acceptance": 100 * accepted / drafted if drafted else None,
            "tokens": sum(r["predicted_n"] for r in rows),
            "mean_tok_per_s": sum(r["predicted_per_second"] for r in rows) / max(1, len(rows))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8093")
    ap.add_argument("--items-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-bench", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1,
                    help="1 = one request at a time, the production (single-user) regime")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    batch = prompts(args.items_dir, args.per_bench)
    result = {"prompts": len(batch), "states": {}}
    for state, scale in (("base", 0.0), ("adapter", 1.0)):
        result["states"][state] = measure(args.base_url, batch, scale, args.workers,
                                          args.max_tokens)
        print(json.dumps({state: result["states"][state]}), flush=True)
    base, adapter = result["states"]["base"], result["states"]["adapter"]
    if base["acceptance"] is not None and adapter["acceptance"] is not None:
        result["drop_points"] = base["acceptance"] - adapter["acceptance"]
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
