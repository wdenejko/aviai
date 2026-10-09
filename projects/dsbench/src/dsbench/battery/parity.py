"""Check the premise of the battery: "scale 0" really is the base model, on a shared server.

Every base-vs-adapter comparison in the battery assumes two things:
  1. a request with LoRA scale 0 computes exactly what a server without the adapter computes; and
  2. switching a slot between scales does not reuse KV-cache entries computed under the other
     scale (llama-server reuses a slot's cached prompt prefix).
The check runs a few fixed prompts one at a time, greedy:
    --tag nolora            on a server started WITHOUT --lora
    --tag lora --steps 0,1,0 on the battery server: scale 0, then 1, then 0 again
Requests run one at a time, so a repeated prompt lands on the slot that just cached it under the
other scale; a stale cache would show up as a difference between the two scale-0 steps.
`--compare` then requires nolora == scale-0 == scale-0-again, and scale 1 != scale 0 somewhere.

That last check is blind to one path (found 2026-10-09 in the probe windows' server logs). The
server also keeps prompts in RAM (`--cache-ram`, 8 GiB by default) and loads the closest one into
a slot, by tokens alone. A request names its adapter scale, and the server drops a slot's cache
when that scale differs from the slot's previous request. It never asks which scale computed the
prompt it just loaded from RAM. So in the scale-1 step, a prompt could start from its own scale-0
KV, loaded from RAM onto a slot whose previous request was already at scale 1. In each of the
seven windows that ran this check before (2026-10-06 to 10-08), 3 of the 8 prompts did so, with
all but their last 4 tokens, and the two scale-0 steps stayed equal.
battery_server.sh now turns the RAM cache off (`--cache-ram 0`) on a server with an adapter. The
check records each reply's cached prompt tokens (the server's `usage.prompt_tokens_details`) and
requires the scale-1 step to take none. Each of its prompts runs at scale 1 for the first time, so
anything cached would be scale 0's. Qwen3.6 can't reuse even a shared template prefix: it is a
hybrid model, so a prefix is reused only up to a saved checkpoint of its recurrent state, and a
short prompt has none before its last few tokens.

A server with two adapters (battery_server.sh with LORA2) is checked for each: `--tag lora2` runs
the same steps on the second adapter (id 1), and `--compare` checks both. A request names one
adapter, and the server puts every adapter it doesn't name at 0. An adapter at 0 is left out of
the compute graph (llama-context.cpp, set_adapters_lora), so each adapter is computed as on a
server that loaded it alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Write a SQL query that returns the top 3 customers by total order amount from "
    "orders(customer_id, amount).",
    "Explain in two sentences what a LEFT JOIN does.",
    "What is 17 * 23? Show the calculation.",
    "Given a pandas DataFrame df with columns city and temp, compute the mean temp per city.",
    "List three differences between TCP and UDP.",
    "Write a haiku about a data pipeline.",
    "Convert the timestamp 2024-03-10 02:30 America/New_York to UTC and explain the DST issue.",
]


def run(base_url: str, scale: float | None, lora_id: int = 0) -> tuple[list[str], list[int | None]]:
    """The replies, and how many of each prompt's tokens the server took from cache (None if it
    doesn't say)."""
    replies, cached = [], []
    with httpx.Client(base_url=base_url, timeout=600) as client:
        for prompt in PROMPTS:
            body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
                    "top_p": 1.0, "seed": 0, "max_tokens": 160,
                    "chat_template_kwargs": {"enable_thinking": False}}
            if scale is not None:
                body["lora"] = [{"id": lora_id, "scale": scale}]  # unlisted ids get 0
            r = client.post("/v1/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
            replies.append(data["choices"][0]["message"].get("content") or "")
            details = (data.get("usage") or {}).get("prompt_tokens_details") or {}
            cached.append(details.get("cached_tokens"))
    return replies, cached


TAG_IDS = {"lora": 0, "lora2": 1}


def _steps(nolora: list[str], steps: dict[str, list[str]],
           cached: dict[str, list[int | None]] | None) -> dict:
    s0, s1, s0b = steps["0:0.0"], steps["1:1.0"], steps["2:0.0"]
    s1_cached = (cached or {}).get("1:1.0")
    return {
        "scale0_equals_nolora": sum(a == b for a, b in zip(nolora, s0, strict=True)),
        "scale0_repeat_equals_scale0": sum(a == b for a, b in zip(s0, s0b, strict=True)),
        "scale1_differs_from_scale0": sum(a != b for a, b in zip(s0, s1, strict=True)),
        # None: not recorded (a parity file from before 2026-10-09, or a server that doesn't say)
        "scale1_tokens_from_cache": (sum(s1_cached) if s1_cached and None not in s1_cached
                                     else None),
        "n": len(PROMPTS),
    }


def compare(path: Path) -> dict:
    """The first adapter's checks at the top level, as for a one-adapter server, and the second's,
    if run, under `lora2`."""
    data = json.loads(path.read_text())
    nolora = data["nolora"]["none"]
    out = _steps(nolora, data["lora"], data.get("lora_cached"))
    if "lora2" in data:
        out["lora2"] = _steps(nolora, data["lora2"], data.get("lora2_cached"))
    return out


def passed(result: dict) -> bool:
    """Every check of every adapter. A scale-1 step whose cached tokens weren't recorded fails:
    the server it ran on can't be shown to keep the scales apart."""
    def ok(r: dict) -> bool:
        return (r["scale0_equals_nolora"] == r["n"] and r["scale0_repeat_equals_scale0"] == r["n"]
                and r["scale1_differs_from_scale0"] > 0 and r["scale1_tokens_from_cache"] == 0)
    return ok(result) and ("lora2" not in result or ok(result["lora2"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8093")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", choices=["nolora", *TAG_IDS],
                    help="nolora: a server without adapters; lora: adapter 0; lora2: adapter 1")
    ap.add_argument("--steps", default="0,1,0", help="LoRA scales to run in order (tag lora)")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        result = compare(args.out)
        print(json.dumps(result))
        raise SystemExit(0 if passed(result) else 1)
    data = json.loads(args.out.read_text()) if args.out.exists() else {}
    if args.tag == "nolora":
        data["nolora"] = {"none": run(args.base_url, None)[0]}
    else:
        steps = {f"{i}:{float(s)}": run(args.base_url, float(s), TAG_IDS[args.tag])
                 for i, s in enumerate(args.steps.split(","))}
        data[args.tag] = {k: replies for k, (replies, _) in steps.items()}
        data[f"{args.tag}_cached"] = {k: cached for k, (_, cached) in steps.items()}
    args.out.write_text(json.dumps(data, indent=1))
    print(f"[parity] {args.tag}: {len(PROMPTS)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
