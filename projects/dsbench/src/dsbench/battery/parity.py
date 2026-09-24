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


def run(base_url: str, scale: float | None) -> list[str]:
    out = []
    with httpx.Client(base_url=base_url, timeout=600) as client:
        for prompt in PROMPTS:
            body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
                    "top_p": 1.0, "seed": 0, "max_tokens": 160,
                    "chat_template_kwargs": {"enable_thinking": False}}
            if scale is not None:
                body["lora"] = [{"id": 0, "scale": scale}]
            r = client.post("/v1/chat/completions", json=body)
            r.raise_for_status()
            out.append(r.json()["choices"][0]["message"].get("content") or "")
    return out


def compare(path: Path) -> dict:
    data = json.loads(path.read_text())
    nolora, steps = data["nolora"]["none"], data["lora"]
    s0, s1, s0b = steps["0:0.0"], steps["1:1.0"], steps["2:0.0"]
    return {
        "scale0_equals_nolora": sum(a == b for a, b in zip(nolora, s0, strict=True)),
        "scale0_repeat_equals_scale0": sum(a == b for a, b in zip(s0, s0b, strict=True)),
        "scale1_differs_from_scale0": sum(a != b for a, b in zip(s0, s1, strict=True)),
        "n": len(PROMPTS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8093")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", choices=["nolora", "lora"])
    ap.add_argument("--steps", default="0,1,0", help="LoRA scales to run in order (tag lora)")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        result = compare(args.out)
        print(json.dumps(result))
        ok = (result["scale0_equals_nolora"] == result["n"]
              and result["scale0_repeat_equals_scale0"] == result["n"]
              and result["scale1_differs_from_scale0"] > 0)
        raise SystemExit(0 if ok else 1)
    data = json.loads(args.out.read_text()) if args.out.exists() else {}
    if args.tag == "nolora":
        data["nolora"] = {"none": run(args.base_url, None)}
    else:
        data["lora"] = {f"{i}:{float(s)}": run(args.base_url, float(s))
                        for i, s in enumerate(args.steps.split(","))}
    args.out.write_text(json.dumps(data, indent=1))
    print(f"[parity] {args.tag}: {len(PROMPTS)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
