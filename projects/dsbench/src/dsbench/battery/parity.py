"""Check the premise of the battery: "scale 0" really is the base model, on a shared server.

Every base-vs-adapter comparison in the battery assumes two things:
  1. a request with LoRA scale 0 computes exactly what a server without the adapter computes; and
  2. switching between scales never reuses KV computed at the other scale (llama-server reuses a
     cached prompt prefix).
The check runs a few fixed prompts one at a time, greedy:
    --tag nolora             on a server started WITHOUT --lora
    --tag lora --steps 0,1,0 on the battery server: scale 0, then 1, then 0 again
`--compare` then requires nolora == scale 0 == scale 0 again, scale 1 != scale 0 somewhere, and
no cached token at a switch.

**Each request runs on an emptied server.** It erases every slot first. The replies are compared
character for character, and a greedy reply repeats only when the request is computed alike: on
a unified KV pool, its attention also spans the cells that other slots still hold. Until
2026-10-09 the server cleared idle slots itself before each request, so this held without the
erase. With `--cache-ram 0` it stopped doing so (the fork ties the two). On the first launch of
Revision 2 against 2.1, scale 0's replies then differed from the bare base's on 1 and 2 of the 8
prompts, and the two scale-0 steps differed on 2 and 1. Revision 2.1's 24 requests took no cached
token at all, so the pool alone made the difference.

**A switch is tested directly.** Before each request of steps 1 on, the same prompt goes first at
the previous step's scale. The slot that holds it is the best match for the request, and the
server must drop it rather than reuse it: the request's cached prompt tokens (the server's
`usage.prompt_tokens_details`) must be 0. Until 2026-10-09 this went untested. The server also
kept a RAM copy of prompts and loaded the closest one into a slot, by tokens alone, never asking
which scale computed it. In each of the seven windows that ran the old check (2026-10-06 to
10-08), 3 of the 8 prompts began their scale-1 reply from their own scale-0 KV, with all but their
last 4 tokens. battery_server.sh now turns that copy off (`--cache-ram 0`). Qwen3.6 reuses a prefix
only up to a saved checkpoint of its recurrent state (it is a hybrid model), and a short prompt
has none before its last 4 tokens. So 0 cached tokens is what a correct switch gives, even with a
shared template.

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


def _erase(client: httpx.Client) -> None:
    """Empty every slot (needs the server's --slot-save-path, which battery_server.sh sets)."""
    for slot in client.get("/slots").json():
        client.post(f"/slots/{slot['id']}", params={"action": "erase"}).raise_for_status()


def _ask(client: httpx.Client, prompt: str, scale: float | None,
         lora_id: int) -> tuple[str, int | None]:
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "top_p": 1.0, "seed": 0, "max_tokens": 160,
            "chat_template_kwargs": {"enable_thinking": False}}
    if scale is not None:
        body["lora"] = [{"id": lora_id, "scale": scale}]  # unlisted ids get 0
    r = client.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    data = r.json()
    details = (data.get("usage") or {}).get("prompt_tokens_details") or {}
    return data["choices"][0]["message"].get("content") or "", details.get("cached_tokens")


def run(base_url: str, scale: float | None, lora_id: int = 0,
        before: float | None = None) -> tuple[list[str], list[int | None]]:
    """Each prompt on an emptied server, at `scale`. With `before`, the same prompt goes first at
    that scale, so the request finds its own prompt cached at another scale. Returns the replies
    and how many prompt tokens each took from cache (None if the server doesn't say)."""
    replies, cached = [], []
    with httpx.Client(base_url=base_url, timeout=600) as client:
        for prompt in PROMPTS:
            _erase(client)
            if before is not None:
                _ask(client, prompt, before, lora_id)
            reply, n = _ask(client, prompt, scale, lora_id)
            replies.append(reply)
            cached.append(n)
    return replies, cached


TAG_IDS = {"lora": 0, "lora2": 1}


def _steps(nolora: list[str], steps: dict[str, list[str]],
           cached: dict[str, list[int | None]] | None) -> dict:
    s0, s1, s0b = steps["0:0.0"], steps["1:1.0"], steps["2:0.0"]
    # the switched requests: step 1 after its prompt at scale 0, step 2 after it at scale 1
    switched = [n for k in ("1:1.0", "2:0.0") for n in (cached or {}).get(k, [None])]
    return {
        "scale0_equals_nolora": sum(a == b for a, b in zip(nolora, s0, strict=True)),
        "scale0_repeat_equals_scale0": sum(a == b for a, b in zip(s0, s0b, strict=True)),
        "scale1_differs_from_scale0": sum(a != b for a, b in zip(s0, s1, strict=True)),
        # None: not recorded (a parity file from before 2026-10-09, or a server that doesn't say)
        "switch_tokens_from_cache": sum(switched) if None not in switched else None,
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
    """Every check of every adapter. A switch whose cached tokens weren't recorded fails: the
    server it ran on can't be shown to keep the scales apart."""
    def ok(r: dict) -> bool:
        return (r["scale0_equals_nolora"] == r["n"] and r["scale0_repeat_equals_scale0"] == r["n"]
                and r["scale1_differs_from_scale0"] > 0 and r["switch_tokens_from_cache"] == 0)
    return ok(result) and ("lora2" not in result or ok(result["lora2"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8093")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", choices=["nolora", *TAG_IDS],
                    help="nolora: a server without adapters; lora: adapter 0; lora2: adapter 1")
    ap.add_argument("--steps", default="0,1,0",
                    help="LoRA scales to run in order (tag lora); each step after the first is "
                         "preceded, prompt by prompt, by the previous step's scale")
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
        scales = [float(s) for s in args.steps.split(",")]
        steps = {f"{i}:{s}": run(args.base_url, s, TAG_IDS[args.tag], scales[i - 1] if i else None)
                 for i, s in enumerate(scales)}
        data[args.tag] = {k: replies for k, (replies, _) in steps.items()}
        data[f"{args.tag}_cached"] = {k: n for k, (_, n) in steps.items()}
    args.out.write_text(json.dumps(data, indent=1))
    print(f"[parity] {args.tag}: {len(PROMPTS)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
