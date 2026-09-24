"""Same server, same held-out prompts: adapter on (scale 1) vs off (scale 0).

Greedy decoding with thinking disabled, so the two answers differ only by the adapter. Runs on the
box against `serve_lora_test.sh` (loopback port 8093):

    python3 lora_onoff.py ~/gate2/eval_buckets_gate2.jsonl
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8093/v1/chat/completions"


def ask(messages: list[dict], scale: float) -> tuple[str, dict]:
    body = {"messages": messages, "max_tokens": 256, "temperature": 0, "seed": 0,
            "lora": [{"id": 0, "scale": scale}],
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(request, timeout=600))
    return reply["choices"][0]["message"]["content"], reply.get("timings", {})


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1])]
    for record in [r for r in rows if r["bucket"] == "targetA_heldout"][:3]:
        prompt = [m for m in record["messages"] if m["role"] != "assistant"]
        reference = next(m["content"] for m in record["messages"] if m["role"] == "assistant")
        print("=" * 100)
        print("Q:", prompt[-1]["content"][:300].replace("\n", " "))
        print("REFERENCE:", reference.strip()[:300].replace("\n", " "))
        for scale in (0.0, 1.0):
            answer, timings = ask(prompt, scale)
            print(f"--- adapter {'ON ' if scale else 'OFF'} | "
                  f"{timings.get('predicted_per_second', 0):6.1f} tok/s gen, "
                  f"{timings.get('prompt_per_second', 0):7.1f} tok/s prompt, "
                  f"{timings.get('predicted_n', 0)} tokens")
            print(answer.strip()[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
