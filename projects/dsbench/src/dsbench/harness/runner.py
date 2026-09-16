"""Run the benchmark against a served model and write a scored run file.

  uv run --package dsbench dsbench-run --base-url http://dashi:8080 --model qwen3.6 --label base

before / during / after are just three runs with different --label; compare any two with
`dsbench-compare`. Runs are JSON so they diff cleanly and a later checker fix can be re-scored
offline (the raw responses are kept).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime as dt
import json
import time
from pathlib import Path

from dsbench.harness.client import ChatClient
from dsbench.harness.extract import extract_code
from dsbench.harness.sandbox import run_candidate
from dsbench.harness.score import render_markdown
from dsbench.loader import load_problems
from dsbench.schema import CATEGORIES, DIFFICULTIES, Problem, ProblemResult

_RUNS = Path(__file__).resolve().parents[3] / "reports" / "runs"


def _attempt(client: ChatClient, problem: Problem) -> ProblemResult:
    t0 = time.time()
    try:
        comp = client.complete(problem.prompt)
    except Exception as e:  # noqa: BLE001 - a transport/HTTP failure is a (recorded) miss, not a crash
        return ProblemResult(problem.id, problem.category, problem.difficulty, False,
                             "http_error", reason=str(e)[:200], latency_s=time.time() - t0)
    raw = comp.content
    code = extract_code(raw, problem.mode)
    if not code and comp.finish_reason == "length":
        # The model hit the token cap without producing any code. For a thinking model this is
        # runaway reasoning -- the whole budget went to the reasoning channel and no answer was
        # emitted. That is a serving/config artifact, NOT a capability miss, so it gets its own
        # status: it must never silently count as a wrong answer, and `dsbench-compare` can drop
        # these pairs so a fine-tune that merely changes reasoning length does not fake a flip.
        return ProblemResult(problem.id, problem.category, problem.difficulty, False, "truncated",
                             reason=(f"hit token cap with no answer (finish=length, "
                                     f"{comp.reasoning_chars} reasoning chars); raise --max-tokens "
                                     f"or lower the server's reasoning effort"),
                             raw_response=raw, latency_s=time.time() - t0)
    passed, status, reason = run_candidate(problem, code)
    return ProblemResult(problem.id, problem.category, problem.difficulty, passed, status,
                         reason=reason, extracted_code=code, raw_response=raw,
                         latency_s=time.time() - t0)


def _run_problem(client: ChatClient, problem: Problem, k: int) -> ProblemResult:
    """pass@k: try up to k samples, return the first pass (else the last attempt)."""
    last = None
    for _ in range(k):
        last = _attempt(client, problem)
        if last.passed:
            return last
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description="Run dsbench against an OpenAI-compatible server.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="served", help="served model id")
    ap.add_argument("--label", default=None, help="run label, e.g. base / step500 / final")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--system", default=None, help="optional system prompt")
    # Choices follow the schema so a new category/tier (e.g. `expert`) needs no edit here.
    ap.add_argument("--category", choices=list(CATEGORIES), default=None)
    ap.add_argument("--difficulty", choices=list(DIFFICULTIES), default=None)
    ap.add_argument("--ids", default=None, help="comma-separated problem ids to run")
    ap.add_argument("--limit", type=int, default=None, help="first N problems (smoke test)")
    ap.add_argument("--k", type=int, default=1, help="pass@k samples (k>1 wants temperature>0)")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel problems")
    ap.add_argument("--no-think", action="store_true",
                    help="turn OFF a reasoning model's thinking via chat_template_kwargs "
                         "(enable_thinking=false). Recommended for thinking models: at temp 0 they "
                         "can spend the whole budget reasoning and never answer (-> truncated).")
    ap.add_argument("--out-dir", default=str(_RUNS))
    args = ap.parse_args()

    ids = set(args.ids.split(",")) if args.ids else None
    problems = load_problems(args.category, args.difficulty, ids)
    if args.limit:
        problems = problems[: args.limit]
    if not problems:
        raise SystemExit("no problems matched the filters")

    label = args.label or dt.datetime.now().strftime("run-%Y%m%d-%H%M%S")
    chat_template_kwargs = {"enable_thinking": False} if args.no_think else None
    client_kw = dict(base_url=args.base_url, model=args.model, temperature=args.temperature,
                     top_p=args.top_p, max_tokens=args.max_tokens, system=args.system,
                     chat_template_kwargs=chat_template_kwargs)

    think = "off" if args.no_think else "on"
    print(f"running {len(problems)} problems, pass@{args.k}, concurrency {args.concurrency}, "
          f"thinking {think} ...")
    results: list[ProblemResult] = []
    # One client per worker thread: httpx.Client is not meant to be shared across threads.
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(_run_problem, ChatClient(**client_kw), p, args.k): p for p in problems}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            mark = "PASS" if r.passed else f"FAIL[{r.status}]"
            print(f"  {mark:12} {r.id}")

    results.sort(key=lambda r: r.id)
    meta = {
        "label": label, "model": args.model, "base_url": args.base_url,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "temperature": args.temperature, "k": args.k, "n": len(results),
        # Recorded so before/after runs can be checked for a matching protocol -- comparing a
        # thinking-off baseline against a thinking-on checkpoint would not be apples-to-apples.
        "max_tokens": args.max_tokens, "no_think": args.no_think,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{label}"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(
        {"meta": meta, "results": [dataclasses.asdict(r) for r in results]}, indent=2))
    md = render_markdown(meta, results)
    (out_dir / f"{stem}.md").write_text(md)

    passed = sum(r.passed for r in results)
    print("\n" + md)
    print(f"\nwrote {json_path}")
    print(f"score: {passed}/{len(results)}")


if __name__ == "__main__":
    main()
