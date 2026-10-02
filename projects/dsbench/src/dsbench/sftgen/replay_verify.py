"""Checks for Revision 2's replay and code replies (ADR-004 Revision 2, action item 6).

The base answers `select_prompts.py`'s items with thinking on (`reasoning_pilot.py generate
--block 8192`), one reply a prompt, and a reply becomes a row only if it passes here. Every pool
asks the same of a reply first:
- it finished (`stop`): a reply cut by its budget either loops or wouldn't fit the block;
- it has reasoning and an answer;
- its answer holds no think tag: a second thinking block in the answer would train one.

Then, by pool:
- **gsm8k:** the answer's final number is the gold one. A wrong solution would train at full
  weight, and the check is free (decision 5 proposes it). `gold_match` is written either way, so
  keeping every finished reply instead is a filter at assembly, not a rerun.
- **opencoder_edu:** the reply's code passes every one of its tests, in the battery's sandbox
  (podman: no network, a read-only root, no capabilities; `battery/score.py`'s `run_sandbox`). The
  program is the fenced block that defines the tested function, with its demo code dropped as for
  HumanEval+ (`battery/extract.py`), then the tests. The prompt showed the first test
  (`select_prompts.EXAMPLE_TEST`); the others are unseen.
- **sciriff:** a prompt that says "Only output the JSON object" (its NER tasks) needs an answer
  that is that JSON and nothing else. A reply that wraps it in a fence or a sentence would teach
  the model to break an explicit format instruction, which IFEval measures.
- **oasst1, aya, swe_swiss:** finishing is the check, as ADR-004's table says for replay; SWE-Swiss
  has no tests.

An item retried after an error has several rows; its last row without an error is checked
(`reasoning_pilot.latest_rows`). Each output row is the generation row plus `check`: `ok`, `why`
(empty when ok) and `finished` (the checks every pool shares), and `gold_match` or `tests` where
the pool has them.

Runs on the box, where the sandbox image is:

    cd SRC && PYTHONPATH=. ~/benchlab/batteryvenv/bin/python -m dsbench.sftgen.replay_verify \\
        --items RUN/items/code.jsonl --gen RUN/gen/code.jsonl \\
        --out RUN/verified/code.jsonl --run-dir RUN

`--gold POOL_FILE` checks the dataset's own answers instead of the base's, as the battery's
`score.py --gold` does: an item whose own answer fails its tests can't check a reply, and is left
out of generation. It runs on the Mac too, in docker (`--engine docker --image IMAGE`), since its
code is the dataset's, not the base's.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from dsbench.battery import extract
from dsbench.sftgen.reasoning_pilot import latest_rows

TEST_TIMEOUT_S = 30  # per program; a test that runs longer is a loop, not a slow right answer
JSON_ONLY = "Only output the JSON object"
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(text: str) -> str | None:
    """The answer's final number: the last \\boxed{} one if the answer boxes it, else the last
    number written. Commas are dropped ("1,250" is 1250), a sign is kept."""
    boxed = _BOXED.findall(text)
    for source in ([boxed[-1]] if boxed else []) + [text]:
        numbers = _NUMBER.findall(source)
        if numbers:
            return numbers[-1].replace(",", "")
    return None


def same_number(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def finished(rec: dict) -> str:
    """Why the reply fails what every pool asks of it, or ''."""
    if rec.get("error"):
        return f"error: {rec['error'][:80]}"
    if rec.get("finish_reason") != "stop":
        return f"unfinished: {rec.get('finish_reason')}"
    if not (rec.get("reasoning") or "").strip():
        return "no reasoning"
    answer = rec.get("answer") or ""
    if not answer.strip():
        return "no answer"
    if "<think>" in answer or "</think>" in answer:
        return "think tag in the answer"
    return ""


def json_only(answer: str) -> bool:
    try:
        json.loads(answer.strip())
    except json.JSONDecodeError:
        return False
    return True


def program(item: dict, answer: str) -> str:
    """The reply's code, then the item's tests: exit status 0 means every test passed."""
    check = item["verify"]
    code = extract.humaneval_program(answer, "", check.get("entry_point") or "")
    return code + "\n\n" + "\n".join(check["tests"]) + "\n"


Sandbox = Callable[[list[dict]], dict[str, dict]]  # jobs -> {key: result}


def verify(items: list[dict], gens: dict[str, dict], sandbox: Sandbox) -> tuple[list[dict], dict]:
    """(checked rows in item order, the report). Items with no generation are counted, not
    written: a run cut short checks what it answered."""
    checked: dict[str, dict] = {}
    jobs: list[dict] = []
    for item in items:
        rec = gens.get(item["id"])
        if rec is None:
            continue
        why = finished(rec)
        check: dict = {"ok": not why, "why": why, "finished": not why}
        answer = rec.get("answer") or ""
        if item["pool"] == "gsm8k":
            check["gold_match"] = not why and same_number(final_number(answer),
                                                          item["verify"]["gold_answer"])
            if not why and not check["gold_match"]:
                check.update(ok=False, why=f"final number {final_number(answer)}, gold "
                                          f"{item['verify']['gold_answer']}")
        elif item["pool"] == "sciriff" and JSON_ONLY in item["messages"][-1]["content"]:
            if not why and not json_only(answer):
                check.update(ok=False, why="not the JSON alone")
        elif "tests" in item.get("verify", {}) and not why:
            jobs.append({"key": item["id"], "kind": "program", "program": program(item, answer),
                         "timeout": TEST_TIMEOUT_S})
        checked[item["id"]] = {**rec, "check": check}

    for key, result in (sandbox(jobs) if jobs else {}).items():
        check = checked[key]["check"]
        check["tests"] = {"passed": result["passed"], "status": result["status"],
                          "detail": result.get("detail", "")[:200]}
        if not result["passed"]:
            check.update(ok=False, why=f"tests: {result['status']}")

    rows = [checked[item["id"]] for item in items if item["id"] in checked]
    return rows, report(items, rows)


def report(items: list[dict], rows: list[dict]) -> dict:
    pool_of = {item["id"]: item["pool"] for item in items}
    total = Counter(item["pool"] for item in items)
    answered = Counter(pool_of[r["id"]] for r in rows)
    kept = Counter(pool_of[r["id"]] for r in rows if r["check"]["ok"])
    finished_n = Counter(pool_of[r["id"]] for r in rows if r["check"]["finished"])
    why: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if not r["check"]["ok"]:
            # the kind of failure, not its detail: "tests: error", "unfinished: length"
            why[pool_of[r["id"]]][re.sub(r"(final number|error:) .*", r"\1 ...",
                                         r["check"]["why"])] += 1
    return {"pools": {pool: {"items": total[pool], "answered": answered[pool],
                             "finished": finished_n[pool], "kept": kept[pool],
                             "failed": dict(why[pool].most_common())}
                      for pool in sorted(total)},
            "kept": sum(kept.values()), "answered": len(rows), "items": len(items)}


def references(items: list[dict], pool_file: Path) -> dict[str, dict]:
    """The dataset's own answer to each item with tests, as a generation row: the first source
    row that makes the item, as `select_prompts.select` keeps the first of duplicates."""
    from dsbench.sftgen.select_prompts import POOLS, make_item, prompt_messages, with_example_test

    wanted = {item["id"] for item in items if "tests" in item.get("verify", {})}
    by_name = {pool.name: pool for pool in POOLS}
    refs: dict[str, dict] = {}
    for line in pool_file.open():
        row = json.loads(line)
        if not row["meta"].get("testcase"):
            continue
        prompt = with_example_test(prompt_messages(row["messages"]), row["meta"]["testcase"])
        for name in {item["pool"] for item in items if item["id"] in wanted}:
            item_id = make_item(by_name[name], row, prompt)["id"]
            if item_id in wanted and item_id not in refs:
                refs[item_id] = {"id": item_id, "answer": row["messages"][-1]["content"],
                                 "reasoning": "(the dataset's answer)", "finish_reason": "stop",
                                 "error": ""}
    return refs


def container_sandbox(run_dir: Path, workers: int, engine: str = "podman",
                      image: str | None = None) -> Sandbox:
    """The battery's sandbox (`battery/score.py`), with an empty data mount: these tests read
    nothing."""
    from dsbench.battery.score import IMAGE, run_sandbox

    data_dir = run_dir / "sandbox-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    harness_src = Path(__file__).resolve().parents[2]  # the dir that holds dsbench/

    def run(jobs: list[dict]) -> dict[str, dict]:
        return run_sandbox(jobs, run_dir=run_dir, data_dir=data_dir.resolve(),
                           harness_src=harness_src, workers=workers, image=image or IMAGE,
                           engine=engine)

    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", type=Path, required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--gen", type=Path, help="the base's replies")
    source.add_argument("--gold", type=Path, metavar="POOL_FILE",
                        help="check the dataset's own answers to the items with tests")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="holds the sandbox's job files")
    ap.add_argument("--report", type=Path, help="default: OUT with .report.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--engine", default="podman", choices=("podman", "docker"))
    ap.add_argument("--image", help="default: the battery's sandbox image")
    args = ap.parse_args()

    items = [json.loads(line) for line in args.items.open()]
    if args.gold:
        gens = references(items, args.gold)
        items = [item for item in items if item["id"] in gens]
    else:
        gens = latest_rows(args.gen)
    sandbox = container_sandbox(args.run_dir, args.workers, args.engine, args.image)
    rows, result = verify(items, gens, sandbox)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    out = args.report or args.out.with_suffix(".report.json")
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
