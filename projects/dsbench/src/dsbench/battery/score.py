"""Score generations: one pass/fail per (benchmark, item, state), written next to the generations.

Benchmarks that execute model code (HumanEval+, DS-1000, LiveCodeBench, BIRD) are turned into jobs
and run inside the podman sandbox (`sandbox_exec.py`). The rest are scored on the host, because
scoring them executes nothing the model wrote: IFEval (Google's checkers on the reply text), the
multiple-choice sets (a regex), and BFCL (its AST checker compares decoded calls to answer keys).

`--gold` scores each benchmark's reference solutions instead of generations. That is the harness
check the ADR asks for at Gate 1 (item 5): if a published reference solution fails here, the
failure is in our harness or environment, and that item cannot measure the model.

Run (on the box):
    python -m dsbench.battery.score --run-dir RUN --bench ds1000 [--gold]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

from dsbench.battery import extract
from dsbench.battery.items import Item, by_id, load_items, read_jsonl, write_jsonl

IMAGE = "localhost/benchlab-battery-sandbox:py310-ds1000"
SANDBOXED = ("humaneval_plus", "ds1000", "lcb", "bird")


# --- jobs for the sandbox ----------------------------------------------------------------------


def humaneval_job(item: Item, content: str | None) -> dict:
    ref = item.ref
    solution = (ref["prompt"] + ref["canonical_solution"] if content is None
                else extract.humaneval_program(content, ref["prompt"], ref["entry_point"]))
    program = f"{solution}\n\n{ref['test']}\n\ncheck({ref['entry_point']})\n"
    return {"kind": "program", "program": program, "timeout": 60}


def ds1000_job(item: Item, content: str | None) -> dict:
    """DS-1000's own test program (test_ds1000.py), 120 s as upstream."""
    code = item.ref["reference_code"] if content is None else extract.ds1000_code(content)
    ctx = item.ref["code_context"]
    program = (ctx + "\n" + f"code = {code!r}\n" + "test_execution(code)\n"
               + ("test_string(code)\n" if "test_string(" in ctx else "\n"))
    return {"kind": "program", "program": program, "timeout": 120}


def lcb_job(item: Item, content: str | None) -> dict | None:
    code = extract.lcb_code(content or "")
    if not code.strip():
        return None
    return {"kind": "lcb", "code": code, "tests": item.ref["tests"],
            "n_tests": item.ref["n_tests"], "timeout": 6}


def bird_job(item: Item, content: str | None) -> dict:
    pred = item.ref["gold_sql"] if content is None else extract.bird_sql(content)
    return {"kind": "sqlite", "db": item.ref["db"], "pred": pred, "gold": item.ref["gold_sql"],
            "timeout": 30}


JOB_BUILDERS = {"humaneval_plus": humaneval_job, "ds1000": ds1000_job, "lcb": lcb_job,
                "bird": bird_job}


def run_sandbox(jobs: list[dict], *, run_dir: Path, data_dir: Path, harness_src: Path,
                workers: int, image: str, engine: str = "podman") -> dict[str, dict]:
    """Run jobs in one container and return {key: result}. The container gets no writable mount.
    `engine`: podman on the box; docker takes the same flags, for a check run on the Mac."""
    work = run_dir / "sandbox-jobs"
    work.mkdir(parents=True, exist_ok=True)
    jobs_file = work / f"jobs-{uuid.uuid4().hex[:8]}.jsonl"
    write_jsonl(jobs_file, jobs)
    cmd = [
        engine, "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "all",
        "--security-opt", "no-new-privileges", "--memory", "32g", "--memory-swap", "32g",
        "--pids-limit", "4096", "--cpus", str(workers), "--tmpfs", "/tmp:rw,exec,size=16g",
        # `z`: relabel for SELinux (Fedora enforces it; without it the container can't read).
        "-v", f"{harness_src}:/harness:ro,z", "-v", f"{data_dir}:/data:ro,z",
        "-v", f"{work}:/work:ro,z", image,
        "python", "/harness/dsbench/battery/sandbox_exec.py", f"/work/{jobs_file.name}",
        "--workers", str(workers),
    ]
    results: dict[str, dict] = {}
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            if line.startswith("{"):
                row = json.loads(line)
                results[row["key"]] = row
    jobs_file.unlink()
    # sandbox_exec always exits 0 (a failing job is a result, not a crash). A non-zero exit or a
    # missing result means the SANDBOX broke; writing those items as model failures would be a lie.
    missing = len(jobs) - len(results)
    if proc.returncode != 0 or missing:
        raise RuntimeError(f"sandbox exited {proc.returncode} with {missing} of {len(jobs)} "
                           "results missing; no scores written")
    return results


# --- host-side scorers -------------------------------------------------------------------------


def _refs_on_path(data_dir: Path) -> None:
    refs = str(data_dir / "refs")
    if refs not in sys.path:
        sys.path.insert(0, refs)


def ifeval_scorer(data_dir: Path):
    """Google's instruction_following_eval, with one change: sentence splitting loads Punkt from
    NLTK's `punkt_tab` (plain tables) instead of the legacy `english.pickle`, which NLTK >= 3.9
    refuses to unpickle. Same Punkt parameters, no pickle."""
    _refs_on_path(data_dir)
    from instruction_following_eval import evaluation_lib, instructions_util
    from nltk.tokenize.punkt import PunktTokenizer

    tokenizer = PunktTokenizer("english")
    instructions_util._get_sentence_tokenizer = lambda: tokenizer

    def score(item: Item, gen: dict) -> dict:
        ref = item.ref
        kwargs = [{k: v for k, v in kw.items() if v is not None} for kw in ref["kwargs"]]
        inp = evaluation_lib.InputExample(key=ref["key"], instruction_id_list=
                                          ref["instruction_id_list"], prompt=ref["prompt"],
                                          kwargs=kwargs)
        response = {ref["prompt"]: extract.strip_think(gen["content"])}
        strict = evaluation_lib.test_instruction_following_strict(inp, response)
        loose = evaluation_lib.test_instruction_following_loose(inp, response)
        return {"passed": bool(strict.follow_all_instructions), "status": "ok",
                "detail": "", "extra": {
                    "strict_list": list(strict.follow_instruction_list),
                    "loose_all": bool(loose.follow_all_instructions),
                    "loose_list": list(loose.follow_instruction_list)}}
    return score


def mcq_scorer(letter_fn):
    def score(item: Item, gen: dict) -> dict:
        got = letter_fn(gen["content"])
        ok = got == item.ref["answer"]
        return {"passed": ok, "status": "ok" if ok else ("no_answer" if got is None else "wrong"),
                "detail": f"got {got}, want {item.ref['answer']}"}
    return score


def bfcl_scorer(data_dir: Path):
    _refs_on_path(data_dir)
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    def score(item: Item, gen: dict) -> dict:
        ref = item.ref
        decoded, why = extract.bfcl_calls(gen.get("tool_calls") or [])
        format_ok = decoded is not None
        if ref["category"] == "irrelevance":
            # BFCL: success means no function call could be decoded from the reply.
            ok = not decoded
            return {"passed": ok, "status": "ok" if ok else "wrong",
                    "detail": "" if ok else "called a tool", "extra": {"format_ok": format_ok}}
        if not format_ok:
            return {"passed": False, "status": "format", "detail": why,
                    "extra": {"format_ok": False}}
        if not decoded:
            return {"passed": False, "status": "no_call", "detail": "no tool call",
                    "extra": {"format_ok": True}}
        res = ast_checker(ref["functions"], decoded, ref["ground_truth"], Language.PYTHON,
                          ref["category"], "battery")
        ok = bool(res["valid"])
        return {"passed": ok, "status": "ok" if ok else "wrong",
                "detail": "" if ok else str(res.get("error"))[:300],
                "extra": {"format_ok": True}}
    return score


def host_scorer(bench: str, data_dir: Path):
    if bench == "ifeval":
        return ifeval_scorer(data_dir)
    if bench == "mmlu_pro":
        return mcq_scorer(extract.mmlu_pro_letter)
    if bench == "gpqa":
        return mcq_scorer(extract.gpqa_letter)
    if bench == "bfcl":
        return bfcl_scorer(data_dir)
    raise ValueError(f"no host scorer for {bench}")


# --- driver ------------------------------------------------------------------------------------


def score_state(bench: str, items: list[Item], gens: dict[str, dict] | None, *, state: str,
                run_dir: Path, data_dir: Path, harness_src: Path, workers: int,
                image: str) -> list[dict]:
    """Score one state's generations (or, with gens=None, the gold references)."""
    rows = []
    if bench in SANDBOXED:
        jobs, pending = [], {}
        for item in items:
            gen = None if gens is None else gens.get(item.id)
            if gens is not None and (gen is None or gen.get("error")):
                rows.append({"id": item.id, "passed": False, "status": "no_generation"})
                continue
            job = JOB_BUILDERS[bench](item, None if gen is None else gen["content"])
            if job is None:
                rows.append({"id": item.id, "passed": False, "status": "no_code"})
                continue
            job["key"] = item.id
            jobs.append(job)
            pending[item.id] = item
        results = run_sandbox(jobs, run_dir=run_dir, data_dir=data_dir, harness_src=harness_src,
                              workers=workers, image=image)
        for key in pending:
            res = results.get(key, {"passed": False, "status": "lost", "detail": "no result"})
            rows.append({"id": key, "passed": bool(res["passed"]), "status": res["status"],
                         "detail": res.get("detail", ""), "elapsed_s": res.get("elapsed_s")})
    else:
        score = host_scorer(bench, data_dir)
        for item in items:
            gen = gens.get(item.id) if gens else None
            if gen is None or gen.get("error"):
                rows.append({"id": item.id, "passed": False, "status": "no_generation"})
                continue
            rows.append({"id": item.id, **score(item, gen)})
    for row in rows:
        row.update(bench=bench, state=state)
        gen = (gens or {}).get(row["id"]) or {}
        row["finish_reason"] = gen.get("finish_reason")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=None, help="default: RUN/data")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--states", default="base,adapter,base_rep,adapter_half")
    ap.add_argument("--gold", action="store_true", help="score the reference solutions instead")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--image", default=IMAGE)
    args = ap.parse_args()
    data_dir = (args.data_dir or args.run_dir / "data").resolve()
    harness_src = Path(__file__).resolve().parents[2]  # the dir that holds dsbench/

    for bench in args.bench.split(","):
        items = load_items(args.run_dir / "items" / f"{bench}.jsonl")
        passes = ["gold"] if args.gold else [
            s for s in args.states.split(",")
            if (args.run_dir / "gen" / f"{bench}.{s}.jsonl").exists()]
        for state in passes:
            gens = None if state == "gold" else by_id(
                read_jsonl(args.run_dir / "gen" / f"{bench}.{state}.jsonl"))
            rows = score_state(bench, items, gens, state=state, run_dir=args.run_dir,
                               data_dir=data_dir, harness_src=harness_src,
                               workers=args.workers, image=args.image)
            out = args.run_dir / "scores" / f"{bench}.{state}.jsonl"
            write_jsonl(out, rows)
            n_pass = sum(r["passed"] for r in rows)
            print(f"[score] {bench} {state}: {n_pass}/{len(rows)} passed "
                  f"({100 * n_pass / max(1, len(rows)):.1f}%) -> {out}", flush=True)


if __name__ == "__main__":
    main()
