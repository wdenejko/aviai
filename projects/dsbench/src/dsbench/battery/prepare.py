"""Build the battery's items from pinned public datasets, and fetch the pinned reference scorers.

Everything is pinned: each dataset to a Hugging Face revision, each reference scorer to a git
commit. Subsets are drawn with a fixed seed and stratified where the benchmark has strata. Rerunning
`prepare` therefore reproduces the same items byte for byte. `manifest.json` records the revisions,
the subset rules and a sha256 of every items file, so a report can say exactly what it measured.

Prompts follow each benchmark's own chat protocol where one exists (DS-1000 `run_openai.py`,
LiveCodeBench `get_generic_question_template_answer`, EvalPlus' chat instruction, MMLU-Pro's CoT
instruction, BFCL's function-calling conversion). Where none exists (GPQA, BIRD), the prompt is
a common public one and is documented next to its builder.

Run (on the box; needs network):
    python -m dsbench.battery.prepare --data-dir DATA --items-dir ITEMS \
        --bench humaneval_plus,ds1000,ifeval
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import pickle
import random
import re
import sqlite3
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

import httpx

from dsbench.battery.items import Item, save_items

SEED = 20260924

DATASETS = {
    "humaneval_plus": ("evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"),
    "ds1000": ("xlangai/DS-1000", "4416080ac5cb80bdf7576aefb8f9a0b4d5426a44"),
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be"),
    "lcb": ("livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"),
    "gpqa": ("Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6"),
    "bfcl": ("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
             "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"),
}
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"  # official mirror, 346 MB

# Reference scorers, fetched (not vendored) at pinned commits. See score.py for how each is used.
_GR = "https://raw.githubusercontent.com/google-research/google-research"
_IFEVAL_COMMIT = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
_LCB = "https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench"
_LCB_COMMIT = "16de2a1ef5c07342f6b9782a415cc29cee9ce6e4"
_BFCL = "https://raw.githubusercontent.com/ShishirPatil/gorilla"
_BFCL_COMMIT = "58f57e9124ea981403792dd51e00a6577e621fae"
REFS = {
    **{
        f"instruction_following_eval/{f}":
            f"{_GR}/{_IFEVAL_COMMIT}/instruction_following_eval/{f}"
        for f in ("instructions.py", "instructions_registry.py", "instructions_util.py",
                  "evaluation_lib.py")
    },
    "lcb_testing_util.py": f"{_LCB}/{_LCB_COMMIT}/lcb_runner/evaluation/testing_util.py",
    "bfcl_eval/eval_checker/ast_eval/ast_checker.py":
        f"{_BFCL}/{_BFCL_COMMIT}/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/"
        "ast_eval/ast_checker.py",
}


# --- fetching ----------------------------------------------------------------------------------


def hf_file(bench: str, filename: str, data_dir: Path) -> Path:
    """Download one file of a pinned dataset revision (gated sets read HF_TOKEN from the env)."""
    from huggingface_hub import hf_hub_download

    repo, revision = DATASETS[bench]
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                cache_dir=str(data_dir / "hf")))


def url_file(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def fetch_refs(refs_dir: Path) -> dict[str, str]:
    """Fetch the pinned reference scorers; return {path: sha256} for the manifest."""
    out = {}
    for rel, url in REFS.items():
        path = url_file(url, refs_dir / rel)
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    for pkg in ("instruction_following_eval", "bfcl_eval", "bfcl_eval/eval_checker",
                "bfcl_eval/eval_checker/ast_eval"):
        (refs_dir / pkg / "__init__.py").touch()
    _write_bfcl_shims(refs_dir)
    return out


def _write_bfcl_shims(refs_dir: Path) -> None:
    """Minimal stand-ins for the four bfcl_eval modules `ast_checker.py` imports.

    Only the Python categories are scored, so the Java/JS converters are never reached. The one
    behaviour that matters is `underscore_to_dot`: OpenAI-style tool names can't contain '.', so
    BFCL sends `math.factorial` as `math_factorial` to function-calling models and its checker maps
    the expected name the same way. `build_bfcl` renames the tools to match.
    """
    files = {
        "bfcl_eval/constants/__init__.py": "",
        "bfcl_eval/constants/enums.py":
            "from enum import Enum\n\n\nclass Language(Enum):\n    PYTHON = 'python'\n"
            "    JAVA = 'java'\n    JAVASCRIPT = 'javascript'\n",
        "bfcl_eval/constants/model_config.py":
            "from collections import defaultdict\nfrom types import SimpleNamespace\n\n"
            "MODEL_CONFIG_MAPPING = defaultdict(lambda: SimpleNamespace(underscore_to_dot=True))\n",
        "bfcl_eval/constants/type_mappings.py":
            "JAVA_TYPE_CONVERSION = {}\nJS_TYPE_CONVERSION = {}\n",
        "bfcl_eval/eval_checker/ast_eval/type_convertor/__init__.py": "",
        "bfcl_eval/eval_checker/ast_eval/type_convertor/java_type_converter.py":
            "def java_type_converter(*a, **k):\n    raise NotImplementedError('python only')\n",
        "bfcl_eval/eval_checker/ast_eval/type_convertor/js_type_converter.py":
            "def js_type_converter(*a, **k):\n    raise NotImplementedError('python only')\n",
    }
    for rel, text in files.items():
        path = refs_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


# --- subsets -----------------------------------------------------------------------------------


def stratified(rows: list, key, per_stratum: int | None, seed: int = SEED) -> list:
    """Up to `per_stratum` rows from each stratum, drawn with a fixed seed, order preserved."""
    if per_stratum is None:
        return rows
    groups: dict = defaultdict(list)
    for i, row in enumerate(rows):
        groups[key(row)].append(i)
    rng = random.Random(seed)
    keep = set()
    for stratum in sorted(groups):
        idx = groups[stratum]
        keep.update(idx if len(idx) <= per_stratum else rng.sample(idx, per_stratum))
    return [row for i, row in enumerate(rows) if i in keep]


def _jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- builders ----------------------------------------------------------------------------------

# EvalPlus' chat instruction (evalplus/provider/utility.py).
_EVALPLUS_INSTRUCTION = ("Please provide a self-contained Python script that solves the following "
                         "problem in a markdown code block:")


def build_humaneval_plus(data_dir: Path, args) -> list[Item]:
    rows = _jsonl(hf_file("humaneval_plus", "test.jsonl", data_dir))
    return [
        Item(bench="humaneval_plus", id=r["task_id"],
             messages=[{"role": "user",
                        "content": f"{_EVALPLUS_INSTRUCTION}\n```\n{r['prompt'].strip()}\n```\n"}],
             gen={"max_tokens": 2048},
             ref={"prompt": r["prompt"], "entry_point": r["entry_point"], "test": r["test"],
                  "canonical_solution": r["canonical_solution"]})
        for r in rows
    ]


# DS-1000's chat protocol (run_openai.py): this system prompt, max_tokens 1024, these stops.
_DS1000_SYSTEM = ("Write a short code following the given format and indentation. Place the "
                  "executable code between <code> and </code> tags, without any other "
                  "non-executable things.")


def build_ds1000(data_dir: Path, args) -> list[Item]:
    rows = _jsonl(hf_file("ds1000", "test.jsonl", data_dir))
    return [
        Item(bench="ds1000", id=str(r["metadata"]["problem_id"]),
             messages=[{"role": "system", "content": _DS1000_SYSTEM},
                       {"role": "user", "content": r["prompt"]}],
             gen={"max_tokens": 1024, "stop": ["</code>", "# SOLUTION END"]},
             ref={"code_context": r["code_context"], "reference_code": r["reference_code"]},
             meta={"library": r["metadata"]["library"],
                   "perturbation": r["metadata"]["perturbation_type"]})
        for r in rows
    ]


def build_ifeval(data_dir: Path, args) -> list[Item]:
    rows = _jsonl(hf_file("ifeval", "ifeval_input_data.jsonl", data_dir))
    # 4096 tokens: some prompts ask for 600+ words, and a truncated reply fails length checks for a
    # reason that is about the harness, not the model.
    return [
        Item(bench="ifeval", id=str(r["key"]),
             messages=[{"role": "user", "content": r["prompt"]}],
             gen={"max_tokens": 4096},
             ref={"key": r["key"], "prompt": r["prompt"],
                  "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]})
        for r in rows
    ]


# MMLU-Pro's CoT instruction (evaluate_from_api.py), zero-shot: the 5-shot exemplars would add
# ~2k prompt tokens per item without changing what a paired regression test measures.
_MMLU_PRO_HEAD = ("The following are multiple choice questions (with answers) about {cat}. Think "
                  "step by step and then finish your answer with \"the answer is (X)\" where X is "
                  "the correct letter choice.")


def build_mmlu_pro(data_dir: Path, args) -> list[Item]:
    import pyarrow.parquet as pq

    table = pq.read_table(hf_file("mmlu_pro", "data/test-00000-of-00001.parquet", data_dir))
    rows = sorted(table.to_pylist(), key=lambda r: (r["category"], r["question_id"]))
    rows = stratified(rows, lambda r: r["category"], args.mmlu_pro_per_category)
    items = []
    for r in rows:
        options = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(r["options"]))
        text = (f"{_MMLU_PRO_HEAD.format(cat=r['category'])}\n\nQuestion: {r['question']}\n"
                f"Options:\n{options}\nAnswer: Let's think step by step.")
        items.append(Item(bench="mmlu_pro", id=str(r["question_id"]),
                          messages=[{"role": "user", "content": text}],
                          gen={"max_tokens": 4096},
                          ref={"answer": r["answer"]}, meta={"category": r["category"]}))
    return items


# simple-evals' multiple-choice template (openai/simple-evals, QUERY_TEMPLATE_MULTICHOICE).
_GPQA_TEMPLATE = ("Answer the following multiple choice question. The last line of your response "
                  "should be of the following format: 'Answer: $LETTER' (without quotes) where "
                  "LETTER is one of ABCD. Think step by step before answering.\n\n{q}\n\n"
                  "A) {a}\nB) {b}\nC) {c}\nD) {d}")


def build_gpqa(data_dir: Path, args) -> list[Item]:
    """GPQA diamond (gated: accept the terms on HF and export HF_TOKEN; this code never sees it).

    The correct option's position is shuffled per question with a seed derived from the record id,
    so the letter distribution is balanced and identical across states and reruns.
    """
    path = hf_file("gpqa", "gpqa_diamond.csv", data_dir)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    items = []
    for r in rows:
        options = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"],
                   r["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(f"{SEED}:{r['Record ID']}").shuffle(order)
        shuffled = [options[i].strip() for i in order]
        answer = "ABCD"[order.index(0)]
        text = _GPQA_TEMPLATE.format(q=r["Question"].strip(), a=shuffled[0], b=shuffled[1],
                                     c=shuffled[2], d=shuffled[3])
        items.append(Item(bench="gpqa", id=r["Record ID"],
                          messages=[{"role": "user", "content": text}],
                          gen={"max_tokens": 4096}, ref={"answer": answer},
                          meta={"domain": r.get("High-level domain", "")}))
    return items


# LiveCodeBench's generic chat prompt (lcb_runner/prompts/code_generation.py).
_LCB_SYSTEM = ("You are an expert Python programmer. You will be given a question (problem "
               "specification) and will generate a correct Python program that matches the "
               "specification and passes all tests.")
_LCB_STARTER = ("You will use the following starter code to write the solution to the problem and "
                "enclose your code within delimiters.")
_LCB_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not "
              "directly test on the sample inputs). Enclose your code within delimiters as "
              "follows. Ensure that when the python program runs, it reads the inputs, runs the "
              "algorithm and writes output to STDOUT.")


class _NoGlobalsUnpickler(pickle.Unpickler):
    """LiveCodeBench ships private tests as base64(zlib(pickle(json_string))). Unpickling
    third-party data can run arbitrary code, so refuse every global: a plain string needs none."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def _lcb_private(blob: str) -> list:
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        raw = zlib.decompress(base64.b64decode(blob.encode("utf-8")))
        return json.loads(_NoGlobalsUnpickler(io.BytesIO(raw)).load())


def _lcb_prompt(r: dict) -> str:
    text = f"### Question:\n{r['question_content']}\n\n"
    if r["starter_code"]:
        text += f"### Format: {_LCB_STARTER}\n```python\n{r['starter_code']}\n```\n\n"
    else:
        text += f"### Format: {_LCB_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n"
    return text + "### Answer: (use the provided format with backticks)\n\n"


def build_lcb(data_dir: Path, args) -> list[Item]:
    """The v5 and v6 additions (contests from late 2024 to spring 2025), all difficulties.

    Tests go to one JSON file per problem under DATA/lcb_tests (some problems carry megabytes of
    private tests); the item only points at that file, and the sandbox mounts DATA read-only.
    """
    tests_dir = data_dir / "lcb_tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.lcb_files.split(","):
        rows.extend(_jsonl(hf_file("lcb", name, data_dir)))
    rows.sort(key=lambda r: (r["contest_date"], r["question_id"]))
    rows = stratified(rows, lambda r: r["difficulty"], args.lcb_per_difficulty)
    items = []
    for r in rows:
        tests = json.loads(r["public_test_cases"]) + _lcb_private(r["private_test_cases"])
        fn_name = json.loads(r["metadata"] or "{}").get("func_name")
        io_spec = {"inputs": [t["input"] for t in tests], "outputs": [t["output"] for t in tests],
                   "fn_name": fn_name}
        rel = f"lcb_tests/{r['question_id']}.json"
        (data_dir / rel).write_text(json.dumps(io_spec))
        items.append(Item(bench="lcb", id=r["question_id"],
                          messages=[{"role": "system", "content": _LCB_SYSTEM},
                                    {"role": "user", "content": _lcb_prompt(r)}],
                          gen={"max_tokens": 4096},
                          ref={"tests": rel, "n_tests": len(tests)},
                          meta={"difficulty": r["difficulty"], "platform": r["platform"],
                                "contest_date": r["contest_date"]}))
    return items


def _bird_root(data_dir: Path) -> Path:
    """Download and unpack BIRD dev once; return the directory holding dev.json."""
    archive = url_file(BIRD_DEV_URL, data_dir / "bird" / "dev.zip")
    root = data_dir / "bird"
    found = list(root.rglob("dev.json"))
    if not found:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
        found = list(root.rglob("dev.json"))
    base = found[0].parent
    if not (base / "dev_databases").exists():
        with zipfile.ZipFile(base / "dev_databases.zip") as zf:
            zf.extractall(base)
    return base


def _bird_schema(db_path: Path, sample_rows: int = 3) -> str:
    """CREATE statements as stored in the database, each followed by a few example rows.

    This is the schema prompt most BIRD baselines use (DDL + 3 rows per table): the rows show value
    formats (dates, codes, casing) that the DDL alone hides.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    parts = []
    tables = con.execute("SELECT name, sql FROM sqlite_master WHERE type='table' "
                         "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    for name, ddl in tables:
        rows = con.execute(f'SELECT * FROM "{name}" LIMIT {sample_rows}').fetchall()
        shown = "\n".join(" | ".join(str(v)[:60] for v in row) for row in rows)
        parts.append(f"{ddl.strip()};\n/* {sample_rows} example rows:\n{shown}\n*/")
    con.close()
    return "\n\n".join(parts)


def build_bird(data_dir: Path, args) -> list[Item]:
    base = _bird_root(data_dir)
    rows = json.loads((base / "dev.json").read_text())
    for i, r in enumerate(rows):
        r.setdefault("question_id", i)
    # Sorted by database so consecutive requests share the schema prefix (llama-server reuses a
    # slot's cached prompt prefix, which turns ~5k-token schemas into a one-off cost per slot).
    rows.sort(key=lambda r: (r["db_id"], r["question_id"]))
    rows = stratified(rows, lambda r: r["difficulty"], args.bird_per_difficulty)
    schemas: dict[str, str] = {}
    items = []
    for r in rows:
        db = r["db_id"]
        rel = f"{base.relative_to(data_dir)}/dev_databases/{db}/{db}.sqlite"
        if db not in schemas:
            schemas[db] = _bird_schema(data_dir / rel)
        evidence = f"\n-- External Knowledge: {r['evidence']}" if r.get("evidence") else ""
        text = (f"{schemas[db]}\n{evidence}\n-- Using valid SQLite and understanding External "
                f"Knowledge, answer the following question for the tables provided above.\n"
                f"-- Question: {r['question']}\n\nReturn only the SQL query, in a ```sql block.")
        items.append(Item(bench="bird", id=str(r["question_id"]),
                          messages=[{"role": "system",
                                     "content": "You are a careful SQLite expert."},
                                    {"role": "user", "content": text}],
                          gen={"max_tokens": 1024},
                          ref={"db": rel, "gold_sql": r["SQL"]},
                          meta={"difficulty": r["difficulty"], "db_id": db}))
    return items


# BFCL's Gorilla -> OpenAPI type map (bfcl_eval/constants/type_mappings.py, GORILLA_TO_OPENAPI).
_GORILLA_TO_OPENAPI = {
    "integer": "integer", "number": "number", "float": "number", "string": "string",
    "boolean": "boolean", "bool": "boolean", "array": "array", "list": "array", "dict": "object",
    "object": "object", "tuple": "array", "any": "string", "byte": "integer", "short": "integer",
    "long": "integer", "double": "number", "char": "string", "ArrayList": "array",
    "Array": "array", "HashMap": "object", "Hashtable": "object", "Queue": "array",
    "Stack": "array", "Any": "string", "String": "string", "Bigint": "integer",
}
BFCL_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")


def _cast_openapi(properties: dict) -> dict:
    """BFCL's `_cast_to_openai_type`, restricted to the shapes the Python categories use."""
    for value in properties.values():
        gorilla_type = value.get("type", "string")
        if gorilla_type == "float":
            value["format"] = "float"
            value["description"] = value.get("description", "") + " This is a float type value."
        value["type"] = _GORILLA_TO_OPENAPI.get(gorilla_type, "string")
        if value["type"] in ("array", "object"):
            if "properties" in value:
                value["properties"] = _cast_openapi(value["properties"])
            elif "items" in value:
                items = value["items"]
                items["type"] = _GORILLA_TO_OPENAPI.get(items.get("type", "string"), "string")
                if items["type"] == "object" and "properties" in items:
                    items["properties"] = _cast_openapi(items["properties"])
                elif items["type"] == "array" and "items" in items:
                    inner = items["items"]
                    inner["type"] = _GORILLA_TO_OPENAPI.get(inner.get("type", "string"), "string")
    return properties


def bfcl_tools(functions: list[dict]) -> list[dict]:
    """BFCL function docs -> OpenAI `tools`, as BFCL prepares them for function-calling models."""
    tools = []
    for fn in copy.deepcopy(functions):
        fn["name"] = re.sub(r"\.", "_", fn["name"])
        fn["description"] = fn.get("description", "") + (
            " Note that the provided function is in Python 3 syntax.")
        params = fn.setdefault("parameters", {"type": "dict", "properties": {}})
        params["type"] = "object"
        params["properties"] = _cast_openapi(params.get("properties", {}))
        tools.append({"type": "function", "function": fn})
    return tools


def build_bfcl(data_dir: Path, args) -> list[Item]:
    items = []
    for cat in BFCL_CATEGORIES:
        rows = _jsonl(hf_file("bfcl", f"BFCL_v3_{cat}.json", data_dir))
        answers = {}
        if cat != "irrelevance":
            answers = {a["id"]: a["ground_truth"] for a in
                       _jsonl(hf_file("bfcl", f"possible_answer/BFCL_v3_{cat}.json", data_dir))}
        for r in rows:
            items.append(Item(bench="bfcl", id=r["id"], messages=r["question"][0],
                              gen={"max_tokens": 1024, "tools": bfcl_tools(r["function"])},
                              ref={"category": cat, "functions": r["function"],
                                   "ground_truth": answers.get(r["id"])},
                              meta={"category": cat}))
    return items


BUILDERS = {
    "humaneval_plus": build_humaneval_plus, "ds1000": build_ds1000, "ifeval": build_ifeval,
    "mmlu_pro": build_mmlu_pro, "gpqa": build_gpqa, "lcb": build_lcb, "bird": build_bird,
    "bfcl": build_bfcl,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--items-dir", required=True, type=Path)
    ap.add_argument("--bench", default=",".join(BUILDERS))
    ap.add_argument("--refs", action="store_true", help="also fetch the pinned reference scorers")
    ap.add_argument("--mmlu-pro-per-category", type=int, default=100)
    ap.add_argument("--lcb-files", default="test5.jsonl,test6.jsonl")
    ap.add_argument("--lcb-per-difficulty", type=int, default=None)
    ap.add_argument("--bird-per-difficulty", type=int, default=None)
    args = ap.parse_args()

    manifest_path = args.items_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("benches", {})
    manifest["seed"] = SEED
    if args.refs:
        manifest["refs"] = fetch_refs(args.data_dir / "refs")
        print(f"refs: {len(manifest['refs'])} files")
    for bench in args.bench.split(","):
        items = BUILDERS[bench](args.data_dir, args)
        path = args.items_dir / f"{bench}.jsonl"
        n = save_items(path, items)
        manifest["benches"][bench] = {
            "n": n, "source": DATASETS.get(bench, (BIRD_DEV_URL, None)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "subset": {k: v for k, v in vars(args).items()
                       if k.startswith(bench.split("_")[0]) and v is not None},
        }
        print(f"{bench}: {n} items -> {path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
