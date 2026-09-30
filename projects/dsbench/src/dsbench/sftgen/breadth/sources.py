"""Registry of PUBLIC breadth/replay sources for the pilot mixture (ADR-001 §B.4).

Two hard rules from ADR-001, encoded here so the registry itself is the audit trail:

  * **Teacher licensing.** Only teachers whose licence permits distil-and-redistribute may enter a
    shippable model: Qwen (Apache-2.0), DeepSeek (MIT), GLM (MIT), gpt-oss (Apache-2.0), Kimi K2
    (modified MIT). OpenAI / Anthropic / Google API outputs may NOT, and an MIT/CC tag on an HF
    repo does not override the upstream provider ToS. Each `Source` carries `teacher` +
    `redistributable`; a source whose rows are GPT/Claude/Gemini-derived is marked
    `redistributable=False` (study-only)
    or filtered row-by-row (`row_ok`) so tainted rows never reach a shippable slice.
  * **Decontamination vs dsbench** runs at acquisition (here) AND again at mixture assembly.

`normalize(row) -> record | None` maps a source row to the training shape
`{messages, tools?, loss_mask_roles:["assistant"], meta:{...}}` (None = drop the row). `row_ok(row)`
is a cheap pre-filter (e.g. keep only clean-teacher subsets of a mixed set). `datasets` is imported
lazily in `acquire.py`, so importing this registry never requires the heavy dep (CI-safe).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

# Teacher families whose licences permit training-and-redistribute (ADR-001 §B.4).
CLEAN_TEACHERS: tuple[str, ...] = ("Qwen", "DeepSeek", "GLM", "gpt-oss", "Kimi")
# Named sets to EXCLUDE from anything redistributable (Claude/GPT/Gemini teachers), per ADR-001.
TAINT_EXCLUDE: tuple[str, ...] = (
    "SWE-smith-trajectories", "SWE-Gym", "R2E-Gym", "Magicoder-OSS-Instruct",
    "Code-Feedback", "text_to_dbt", "Think2SQL",
)


@dataclass(frozen=True)
class Source:
    key: str                       # short local id (also the output filename stem)
    hf_id: str                     # HuggingFace dataset id
    config: str | None             # config/subset name (None = default)
    split: str
    bucket: str  # ds_notebooks|text_to_sql|data_eng|swe|general_code|replay
    licence: str
    teacher: str                   # who generated the assistant turns
    redistributable: bool          # teacher licence permits distil+redistribute (else study-only)
    gated: bool                    # needs an HF token / accepted terms to download
    normalize: Callable[[dict], dict | None]
    row_ok: Callable[[dict], bool] = lambda r: True
    heavy: bool = False            # too big to stream in bulk (shards); excluded from --all-public
    notes: str = ""
    revision: str | None = None    # a pinned commit, so a re-fetch returns the same rows


# ---- normalisers (pure; defensive .get so a malformed row is dropped, not fatal) ----

def _valid_messages(msgs: object) -> bool:
    """A usable chat trace: >=1 user turn and >=1 assistant turn with real content."""
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    has_user = "user" in roles
    has_asst = any(
        isinstance(m, dict)
        and m.get("role") == "assistant"
        and (m.get("content") or m.get("tool_calls"))
        for m in msgs
    )
    return has_user and has_asst


def _rec(messages: list, meta: dict, tools: object = None) -> dict:
    rec: dict = {"messages": messages, "loss_mask_roles": ["assistant"], "meta": meta}
    if tools:
        rec["tools"] = tools
    return rec


def _norm_messages_passthrough(row: dict) -> dict | None:
    """Sources already in OpenAI chat shape (jupyter-agent, Tulu-3): keep messages [+ tools]."""
    msgs = row.get("messages")
    if not _valid_messages(msgs):
        return None
    meta: dict = {}
    for k in ("id", "edu_score", "executor_type", "source", "question"):
        if k in row and row[k] is not None:
            meta[k] = row[k]
    return _rec(msgs, meta, tools=row.get("tools"))


def _norm_jupyter_agent(row: dict) -> dict | None:
    """jupyter-agent messages are OpenAI-ish but store `tool_calls[].function.arguments` as a DICT
    (not the JSON string the Qwen/OpenAI template expects) and omit call ids. Rewrite to the
    standard shape our Target C rows already use: string arguments, a `type`/`id` per call, and each
    tool response linked to its call id (positional 1:1, as this dataset is structured)."""
    msgs = row.get("messages")
    if not _valid_messages(msgs):
        return None
    out: list[dict] = []
    pending_ids: list[str] = []  # tool_call ids from the latest assistant turn, to link responses
    resp_ix = 0
    call_ix = 0
    for m in msgs:
        role = m.get("role")
        nm: dict = {"role": role, "content": m.get("content") or ""}
        tcs = m.get("tool_calls")
        if role == "assistant" and isinstance(tcs, list) and tcs:
            norm: list[dict] = []
            for tc in tcs:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args if args is not None else {}, ensure_ascii=False)
                cid = tc.get("id") or f"call_{call_ix}"
                call_ix += 1
                norm.append({"id": cid, "type": "function",
                             "function": {"name": fn.get("name"), "arguments": args}})
            nm["tool_calls"] = norm
            pending_ids = [c["id"] for c in norm]
            resp_ix = 0
        elif role == "tool" and pending_ids:
            nm["tool_call_id"] = pending_ids[min(resp_ix, len(pending_ids) - 1)]
            resp_ix += 1
        out.append(nm)
    meta = {k: row[k] for k in ("id", "edu_score", "executor_type", "kaggle_dataset_name")
            if row.get(k) is not None}
    return _rec(out, meta, tools=row.get("tools"))


def _norm_trajectory(row: dict) -> dict | None:
    """DataMind-12K: the completed rollout lives in `trajectory` (a message list)."""
    msgs = row.get("trajectory")
    if not _valid_messages(msgs):
        return None
    meta = {k: row[k] for k in ("db_id", "task_id", "ability", "data_source") if row.get(k)}
    return _rec(msgs, meta)


def _norm_instruction_output(row: dict) -> dict | None:
    """OpenCoder opc-sft-stage2: {instruction, output[, testcase]} -> a 2-turn chat.

    These carry `testcase` asserts (execution-checkable); we keep that provenance in
    meta (a later pass can re-verify in the sandbox, mirroring our own generators' discipline).
    """
    ins, out = row.get("instruction"), row.get("output")
    if not (isinstance(ins, str) and ins.strip() and isinstance(out, str) and out.strip()):
        return None
    msgs = [{"role": "user", "content": ins}, {"role": "assistant", "content": out}]
    meta: dict = {}
    for k in ("seq_id", "entry_point"):
        if row.get(k) is not None:
            meta[k] = row[k]
    if row.get("testcase"):
        meta["has_testcase"] = True
    return _rec(msgs, meta)


def _norm_text_to_sql(row: dict) -> dict | None:
    """SynSQL-2.5M / OmniSQL: schema + question (+ external knowledge / CoT) -> SQL answer.

    Field names are confirmed at acquire time from a live row; this handles the documented shape and
    drops anything that doesn't carry a question + SQL so a schema drift fails closed, not silently.
    """
    q = row.get("question") or row.get("query") or row.get("sql_prompt")
    sql = row.get("sql") or row.get("SQL") or row.get("answer")
    if not (isinstance(q, str) and q.strip() and isinstance(sql, str) and sql.strip()):
        return None
    schema = (row.get("schema") or row.get("ddl") or row.get("db_schema")
              or row.get("sql_context") or "")
    know = row.get("external_knowledge") or row.get("evidence") or ""
    cot = (row.get("cot") or row.get("chain_of_thought") or row.get("reasoning")
           or row.get("sql_explanation") or "")
    user = (f"{schema}\n\n" if schema else "") + (f"Knowledge: {know}\n\n" if know else "") + q
    answer = (f"{cot}\n\n" if cot else "") + f"```sql\n{sql.strip()}\n```"
    msgs = [{"role": "user", "content": user.strip()}, {"role": "assistant", "content": answer}]
    meta = {k: row[k] for k in ("db_id", "sql_complexity") if row.get(k)}
    return _rec(msgs, meta)


# Tulu-3 is a MIXTURE of 19 subsets (its `source` field), and most can't go into a redistributable
# model. Some were written by a GPT-4 teacher (the persona sets, WildChat's answers, ...), and No
# Robots, though human-written, is CC BY-NC 4.0. Kept: exact subset names, never substrings, so a
# renamed or new subset fails closed instead of passing on a familiar word. Licences are as the
# mixture's card lists them (checked 2026-09-29). Aya and SciRIFF are clean too, but they come from
# their own repos (below), which keep the fields their gates need; the mixture keeps only messages.
_TULU_CLEAN_SOURCES: dict[str, str] = {
    "ai2-adapt-dev/oasst1_converted": "Apache-2.0",  # crowdsourced
    "ai2-adapt-dev/flan_v2_converted": "unspecified",  # templated tasks; the card gives no licence
    "ai2-adapt-dev/tulu_hard_coded_repeated_10": "CC-BY-4.0",  # hand-written identity prompts
}


def _tulu_row_ok(row: dict) -> bool:
    return (row.get("source") or "") in _TULU_CLEAN_SOURCES


# Tulu 3's Aya and SciRIFF subsets, read from their own repos (revisions pinned). Their rows are the
# mixture's (checked 2026-09-30: the same messages, row for row), and they keep what the mixture
# drops: the SciRIFF task and the Aya language.
#
# Aya (CohereLabs/aya_dataset, Apache-2.0, "for any purpose"): 100,000 rows in 71 languages, written
# by fluent speakers. About a third are re-annotations, human edits of machine-generated prompts and
# answers; the rest were written from scratch. Language and annotation type go to meta, so prompt
# selection can balance them; the annotator's hashed id stays behind.
def _norm_aya(row: dict) -> dict | None:
    msgs = row.get("messages")
    if not _valid_messages(msgs):
        return None
    meta = {k: row[k] for k in ("language", "language_code", "annotation_type") if row.get(k)}
    return _rec(msgs, meta)


# SciRIFF (allenai/SciRIFF, ODC-BY) repurposes existing scientific-literature datasets as tasks,
# and unlike FLAN v2's, its card lists each source's licence. Kept: the tasks whose source is
# under CC BY, CC0, Apache-2.0 or MIT (24 of the 45 in Tulu's sample, 4,904 of its 10,000 rows).
# Out: CC BY-NC (as No Robots), GPL-3.0 (copyleft), and the tasks with no licence listed, which
# ADR-004 decision 5 puts to the owner. As the card lists them (checked 2026-09-30); a task named
# in neither table fails closed.
_SCIRIFF_KEPT: dict[str, tuple[str, ...]] = {
    "CC BY": ("anat_em_ner", "bioasq_factoid_qa", "bioasq_general_qa", "bioasq_yesno_qa",
              "chia_ner", "ddi_ner", "genia_ner", "linnaeus_ner", "qasper_extractive_qa"),
    "CC 0": ("medmentions_ner", "ncbi_ner", "nlmchem_ner", "nlmgene_ner"),
    "Apache 2.0": ("covid_deepset_qa", "data_reco_mcq_mc", "data_reco_mcq_sc", "mltables_te",
                   "mslr2022_cochrane_multidoc_summarization",
                   "mslr2022_ms2_multidoc_summarization", "scitldr_aic"),
    "MIT": ("annotated_materials_syntheses_events", "multixscience_multidoc_summarization",
            "pubmedqa_qa", "qasa_abstractive_qa"),
}
_SCIRIFF_LEFT_OUT: dict[str, tuple[str, ...]] = {
    "CC BY-NC": ("scireviewgen_multidoc_summarization",),
    "GPL 3.0": ("chemtables_te",),
    "none listed": ("acl_arc_intent_classification", "bc7_litcovid_topic_classification",
                    "cdr_ner", "chemdner_ner", "chemprot_ner", "chemprot_re",
                    "chemsum_single_document_summarization", "covidfact_entailment",
                    "craftchem_ner", "drug_combo_extraction_re", "gnormplus_ner",
                    "healthver_entailment", "pico_ner", "scicite_classification",
                    "scientific_lay_summarisation_elife_single_doc_summ",
                    "scientific_lay_summarisation_plos_single_doc_summ",
                    "scientific_papers_summarization_single_doc_arxiv",
                    "scientific_papers_summarization_single_doc_pubmed", "scierc_re"),
}
_SCIRIFF_TASK_LICENCE: dict[str, str] = {
    task: licence for licence, tasks in _SCIRIFF_KEPT.items() for task in tasks
}


def _sciriff_task(row: dict) -> str:
    return (row.get("dataset") or "").removeprefix("science.")


def _sciriff_row_ok(row: dict) -> bool:
    return _sciriff_task(row) in _SCIRIFF_TASK_LICENCE


def _norm_sciriff(row: dict) -> dict | None:
    msgs = row.get("messages")
    task = _sciriff_task(row)
    if not _valid_messages(msgs) or task not in _SCIRIFF_TASK_LICENCE:
        return None
    meta = {"task": task, "task_licence": _SCIRIFF_TASK_LICENCE[task]}
    if row.get("id"):
        meta["id"] = row["id"]
    return _rec(msgs, meta)


# GSM8K (Cobbe et al., 2021): grade-school maths word problems. Hired writers (Upwork, then Surge
# AI) wrote the questions and their worked solutions. OpenAI released it under MIT, but no model
# wrote it, so ADR-001's teacher rule doesn't apply. Train split only: test is GSM8K's benchmark.
# Revision 2 uses the questions as replay prompts and the base writes the answers, so the solution
# stays only as a clean reference:
# - the calculator annotations (`<<48/2=24>>`) are stripped, as the dataset's README says to do.
#   They were a hook for the paper's models, which handed the arithmetic to a calculator;
# - the last line, `#### 72`, becomes a sentence, and its number goes to meta as the gold answer,
#   so the base's answers can be checked against it.
_GSM8K_CALC = re.compile(r"<<[^<>]*>>")
_GSM8K_FINAL = re.compile(r"\n####\s*(\S+)\s*\Z")


def _norm_gsm8k(row: dict) -> dict | None:
    """{question, answer} -> a 2-turn chat; `meta.gold_answer` is the final number, commas out."""
    q, a = row.get("question"), row.get("answer")
    if not (isinstance(q, str) and q.strip() and isinstance(a, str)) or a.count("####") != 1:
        return None
    m = _GSM8K_FINAL.search(a)
    if m is None:
        return None
    steps = _GSM8K_CALC.sub("", a[: m.start()]).strip()
    final = m.group(1)
    answer = f"{steps}\n\nThe answer is {final}." if steps else f"The answer is {final}."
    msgs = [{"role": "user", "content": q.strip()}, {"role": "assistant", "content": answer}]
    return _rec(msgs, {"gold_answer": final.replace(",", "")})


# ADR-001 non-negotiable: exclude SWE-bench-Verified's repos from every SWE source (they are an eval
# target). Conservative: a trajectory that names any of these owner/repo slugs anywhere is dropped.
_SWEBENCH_VERIFIED_REPOS: tuple[str, ...] = (
    "astropy/astropy", "django/django", "matplotlib/matplotlib", "mwaskom/seaborn",
    "pallets/flask", "psf/requests", "pydata/xarray", "pylint-dev/pylint", "pytest-dev/pytest",
    "scikit-learn/scikit-learn", "sphinx-doc/sphinx", "sympy/sympy",
)


def _swe_row_ok(row: dict) -> bool:
    text = " ".join(str(m.get("content") or "") for m in (row.get("messages") or [])).lower()
    return not any(repo in text for repo in _SWEBENCH_VERIFIED_REPOS)


# ---- the registry ----

SOURCES: list[Source] = [
    Source(
        key="jupyter_agent", hf_id="jupyter-agent/jupyter-agent-dataset", config="default",
        split="non_thinking", bucket="ds_notebooks", licence="Apache-2.0", teacher="Qwen",
        redistributable=True, gated=False, normalize=_norm_jupyter_agent,
        notes="E2B-executed DS notebooks; messages+tools native. Also a `thinking` split.",
    ),
    Source(
        key="opencoder_edu", hf_id="OpenCoder-LLM/opc-sft-stage2", config="educational_instruct",
        split="train", bucket="general_code", licence="Apache-2.0 (MIT code)",
        teacher="open pipeline", redistributable=True, gated=False,
        normalize=_norm_instruction_output,
        notes="Has executable `testcase` asserts; also a `package_instruct` config.",
    ),
    Source(
        key="gretel_sql", hf_id="gretelai/synthetic_text_to_sql", config="default", split="train",
        bucket="text_to_sql", licence="Apache-2.0", teacher="Gretel Navigator",
        redistributable=True, gated=False, normalize=_norm_text_to_sql,
        notes="Self-contained: embeds the DDL per row (sql_context) + explanation. Clean teacher.",
    ),
    Source(
        key="synsql", hf_id="seeklhy/SynSQL-2.5M", config="default", split="train",
        bucket="text_to_sql", licence="Apache-2.0", teacher="Qwen (verify)", redistributable=True,
        gated=False, heavy=True, normalize=_norm_text_to_sql,
        notes="OmniSQL SynSQL-2.5M: 2.5M rows in huge shards (heavy to stream) and schema is by "
              "db_id, not per-row -> needs a schema-join. Prefer gretel_sql for the pilot.",
    ),
    Source(
        key="datamind", hf_id="zjunlp/DataMind-12K", config="default", split="train",
        bucket="ds_notebooks", licence="MIT", teacher="DataMind pipeline", redistributable=True,
        gated=False, normalize=_norm_trajectory,
        notes="Data-analysis rollouts in `trajectory`.",
    ),
    Source(
        key="swe_swiss", hf_id="SWE-Swiss/SWESwiss-SFT-Merged-10K", config="default", split="train",
        bucket="swe", licence="MIT", teacher="DeepSeek-R1", redistributable=True, gated=False,
        normalize=_norm_messages_passthrough, row_ok=_swe_row_ok,
        notes="ADR-named SWE SFT, public at the un-hyphenated id. row_ok drops any trajectory "
              "naming a SWE-bench-Verified repo (eval hygiene).",
    ),
    Source(
        key="tulu3", hf_id="allenai/tulu-3-sft-mixture", config="default", split="train",
        bucket="replay", licence="ODC-BY", teacher="mixed (row-filtered)", redistributable=True,
        gated=False, normalize=_norm_messages_passthrough, row_ok=_tulu_row_ok,
        notes="Mixture: only clean-origin, redistributable subsets kept via row_ok (exact names); "
              "GPT-teacher subsets and No Robots (CC BY-NC) dropped; Aya and SciRIFF come from "
              "their own repos.",
    ),
    Source(
        key="tulu3_aya", hf_id="ai2-adapt-dev/tulu_v3.9_aya_100k", config="default",
        split="train", revision="22532285925b4e2dd2895a68ae3daa2bf8ddac31", bucket="replay",
        licence="Apache-2.0", teacher="human-written", redistributable=True, gated=False,
        normalize=_norm_aya,
        notes="Tulu 3's Aya subset (the mixture's rows) from its own repo, which keeps the "
              "language and annotation type. Replay prompts (ADR-004 Rev 2).",
    ),
    Source(
        key="tulu3_sciriff", hf_id="ai2-adapt-dev/tulu_v3.9_sciriff_10k", config="default",
        split="train", revision="2974056c17086b2fcfd2758f94a3ad922e8d1b41", bucket="replay",
        licence="ODC-BY-1.0, plus each task's source licence (meta.task_licence)",
        teacher="none (existing datasets, templated)", redistributable=True, gated=False,
        normalize=_norm_sciriff, row_ok=_sciriff_row_ok,
        notes="Tulu 3's SciRIFF subset (the mixture's rows) from its own repo, which keeps the "
              "task; row_ok keeps the tasks whose source licence is permissive.",
    ),
    Source(
        key="gsm8k", hf_id="openai/gsm8k", config="main", split="train", bucket="replay",
        licence="MIT", teacher="human-written", redistributable=True, gated=False,
        normalize=_norm_gsm8k,
        notes="Grade-school maths by hired writers; released by OpenAI, written by no model. "
              "Replay prompts (ADR-004 Rev 2); the gold number is kept in meta.",
    ),
]

SOURCES_BY_KEY: dict[str, Source] = {s.key: s for s in SOURCES}


def public_sources() -> list[Source]:
    """Non-gated sources safe to stream in bulk (`--all-public`); heavy sources are opt-in only."""
    return [s for s in SOURCES if not s.gated and not s.heavy]
