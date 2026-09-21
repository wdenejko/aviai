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


# Tulu-3 is a MIXTURE; some subsets are GPT-4-teacher (personahub, wildchat, ...). Keep only
# provably-clean origins for a redistributable slice. Conservative allow-list on the `source` field.
_TULU_CLEAN_SOURCE_SUBSTR: tuple[str, ...] = (
    "oasst", "no_robots", "flan_v2", "hard_coded", "aya", "sciriff", "tulu_v3.9_sft",
)


def _tulu_row_ok(row: dict) -> bool:
    src = (row.get("source") or "").lower()
    return any(s in src for s in _TULU_CLEAN_SOURCE_SUBSTR)


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
        key="tulu3", hf_id="allenai/tulu-3-sft-mixture", config="default", split="train",
        bucket="replay", licence="ODC-BY", teacher="mixed (row-filtered)", redistributable=True,
        gated=False, normalize=_norm_messages_passthrough, row_ok=_tulu_row_ok,
        notes="Mixture: only clean-origin subsets kept via row_ok; GPT-teacher subsets dropped.",
    ),
]

SOURCES_BY_KEY: dict[str, Source] = {s.key: s for s in SOURCES}


def public_sources() -> list[Source]:
    """Non-gated sources safe to stream in bulk (`--all-public`); heavy sources are opt-in only."""
    return [s for s in SOURCES if not s.gated and not s.heavy]
