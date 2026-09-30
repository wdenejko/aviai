# dsbench targeted-SFT data zone (ADR-004)

Durable home for the **generated** fine-tune slices. The generators live in
`src/dsbench/sftgen/` (committed, reviewed, tested); the JSONL here is their **output**.

## Why this exists (and why it is gitignored)

Generated data is not source — it is reproducible from `(generator, seed, teacher)`. The repo's
data-zone rule (see `projects/avtext/data/README.md`) keeps bulk data out of git and tracks only
tiny **provenance anchors**. So here `*.jsonl` is ignored and only `README.md` + `manifest.json`
are committed.

The one nuance vs. a pure "regenerate on demand" stance: **Target C costs GPU hours** (a licence-clean
teacher run as a sandbox agent on the box), so we keep its slice on disk rather than only in a
session scratchpad. `manifest.json` records each file's sha256, row count and the exact regen command,
so a lost or stale file is unambiguous to rebuild.

## Contents

| File | Target | Skill | Provenance |
|---|---|---|---|
| `targetA_dialect_conventions.raw.jsonl` | A | SQL-dialect date/time conventions | **teacher-free**, execution-verified against 4 real engines |
| `targetA_dialect_conventions.train.jsonl` | A | (rendered from the raw) | Qwen chat-template training rows |
| `targetC_ml_delivery.jsonl` | C | agentic ML-workflow *delivery* | Ling-3.0-flash (MIT) run in the ADR-003 sandbox; **only oracle-passing** trajectories kept |
| `pilot_mixture.jsonl` | — | the assembled Gate-1 pilot | A + C + breadth/replay, decontaminated + proportioned (see `pilot_manifest.json`) |

Target **B** was measured to be a non-gap and **dropped** (see ADR-004); no B slice is carried.

## Pilot mixture (Gate 1)

`pilot_mixture.jsonl` is the assembled Gate-1 SFT mixture (ADR-001 Gate 1) — the shuffled union of
the targeted A/C slices with the public breadth/replay buckets, each sampled to its ADR-004
proportion, **re-decontaminated as a whole** against dsbench, in the common `{messages, tools?,
loss_mask_roles, meta}` shape (the Qwen chat template is applied by the trainer). `pilot_manifest.json`
(tracked) is the audit record: per-pool tokens/rows, licence+teacher, `dropped_contam`, and
`study_only_rows` (0 ⇒ every row is redistributable-clean). ~0.88M tokens (chars/3.5), targeted
~16.5% / breadth ~55% / replay ~29%. **Gap:** the data-engineering + exec-feedback breadth buckets
are own-generation tasks (like A/C), not yet built — left out and documented, not padded.

Rebuild it from the acquired pools:

```bash
# 1. acquire the breadth/replay buckets (needs `datasets`; writes breadth_<key>.jsonl to a scratch dir)
uv run --with datasets python -m dsbench.sftgen.breadth.acquire --all-public --out-dir <scratch>
# 2. assemble: sample each pool to its ADR-004 target, decontaminate the whole, shuffle
uv run python -m dsbench.sftgen.assemble --data-zone data/sft --breadth-dir <scratch> \
    --out data/sft/pilot_mixture.jsonl --report data/sft/pilot_manifest.json
```

## Revision 2 replay pools

`breadth_tulu3_aya.jsonl`, `breadth_tulu3_sciriff.jsonl`, `breadth_gsm8k.jsonl`,
`breadth_opencoder_edu_rev2.jsonl` (with its tests) and `breadth_swe_swiss_rev2.jsonl` (the whole
dataset) are acquired through the licensing gate (`sftgen/breadth/sources.py`) at pinned
revisions; the Gate-2 pools keep their names. Revision 2 takes its prompts from them and from the
Gate-2 pool of oasst1, and the base writes the answers (ADR-004 Revision 2). `rev2_breadth_manifest.json` (tracked) records each pool's revision,
sha256, licences and gate results, and the command that rebuilds it.

`rev2_prompts.jsonl` is the selection (`sftgen/select_prompts.py`); `rev2_prompts_manifest.json`
(tracked) records its parameters and per-pool counts. Rebuild it with:

```bash
uv run python -m dsbench.sftgen.select_prompts --battery-items data/battery/items \
    --out data/sft/rev2_prompts.jsonl --report data/sft/rev2_prompts_manifest.json
```

`rev2_tool_prompts.jsonl` holds the tool rows' prompts (`sftgen/tool_rows.py`: a tool list, a
request, and the check the reply must pass); `rev2_tool_prompts_manifest.json` (tracked) records
the parameters, the counts and the output's hash. Rebuild it with:

```bash
uv run python -m dsbench.sftgen.tool_rows generate --battery-items data/battery/items \
    --out data/sft/rev2_tool_prompts.jsonl --report data/sft/rev2_tool_prompts_manifest.json
```

## Provenance discipline (every row earns its place)

- **Target A** emits a row only when its SQL, executed on the real engine, equals the independent
  pandas truth. Teacher-free ⇒ Apache-2.0-clean by construction.
- **Target C** keeps a trajectory only when the oracle passes (correct-schema output table, every
  test row predicted, metric beats the bar) — a kept trajectory *is* a demonstration of finishing
  the loop. Teacher supplies phrasing; the oracle supplies correctness.
- **Decontamination** vs. dsbench (13-gram + schema-identifier + numeric-answer gate) and, for the
  Revision 2 retrain, vs. the acceptance battery (`--battery-items data/battery/items`, a
  git-ignored copy of the battery run's `items/`) runs over the
  rendered **mixture** just before training — that is the authoritative gate. The C slice here was
  already scanned clean (58/58) at generation time as a smoke check.

## Regeneration

Exact commands per file are in `manifest.json`. Target A needs only the local ClickHouse/DuckDB
sandbox (+ optional throwaway PG/MySQL for the 4-dialect contrast); Target C needs the teacher
served on dashi behind the `localhost:18080 → dashi:8080` tunnel. Token counts in the manifest are
`chars/3.5` estimates for sizing only — the real accounting happens at mixture assembly with the
actual tokenizer.
