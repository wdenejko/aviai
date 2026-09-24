# Patches to the upstream training recipe

These apply to `woct0rdho/transformers5-qwen3.5-recipe`, cloned on the box at
`~/src/transformers5-qwen3.5-recipe`. They are kept here so the box is reproducible; upstream is not
vendored into this repo.

## `recipe-fast_lora-mmq-generic-fallback.patch`
The `torch-ggml-ops` MMQ bundle is generated for the TRAINING geometry only (dense M in
{2048, 8192, 32768}; grouped-pair r in {16384, 65536, 262144}). Anything else raises
`unsupported exact deployment key`. That makes ordinary generation impossible: prefill presents
M = prompt length and each decode step M = 1.

Regenerating the bundle cannot fix this in general — prefill M is unbounded, so you would need a
kernel per possible prompt length. Instead this patch lets a *dense* projection fall back to the
generic compiled-dequant base forward (`base(x)`, the same path the GDN projections already use)
when a shape has no compiled kernel, caching the decision so it costs one exception per shape.

Training is unaffected: M=2048 is compiled, so it still takes the MMQ path.

NOTE: this only covers the dense path. The MoE expert path (`fast_moe_lora.py`, `grouped_mmq_pair`)
has no generic counterpart in the recipe, and swapping to a stock transformers experts
implementation would silently drop the expert LoRA. For faithful generation use the fixed-2048-token
window decoder (`dsbench.sftgen.gen_fixed_window`) instead, which keeps every op on a compiled shape.

## `recipe-train-assistant-only-loss.patch`
The recipe's `fixed_length_lm_collator` built labels from `input_ids`, so loss covered the whole
packed block. Our records carry `loss_mask_roles: ["assistant"]`, and only 53.7% of tokens are
assistant-authored — so 46.3% of the gradient was spent predicting prompts/tool output/system text.

This patch makes the collator use dataset-provided `labels` when present (falling back to the old
behaviour when absent, so it is safe for the recipe's own datasets). Build the labelled dataset with
`dsbench.sftgen.build_masked_dataset`, which needs `remove_unused_columns=False` — already set.

Measured effect (see reports/gate-evals/20260922-gate1-masking-ab.md): assistant-only loss improves
on both targets (targetC by 20%), general capability unchanged.

## `recipe-train-overridable-paths.patch`
`dataset_dir` and `output_dir` were hardcoded to `data_tokenized_qwen3.5` and `out_qwen36_35b`, so
training a second gate would have overwritten the first gate's tokenised dataset AND its adapter —
including the checkpoints needed to compare the two. Discovered on the way into Gate 2, with the
Gate-1 adapter (`final` plus checkpoints 100/200/300/390) still sitting in that directory.

Adds `QWEN35_DATASET_DIR` and `QWEN35_OUTPUT_DIR`, matching the existing `QWEN35_*` override style.
Defaults are unchanged, so the recipe's own invocation still works.

    QWEN35_DATASET_DIR=~/data_tokenized_gate2 QWEN35_OUTPUT_DIR=~/out_qwen36_35b_gate2 \
        python train_qwen3_5_35b.py

## `recipe-train-resume.patch`
ADR-001 Gate 2 requires a resumable launcher. The recipe had the resume call commented out; this adds
`QWEN35_RESUME=1`, which continues from the newest checkpoint in `output_dir` (`save_steps=100`).
A ~9-hour run on an APU that thermally throttles after ~2 h has to survive a crash without starting
over.

## `fttrain-thermostat-pattern.patch` (box tooling, not the recipe)
`~/fttrain/thermostat.sh` is the userspace thermal governor (SIGSTOP the trainer at >= 101 °C,
SIGCONT at <= 97 °C). It hardcoded `pgrep -f "python.*train_lora_peft.py"` — the avtext/Gemma
trainer — so on a Qwen run it would never find its target and silently do nothing.

Adds a `PATTERN` override (default unchanged). The pattern for this recipe has to be anchored on
the interpreter's argv: a loose `python.*train_qwen3_5_35b.py` also matches the `toolbox run` and
`bash -lc` wrappers, whose command lines contain the same string. They start first, so `head -1`
picks a wrapper — and stopping a wrapper shell leaves Python training at full temperature.

    PATTERN='^([^ ]*/)?python[0-9.]* +train_qwen3_5_35b\.py' ~/fttrain/thermostat.sh

Verified against the real command lines before launch: matches `python ...` and
`/path/to/python3.12 ...`, rejects the `bash -lc`, `toolbox run` and `podman exec` wrappers.

## `gate2_train.sh`
The Gate-2 launcher. Follows `gate1_train.sh` (stop the OCR **user** unit for the window, restart it
on exit) and adds what a long run needs: the GTT drain gate before touching the GPU, the governor
with the anchored pattern, the Gate-2 dataset/output paths, and `QWEN35_RESUME` passthrough. It
logs straight to `~/gate2/train.log` rather than through `| tail`, which buffers until EOF.

Launch detached, or it dies with the ssh session that started it:

    ssh dashi 'nohup setsid ~/gate2/gate2_train.sh >/dev/null 2>&1 </dev/null &'
    ssh dashi 'QWEN35_RESUME=1 nohup setsid ~/gate2/gate2_train.sh >/dev/null 2>&1 </dev/null &'

## `gate2_ab.sh`
The Gate-2 loss eval launcher: base vs Gate-1 vs Gate-2 adapters on `eval_buckets_gate2.jsonl`
(built by `dsbench.sftgen.eval_buckets_gate2`). Same shape as `gate1_ab.sh`: OCR user unit stopped
for the ~30-minute window and restarted on exit, GTT drain before touching the GPU,
`PYTHONUNBUFFERED=1` so progress is visible live. Launch detached and on its own — putting `&` at
the end of an `a && b && nohup c &` chain backgrounds the WHOLE chain, which keeps the ssh channel
open until the eval ends:

    ssh dashi 'nohup setsid ~/gate2/gate2_ab.sh >/dev/null 2>&1 </dev/null &'
