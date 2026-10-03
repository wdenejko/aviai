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

## `recipe-seq4096-gmm-configs.patch`
The recipe's AITER GMM/PTGMM configs (`moe_gmm_configs.py`) are exact keys on the routed row
count, tuned for batch 1/4/16 at 2048 tokens (16,384 / 65,536 / 262,144 rows at top-8). A batch-1
seq-4096 step routes 32,768 rows and failed with "No tuned gfx1151 AITER GMM config". The patch
gives each 16,384-row Qwen key a 32,768-row twin with the same config (a Triton config sets speed,
not results). Batch-1 seq 8192 needs nothing: it routes 65,536 rows, a tuned key. See
`reports/gate-evals/20260928-seq4096-enablement.md`.

## `recipe-train-packed-rows.patch`
Revision 2 packs several whole rows into each 8,192-token block (`tokenize_masked.pack_thinking`).
Without boundaries, a row attends to the rows before it in its block and inherits their
GatedDeltaNet state, and the convolution mixes its first 3 tokens with the previous row's last 3.
The dataset now lists each block's segments (`seq_lens`: each row with its separator, then the
padding), and this patch hands them to the model:
- the collator pops `seq_lens` before `default_data_collator`, which can't stack lists of
  different lengths, and returns `packed_rows.add_segments(batch, seq_lens)`. That adds position
  ids that restart at each row, `cu_seq_lens_q/k` and `max_length_q/k`. A dataset without
  `seq_lens` trains as before;
- `main()` calls `configure_qwen35_packed_conv()`, which sends the convolution to FLA's
  `causal_conv1d` when the boundaries are given;
- training refuses `seq_lens` unless the attention is flash's, because the other implementations
  take no boundaries.

Copy `packing/packed_rows.py` into the recipe directory along with the patch. Batch size stays 1,
because FLA's varlen rule flattens the batch. Checked by `packing/check_collator.py` (CPU) and
`packing/check_packed_rows.py` (GPU), below.

## `recipe-train-save-limit.patch`
The recipe keeps its last 5 checkpoints (`save_total_limit=5`) and deletes the older ones as it
goes. Revision 2 trains about 1,220 steps with a checkpoint every 100, and gates checkpoints from
the whole run on the mini-battery, so the early ones would be gone before the gate could read
them. `QWEN35_SAVE_TOTAL_LIMIT` sets the limit, and 0 keeps every checkpoint (about 0.9 GB each);
without it the recipe keeps 5, as before. It applies on top of `recipe-train-packed-rows.patch`
(checked on a copy of the box's recipe, 2026-10-02).

## `rev2_train.sh`
Revision 2's training window, `gate2_train.sh` with `battery_window.sh`'s guards: it refuses to
start beside production, waits for it with `ARM=1`, and stops OCR only once it has loaded and
restores it only if it was running. Before stopping anything, it checks that the recipe carries
the packed-rows and save-limit patches and `packed_rows.py`, and that the dataset has its
`seq_lens`: without them rows would see each other in their block, silently. `MAX_HOURS` stops the
trainer from inside the container, losing the steps since its last checkpoint; `QWEN35_RESUME=1`
continues. A step limit (`QWEN35_MAX_STEPS`) would change the learning-rate schedule, which the
trainer computes from it.

    git apply -p0 recipe-train-packed-rows.patch; git apply -p0 recipe-train-save-limit.patch
    cp packing/packed_rows.py ~/src/transformers5-qwen3.5-recipe/
    ARM=1 nohup setsid ~/benchlab/scripts/rev2-train/train.sh RUN </dev/null >/dev/null 2>&1 &

The patches are applied in `~/src/transformers5-qwen3.5-recipe` when the training window is
prepared, not before: the recipe is shared with every other training run on the box.

## `torch-ggml-ops-gfx1151-build.patch` (`~/src/torch-ggml-ops`, box-local)
The two source fixes the Gate-0 build of torch-ggml-ops needed on dashi, kept as working-tree
changes there: `tools/mmq_deployment_bundle.py` imports torch before `tools.ggtensile` (TheRock's
ROCm has to load before rocisa pulls the system ROCm), and `tools/ggtensile/toolchain.py` keeps the
`amdclang++` path unresolved (resolving it reaches the `amdllvm` multicall binary, which dispatches
on its argv[0]). Build with `~/ftgguf/bin/python -m pip install --no-build-isolation --no-deps -e .`
in the `llama-rocm-unlimited-build` toolbox, without activating the venv
(`seq4096/build_ggml_ops_seq4096.sh` says why).

## `torch-ggml-ops-seq4096.patch` (`~/src/torch-ggml-ops`, branch `seq4096`)
The two commits of the box-local branch that let the MMQ bundle serve batch-1 seq-4096 training:
38 exact keys (every ordinary M=2048 key gets M=4096, every top-8 grouped R=16384 key R=32768, on
the twin's tuned kernel spec, except paired-backward Q3_K at R=32768, which failed the library's
oracle on that spec and uses the R=65536 one), the oracle's own GMM tables extended the same way,
the inventory tests' counts, and the regenerated exact-key table.

## `seq4096/` (box scripts behind the seq-4096 report)
- `add_seq4096_keys.py`, `fix_pair_bwd_q3k.py`: write the 38 catalog keys, then move the one
  failing key to spec 1.
- `build_ggml_ops_seq4096.sh`: rebuilds the bundle and the extension into `~/ftgguf`.
- `validate_seq4096_keys.py`: every new key and its seq-2048 twin against the library's external
  oracle (public and direct routes, repeatability, input dependence). Keys whose test tensors
  come from a model that is not on the box are reported as skipped.
- `make_audit_long_seq.py`: writes `audit_long_seq.py`, the recipe's training-step audit with
  sequences over 2048 (it glues 2048-token dataset rows).
- `seq4096_window.sh`: the GPU window (validation, then audits at 2048, 4096 and 8192). It refuses
  to start while any llama-server or OCR runs, and gates on the validation's JSON, because a Python
  process that initialized HIP exits 0 on this box whatever it asks for.
- `audit_summary.py`: the warmed step, memory, losses and gradient checks from audit reports.

## `thinking-packing/` (box scripts behind the thinking-on rendering report)
- `template_probe.py`: exports the training tokenizer's chat template (the test fixture
  `tests/fixtures/qwen36_chat_template.jinja` is its output, byte for byte) and renders sample
  conversations: history, tool loops, the thinking-on and thinking-off generation prompts.
- `tokenizer_probe.py`: which tokens the GGUF marks CONTROL or USER_DEFINED (llama.cpp keeps them
  whole), and how the HF tokenizer splits them before and after `match_llama_tokenization`.
- `validate_pilot_packing.py`: on the reasoning pilot's 200 traces, the prompt token counts against
  llama-server's own, the label invariants on real ids, and packing at 8192. It needs
  `tokenize_masked.py` beside it.

All three are CPU-only but run inside the `llama-rocm-unlimited-build` toolbox, because the
torch in `~/ftgguf` needs `libatomic.so.1` and the host lacks it. See
`reports/gate-evals/20260929-thinking-rendering-packing.md`.

## `packing/` (packed rows: the module the recipe patch imports, and its checks)
- `packed_rows.py`:
  - `segment_kwargs`: one block's packed-sequence arguments;
  - `add_segments`: the collator's step, checked against the block's length;
  - `configure_qwen35_packed_conv`: the convolution's boundaries.
- `check_collator.py` (CPU): runs a dataset built by `build_masked_dataset.py` through the patched
  collator, block by block. The collator is compiled from the patched recipe's source. Run
  2026-10-02 on the reasoning pilot's 199 rows: 56 blocks and 190 rows, as built.
- `check_packed_rows.py` (GPU): one block of four real rows (two Target C, two from the reasoning
  pilot) through the recipe's training model three ways: with its boundaries, without them, and
  row by row. It compares hidden states, the loss and the LoRA-B gradients. `--dry-run` stops
  before the model loads.
- `check_kernels.py` (GPU, seconds): the three kernels the boundaries switch (attention's
  variable-length path, FLA's rule with `cu_seqlens`, FLA's convolution), forward and backward,
  on random inputs at the model's shapes. It runs against the plain kernels, float32 attention,
  and each segment alone.
- Run 2026-10-02 (`reports/gate-evals/20261002-packed-rows-check.md`): packed, each row trains
  exactly as if it were alone, and no gradient crosses a boundary.
- `window.sh`: the check's window.
  - It waits up to 12 hours for the owner to stop production.
  - It stops OCR once it has loaded, and restores it as it was.
  - It runs the check under the thermostat.

  Deploy it to `~/benchlab/scripts/packing/` with `check_packed_rows.py`, `packed_rows.py` and
  `src/dsbench/sftgen/tokenize_masked.py`.

The CPU checks also run in the toolbox, for the same reason as above.

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
logs straight to `train.log` in its run directory rather than through `| tail`, which buffers until
EOF.

Paths: this script, `gate2_ab.sh` and the LoRA export scripts below ran from `~/gate2/` until the
box was reorganised on 2026-09-28, and now live in `~/benchlab/runs/2026-09-23-qwen36-gate2-train/`.
The copies here carry the new paths. The box copies keep the old ones as a record of what ran
(`~/benchlab/RULES.md`; old to new in `~/benchlab/runs/REORG-2026-09-28.tsv`), so to rerun a script,
deploy it from here first.

Launch detached, or it dies with the ssh session that started it:

    ssh dashi 'nohup setsid ~/benchlab/runs/2026-09-23-qwen36-gate2-train/gate2_train.sh >/dev/null 2>&1 </dev/null &'
    ssh dashi 'QWEN35_RESUME=1 nohup setsid ~/benchlab/runs/2026-09-23-qwen36-gate2-train/gate2_train.sh >/dev/null 2>&1 </dev/null &'

## `gate2_ab.sh`
The Gate-2 loss eval launcher: base vs Gate-1 vs Gate-2 adapters on `eval_buckets_gate2.jsonl`
(built by `dsbench.sftgen.eval_buckets_gate2`). Same shape as `gate1_ab.sh`: OCR user unit stopped
for the ~30-minute window and restarted on exit, GTT drain before touching the GPU,
`PYTHONUNBUFFERED=1` so progress is visible live. Launch detached and on its own — putting `&` at
the end of an `a && b && nohup c &` chain backgrounds the WHOLE chain, which keeps the ssh channel
open until the eval ends:

    ssh dashi 'nohup setsid ~/benchlab/runs/2026-09-23-qwen36-gate2-train/gate2_ab.sh >/dev/null 2>&1 </dev/null &'

## LoRA export verification scripts
Box-side scripts behind `reports/gate-evals/20260924-gguf-export.md`:

- `render_ppl_text.py` — held-out eval records as plain role-labelled text for `llama-perplexity`
  (it tokenizes without parsing special tokens, so chat markup would be split into characters).
- `lora_ablation.sh` — base vs correct vs deliberately broken exports (`no_vperm`, `swap_gate_up`).
- `gdn_ablation.sh` — the isolated test: GDN-only adapters, with and without the V permutation.
- `serve_lora_test.sh` — production `llama-server` with the adapter, loopback on port 8093.
- `lora_onoff.py` — same server, adapter scale 1 vs 0 per request: speed and answers.

The ablations use `build-v2-85cc-bak`: `build-v2`'s `llama-perplexity` predates its `libllama` and
segfaults on startup.

## Acceptance battery scripts (`dsbench.battery`)
Box-side launchers behind `reports/gate-evals/20260924-gate2-battery.md`. During the battery they
lived in `~/benchlab/scripts/battery/` next to a synced copy of `src/`, with logs in
`~/benchlab/logs/`; at the end both were moved into the run directory,
`~/benchlab/runs/2026-09-24-gate2-battery/{scripts/battery,logs}/` (per `~/benchlab/RULES.md`).

- `battery_server.sh` — production `llama-server` (`build-v2`) + I-Mini + the Gate-2 LoRA. 8 slots
  on one unified KV pool, thinking off by default, `--slot-save-path` so passes can erase slot
  caches. `--lora-init-without-apply` alone leaves the global scale at 1.0, so the window posts
  scale 0 after start: a request that names no state gets the base. `NP`, `CTX`, `NOLORA` and
  `EXTRA_ARGS` override the defaults.
- `battery_window.sh` — one GPU window: OCR stopped (restarted on exit), GTT drain, the thermal
  governor on the server, the optional parity check (`PARITY=1`), then the generation passes in
  order. A `hold:NAME` step keeps the server up for a client that runs elsewhere (the Mac-side
  pi harness) until `RUN/hold/NAME.done` appears or `HOLD_MAX` seconds pass. Since 2026-10-02:
  - it refuses to start while a llama-server other than OCR's runs;
  - `ARM=1` waits up to `DEADLINE_H` hours for the owner to stop production;
  - `MAX_HOURS` ends the window. A pass still running then is stopped, and running the plan again
    resumes it, since `generate.py` writes each row in one write and skips answered items.
- `battery_queue_half.sh` — ADR step 4: queued behind the MTP window, reruns the public benchmarks
  at LoRA scale 0.5 (state `adapter_half`).
- `battery_tput_sweep.sh` — the slot-count sweep that fixed `NP=8`: 104 tok/s at 8 slots, about 40
  to 80 at 16. Vulkan's mat-vec kernels serve batches of up to 8 tokens (`mul_mat_vec_max_cols`),
  and past that MoE decode falls back to the general matmul path.
- `battery_dsbench_pi.sh` — Mac side of the `hold:dsbench` step: tunnels `localhost:18080` to the
  battery server, sets the server's adapter scale per pass and runs the agentic suite through pi.
- `fetch_mtp_src.py` + `convert_mtp.sh` — export the MTP head from 3 of the 26 HF shards with the
  fork's `--mtp` converter (index narrowed to the shards present; the full one is kept).
- `battery_mtp.sh` — the MTP-acceptance window: one slot, `--spec-type draft-mtp`, the same prompts
  at LoRA scale 0 and 1.

**The thinking-on mini-battery** (`dsbench.battery.mini`, ADR-004 Revision 2 item 8) uses the
same scripts. Its run directory holds four things:
- `items/`, from `mini prepare --items <the pinned battery items>`;
- `data`, a link to the Gate-2 run's scorer data;
- the gold scores for BIRD and HumanEval+, copied from the Gate-2 run, since their unmeasurable
  items don't change;
- the plan, base and A/A pass benchmark by benchmark, so a window cut short leaves whole pairs.

The calibration (`~/benchlab/runs/2026-10-02-qwen36-mini-battery-base/`; run 2026-10-02,
`reports/gate-evals/20261002-mini-battery-calibration.md`):

    ARM=1 CTX=196608 MAX_HOURS=9 nohup setsid ~/benchlab/scripts/battery/battery_window.sh \
      ~/benchlab/runs/2026-10-02-qwen36-mini-battery-base \
      "bfcl:base bfcl:base_rep ifeval:base ifeval:base_rep humaneval_plus:base \
       humaneval_plus:base_rep bird:base bird:base_rep" </dev/null >/dev/null 2>&1 &

The time limit stopped BIRD's A/A pass at 116 of 150 items. A second window, launched while the
first ran, finished it: armed, it waits until the first window's server is gone (any
llama-server but OCR's counts as production), then resumes the pass where it stopped.

    ARM=1 DEADLINE_H=6 CTX=196608 MAX_HOURS=3 nohup setsid \
      ~/benchlab/scripts/battery/battery_window.sh ~/benchlab/runs/2026-10-02-qwen36-mini-battery-base \
      "bird:base bird:base_rep" </dev/null >/dev/null 2>&1 &

Scoring needs no GPU: `autoscore --run-dir RUN`, beside the window, scores each pass as it
completes; `score --run-dir RUN --bench ifeval,bfcl,bird,humaneval_plus` scores every pass that
has generations. Then `mini summary --run-dir RUN --state base_rep --aa none` reads the noise, and
`mini-battery/calibration.py RUN GATE2_RUN LOG[,LOG...] OUT.json` the cost, the budget, the
thinking-off comparison and the retrospective on Gate 2's adapter.

**The full battery with thinking on** (`dsbench.battery.full`, ADR-004 decision 9) runs the same
way, on all 6,419 pinned items. The base's run directory,
`~/benchlab/runs/2026-10-02-qwen36-full-battery-thinking/`, is prepared:
- `items/` from `full --items <the pinned items> --out RUN/items`: thinking on within 12,288
  tokens, stop strings dropped, each benchmark in a seeded order;
- `data` linked and the gold scores copied from the Gate-2 run, as for the mini-battery.

One plan serves every window, since each pass resumes where the last window stopped:

    ARM=1 CTX=196608 MAX_HOURS=9 nohup setsid ~/benchlab/scripts/battery/battery_window.sh \
      ~/benchlab/runs/2026-10-02-qwen36-full-battery-thinking \
      "mmlu_pro:base:80 gpqa:base:80 lcb:base:80 ds1000:base:80 mmlu_pro:base gpqa:base \
       lcb:base ds1000:base ifeval:base bfcl:base bird:base humaneval_plus:base" \
      </dev/null >/dev/null 2>&1 &

The first four steps answer 80 random items of each benchmark the calibration didn't run, so
the first window's log gives their minutes per item before the rest is planned. `report
--run-dir RUN` compares a candidate's passes with the base's, counting a reply that never
closed as a failure and each pass only on the items it answered.

**A Revision 2 checkpoint** gets its own run directory from `mini-battery/checkpoint.sh TRAIN_RUN
STEP CAL_RUN CKPT_RUN`, with no GPU: the checkpoint's adapter exported to a LoRA GGUF, and links to
the calibration's items, data, gold scores, base and A/A passes. A battery window with
`LORA=CKPT_RUN/lora.gguf` and the plan `bfcl:adapter ifeval:adapter humaneval_plus:adapter
bird:adapter` then answers only the checkpoint's passes, and `mini summary --state adapter --aa
base_rep` compares them with the base. The script refuses a directory that exists. Rehearsed on
the Mac with a stand-in for `toolbox`. One checkpoint's passes take a window of about 5 hours.

Three box gotchas the battery hit: podman bind mounts need `:z` (SELinux is enforcing, and without
the relabel the sandbox can't read its own inputs); llama-server's slot actions (`erase`) return 501
unless `--slot-save-path` is set; and stopping the OCR unit while it is still loading orphans its
llama-server inside the toolbox container (systemd's stop times out and kills only the wrapper).
Both window launchers therefore stop OCR through `stop_ocr`, which waits for OCR's `/health` and
kills any OCR server that outlives the stop.

## ADR-004 Revision 2 pilots

Each pilot's window refuses to start while production runs, handles OCR as the battery's
launchers do, and has an arming script that starts it once the owner has stopped production.

- `target_a_hints_window.sh` + `target_a_hints_arm.sh`: Target A's hint pilot. The base answers
  the pilot's items, plain and with the convention in the prompt.
- `target_a_prefill_window.sh` + `target_a_prefill_arm.sh`: the reasoning-prefill pilot. Phase 1
  (plain and `start`), the `recall` splice on the box, then phase 2.
- `target_c_pilot_window.sh` + `target_c_pilot_arm.sh` + `target_c_pilot_mac.sh`: Target C's
  agentic pilot. The box serves the base and holds until `hold/done`. The Mac tunnels
  `localhost:18080` to it and runs the agent loop beside the sandbox.
- `target-c-pilot/measure_trajectories.py`: puts each trajectory through `thinking_record` on the
  box, giving its tokens, whether it fits 8,192 and whether it is trainable.

## Target C's volume run (ADR-004 Revision 2, action item 5)

The pilot's window, held for hours and safe to repeat. On the box, in
`~/benchlab/scripts/target-c-volume/`: `target_c_volume_window.sh` as `window.sh`,
`target_c_volume_arm.sh` as `arm.sh`, `target_c_volume_hold.sh` as `hold.sh`, and
`battery_server.sh`. On the Mac, from `projects/dsbench`: `target_c_volume_mac.sh`, which runs
the generator's quota mode (22 rows a family, 8,192-token rows) into `data/sft/rev2_target_c/`.

What changed against the pilot's scripts:
- **The hold lasts 5 hours** (`HOLD_MAX`), and its deadline goes into the hold's `ready` marker.
  The Mac starts no run in its last 30 minutes (`DRAIN_MIN`), so the last ones finish in time.
- **The hold ends early** when the Mac hasn't started 20 minutes after the server came up, when
  the server has had no request for 15 minutes since (the Mac died mid-run), or when the server
  dies: production is down only while the box works.
- **One window or two.** Each window holds in its own directory, `hold/<stamp>/` (`hold.sh`), so
  a second one can't mistake the first one's markers for its own. The Mac resumes from its
  output files, past the run indices already used.
- **A dropped tunnel doesn't end the run.** The tunnel restarts itself, the generator waits out
  a lost connection, and after a run of errors the Mac starts another pass once the server and
  the sandbox answer (`PASSES`, 3 in all).
- **The box's clock is about 2 hours behind** (found 2026-10-01: NTP off, the RTC in local
  time). The Mac measures the difference and converts the deadline into its own clock.

Deploy, then start the Mac side before arming: it checks the sandbox, the tunnel's port, the
box scripts and the tasks' oracle gate, then waits for the hold.

```bash
ssh dashi 'mkdir -p ~/benchlab/scripts/target-c-volume'
scp patches/target_c_volume_window.sh dashi:benchlab/scripts/target-c-volume/window.sh
scp patches/target_c_volume_arm.sh dashi:benchlab/scripts/target-c-volume/arm.sh
scp patches/target_c_volume_hold.sh dashi:benchlab/scripts/target-c-volume/hold.sh
scp patches/battery_server.sh dashi:benchlab/scripts/target-c-volume/battery_server.sh
docker compose -f sandbox/docker-compose.yml up -d --no-build clickhouse workspace
nohup caffeinate -is patches/target_c_volume_mac.sh >/dev/null 2>&1 &
ssh dashi 'nohup setsid ~/benchlab/scripts/target-c-volume/arm.sh </dev/null >/dev/null 2>&1 &'
```

A second window is the same two last commands. Progress is in `data/sft/rev2_target_c/mac.log`,
each run in its `generate.log`, and the box side in `~/benchlab/logs/target-c-volume-*`.

Rehearsed 2026-10-01 against a stand-in on the box (a hold made by hand and a server that answers
only `/health`, in `~/benchlab/scratch/target-c-volume-rehearsal/`): the Mac converted the
deadline across the clock difference, tunnelled, ran two passes that each stopped on 8 errors in a
row, resumed the second past every index the first used, and released the hold. Killed mid-pass,
it released the hold and closed its tunnel. `hold.sh wait` was tested on its five ways out: done,
timeout, no client, idle (a stand-in `/metrics`), and server gone.

## Revision 2's single-turn generation (ADR-004 Revision 2, action items 3, 4 and 6)

`rev2_generate_window.sh` runs on the box as `~/benchlab/scripts/rev2-gen/window.sh`, beside a
synced copy of `src/` in `~/benchlab/scripts/rev2-gen/src/`. It serves the bare base with the
battery's `battery_server.sh` (`NOLORA=1`, 8 slots), then answers each step's items through
`reasoning_pilot.py generate --block 8192`. It guards production, OCR and the thermals as
`battery_window.sh` does, with `ARM`, `DEADLINE_H` and `MAX_HOURS`.

A step `NAME` answers `RUN/items/NAME.jsonl` into `RUN/gen/NAME.jsonl`. The four steps, staged
from the Mac:

| Step | Items | From |
|---|---:|---|
| `tools` | 518 | `data/sft/rev2_tool_prompts.jsonl` |
| `sql` | 1,112 | `data/sft/rev2_sql_prompts.jsonl` |
| `code` | 961 | `data/sft/rev2_prompts.jsonl`, bucket `code` (OpenCoder, SWE-Swiss) |
| `replay` | 1,685 | `data/sft/rev2_prompts.jsonl`, bucket `replay`; rerun the selection first if decisions 5 or 6 change it |

Deploy and stage, from `projects/dsbench`:

```bash
R=/home/wdenejko/benchlab/runs/2026-10-0X-qwen36-rev2-generation
ssh dashi "mkdir -p ~/benchlab/scripts/rev2-gen $R/items"
rsync -a --exclude __pycache__ src/ dashi:benchlab/scripts/rev2-gen/src/
scp patches/rev2_generate_window.sh dashi:benchlab/scripts/rev2-gen/window.sh
scp data/sft/rev2_tool_prompts.jsonl dashi:$R/items/tools.jsonl
scp data/sft/rev2_sql_prompts.jsonl dashi:$R/items/sql.jsonl
python3 -c "import json,sys; [print(l, end='') for l in open('data/sft/rev2_prompts.jsonl') if json.loads(l)['bucket'] == sys.argv[1]]" code | ssh dashi "cat > $R/items/code.jsonl"
python3 -c "import json,sys; [print(l, end='') for l in open('data/sft/rev2_prompts.jsonl') if json.loads(l)['bucket'] == sys.argv[1]]" replay | ssh dashi "cat > $R/items/replay.jsonl"
```

Launch (the owner stops production; the window waits for it):

```bash
ssh dashi "ARM=1 MAX_HOURS=9 nohup setsid ~/benchlab/scripts/rev2-gen/window.sh $R 'tools sql code replay' </dev/null >/dev/null 2>&1 &"
```

**Target A** (once decision 2 is taken) adds four steps, about 3.5 hours. Stage
`data/sft/rev2_target_a/{ta_plain,ta_cells}.jsonl` (`target_a_hints volume-items`) as
`$R/items/ta_plain.jsonl` and `$R/items/ta_cells.jsonl`, and add to the plan:

    ta_cells ta_plain splice:ta_plain:ta_recall ta_recall

`ta_plain`'s replies stop at 512 tokens (each item's `max_tokens`). The `splice` step cuts them
and writes the convention there, as `$R/items/ta_recall.jsonl`. It runs again in every window, so
a plain phase cut short by the time limit feeds it once it is resumed.

A later window is the same command: every step skips the items already answered. In the first
minutes, the first rows show whether the budget works on the real server: `rendered_prompt_tokens`
should equal `prompt_tokens` (a tool item too, which tests that `/apply-template` renders the
tools), and `max_tokens` should be 8,208 less the prompt.

The checks, each reading one row an item:
- `tools` (anywhere): `python -m dsbench.sftgen.tool_rows verify --items ... --gen ... --out ...`;
- `sql` (the Mac, the sandbox's `sqlite` container): `python -m dsbench.sftgen.synsql verify
  --items ... --gen ... --out ...`;
- Target A (the Mac, the sandbox's four engines): `python -m dsbench.sftgen.target_a_hints
  verify --items ... --gen ... --out ...`, for `ta_cells`, and for `ta_recall` with the items the
  splice wrote (copy `$R/items/ta_recall.jsonl` back too). It counts the doubts; the assembler
  drops them by default (`--target-a-doubts`);
- `code` and `replay` (the box, the battery's podman sandbox): from
  `~/benchlab/scripts/rev2-gen/src`, `PYTHONPATH=. ~/benchlab/batteryvenv/bin/python -m
  dsbench.sftgen.replay_verify --items $R/items/code.jsonl --gen $R/gen/code.jsonl --out
  $R/verified/code.jsonl --run-dir $R`.

Rehearsed 2026-10-02 on the Mac against a stand-in server (100 items over the four steps): a pass
stopped mid-run left whole rows and resumed the rest, and every checker read its step's replies.

Then the mixture, on the Mac, from every checker's output and Target C's loops
(`sftgen/assemble_rev2.py`; ADR-004 "Assembly"):

```bash
uv run python -m dsbench.sftgen.assemble_rev2 --battery-items data/battery/items \
  --verified ITEMS_tools VERIFIED_tools --verified ITEMS_sql VERIFIED_sql \
  --verified ITEMS_code VERIFIED_code --verified ITEMS_replay VERIFIED_replay \
  --trajectories data/sft/rev2_target_c/trajectories.jsonl \
  --out data/sft/rev2_mixture.jsonl --report data/sft/rev2_mixture_manifest.json
```

and its blocks on the box, with `build_masked_dataset --records <the mixture>` (8,192 tokens,
packed rows), whose exact counts check the assembler's: its report's `estimates` compares
them with the counts each record was selected by. On Target C's rows (2026-10-02), 129 of 154
matched and 25 were over by up to 28 tokens, never under.
