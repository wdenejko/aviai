#!/bin/bash
# Revision 2's training window (ADR-004 Revision 2, action item 9): rank 4, thinking-on rows packed
# whole into 8,192-token blocks. Launch detached, once the owner has agreed to the window:
#   ARM=1 nohup setsid ~/benchlab/scripts/rev2-train/train.sh RUN </dev/null >/dev/null 2>&1 &
# RUN holds data_tokenized/ (build_masked_dataset's blocks, with `seq_lens`). The trainer writes
# RUN/out_qwen36_35b_rev2/: a checkpoint every 100 steps, all of them kept for the checkpoint gate
# (QWEN35_SAVE_TOTAL_LIMIT=0, recipe-train-save-limit.patch), and final/. The log is RUN/train.log.
# QWEN35_RESUME=1 continues from the newest checkpoint.
#
# The recipe must carry recipe-train-packed-rows.patch and packing/packed_rows.py: without them a
# row attends to the rows before it in its block, silently. The window checks both, and that the
# dataset has its `seq_lens`, before stopping anything.
#
# As battery_window.sh does: production is the owner's to stop, so the window refuses to start
# beside it, and with ARM=1 waits for it to stop (up to DEADLINE_H hours, default 12); OCR is
# stopped once it has finished loading and restarted on exit only if it was running when the
# window began; gttwait before the GPU; the thermal governor anchored on the trainer. MAX_HOURS
# stops the trainer; the steps since its last checkpoint (at most 99) are lost, and a later window
# resumes with QWEN35_RESUME=1. MAX_HOURS, not QWEN35_MAX_STEPS: the trainer computes its learning
# rate schedule from max_steps, so a window's step limit would change the schedule.
set -u
RUN=$1
STAMP=$(date +%Y%m%d-%H%M%S)
LOG=$RUN/train.log
RECIPE=~/src/transformers5-qwen3.5-recipe
PAT='^([^ ]*/)?python[0-9.]* +train_qwen3_5_35b\.py'
OCR_UNIT=dashi-unlimited-ocr.service
OCR_PAT='^/home/wdenejko/src/llama-unlimited-ocr/[^ ]*/llama-server '
log(){ echo "[window] $(date +%H:%M:%S) $*" >>"$LOG"; }
production(){ pgrep -af '^[^ ]*/llama-server( |$)' | grep -qv 'llama-unlimited-ocr/'; }
mkdir -p "$RUN"
# The checks that need no GPU, before anything is stopped.
grep -q configure_qwen35_packed_conv $RECIPE/train_qwen3_5_35b.py && [ -f $RECIPE/packed_rows.py ] \
  || { log "the recipe lacks the packed-rows patch or packed_rows.py: not starting"; exit 1; }
grep -q QWEN35_SAVE_TOTAL_LIMIT $RECIPE/train_qwen3_5_35b.py \
  || { log "the recipe lacks the save-limit patch: early checkpoints would be deleted; not starting"; exit 1; }
toolbox run -c llama-rocm-unlimited-build bash -lc "source ~/ftgguf/bin/activate; python -c \"
from datasets import load_from_disk
d = load_from_disk('$RUN/data_tokenized')
assert 'seq_lens' in d.column_names, 'no seq_lens: rows would see each other'
print('dataset ok:', len(d), 'blocks of', len(d[0]['input_ids']), 'tokens')\"" </dev/null >>"$LOG" 2>&1 \
  || { log "the dataset check failed: not starting"; exit 1; }
if [ "${ARM:-0}" = 1 ]; then
  end=$(( $(date +%s) + ${DEADLINE_H:-12} * 3600 )); down=no
  log "armed: waiting for production to be stopped"
  while [ "$(date +%s)" -lt "$end" ]; do
    if ! production; then
      sleep 60  # a restart (a supervisor, a config change) brings it back within a minute
      if ! production; then down=yes; break; fi
    fi
    sleep 30
  done
  [ $down = yes ] || { log "deadline passed: the window was not started"; exit 1; }
fi
if production; then log "a llama-server other than OCR's is running: not starting"; exit 1; fi
LIMIT=""
[ -n "${MAX_HOURS:-}" ] && LIMIT="timeout -s TERM $(( MAX_HOURS * 3600 ))"
# Stopping OCR while it is still loading left its llama-server stuck inside the container, holding
# GPU memory (2026-09-25): wait for its /health first, and kill a server that outlives the stop.
stop_ocr(){
  if [ "$(systemctl --user is-active $OCR_UNIT)" = active ]; then
    for _ in $(seq 1 24); do curl -sf http://127.0.0.1:8144/health >/dev/null && break; sleep 5; done
  fi
  systemctl --user stop $OCR_UNIT; sleep 3
  if pgrep -f "$OCR_PAT" >/dev/null; then
    pkill -f "$OCR_PAT"; sleep 10; pkill -9 -f "$OCR_PAT"; log "killed an OCR server that outlived the stop"
  fi; }
OCR_WAS=$(systemctl --user is-active $OCR_UNIT)
restore(){ [ -n "${THERMO:-}" ] && kill "$THERMO" 2>/dev/null
  if [ "$OCR_WAS" = active ]; then systemctl --user start $OCR_UNIT; sleep 2; fi
  log "OCR -> $(systemctl --user is-active $OCR_UNIT) (was $OCR_WAS)"; }
trap restore EXIT
log "stopping OCR (was $OCR_WAS); resume=${QWEN35_RESUME:-0} max_hours=${MAX_HOURS:-none}"
stop_ocr
source ~/fttrain/gttwait.sh; gttwait >>"$LOG" 2>&1  # a just-exited GPU process can hold 80-95 GiB
PATTERN="$PAT" nohup ~/fttrain/thermostat.sh >>$RUN/thermostat-$STAMP.log 2>&1 </dev/null &
THERMO=$!
log "gtt drained; starting the trainer"
# The time limit runs inside the container, on the trainer itself: stopping the `toolbox run`
# wrapper from outside leaves the process inside running (OCR's stop, 2026-09-25).
toolbox run -c llama-rocm-unlimited-build bash -lc "
  source ~/ftgguf/bin/activate
  export PYTHONPATH=\$HOME/src/aiter PYTORCH_ROCM_ARCH=gfx1151 GPU_ARCHS=gfx1151 \
         QWEN35_ATTN_IMPL=kernels-community/aiter-flash-attn \
         QWEN35_REPORT_TO=none QWEN35_MAX_STEPS=-1 WANDB_DISABLED=true \
         QWEN35_DATASET_DIR=$RUN/data_tokenized QWEN35_OUTPUT_DIR=$RUN/out_qwen36_35b_rev2 \
         QWEN35_RESUME=${QWEN35_RESUME:-0} QWEN35_SAVE_TOTAL_LIMIT=0 \
         TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 HIP_FORCE_DEV_KERNARG=1 \
         PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
  cd $RECIPE
  $LIMIT python train_qwen3_5_35b.py
" </dev/null >>"$LOG" 2>&1
rc=$?
[ $rc = 124 ] && log "time limit: the trainer was stopped; resume with QWEN35_RESUME=1"
log "trainer exited rc=$rc"
