#!/bin/bash
# GPU window for the packed-rows check (ADR-004 Revision 2, action item 1): one process,
# check_packed_rows.py, about half an hour (the model loads once). Deploy it to
# ~/benchlab/scripts/packing/ with check_packed_rows.py, packed_rows.py and
# src/dsbench/sftgen/tokenize_masked.py, then launch detached:
#   nohup setsid ~/benchlab/scripts/packing/window.sh </dev/null >/dev/null 2>&1 &
# It waits up to DEADLINE_H hours (default 12) for the owner to stop production, and never stops
# production itself. Then, as in the other windows, OCR is stopped once it has finished loading,
# and restarted on exit only if it was running when the window began.
# Cancel while it waits: pkill -f packing/window.sh
# The report: ~/benchlab/logs/packed-rows-check-<stamp>.json, written as the check goes, with the
# process's output in packed-rows-check-<stamp>.log. A process that initialized HIP exits 0 on
# this box whatever happened, so the report is the result, not the exit status.
set -u
D=~/benchlab/scripts/packing
STAMP=$(date +%Y%m%d-%H%M%S)
L=~/benchlab/logs; LOG=$L/packed-rows-window-$STAMP.log
R=~/benchlab/runs
TARGET_C=${TARGET_C:-$R/2026-10-01-qwen36-target-c-volume/mac/trajectories.jsonl}
PILOT=${PILOT:-$R/2026-09-29-qwen36-thinking-packing/pilot_records.jsonl}
OCR_UNIT=dashi-unlimited-ocr.service
OCR_PAT='^/home/wdenejko/src/llama-unlimited-ocr/[^ ]*/llama-server '
mkdir -p "$L"
log(){ echo "[window] $(date +%H:%M:%S) $*" >>"$LOG"; }
production(){ pgrep -af '^[^ ]*/llama-server( |$)' | grep -qv 'llama-unlimited-ocr/'; }
end=$(( $(date +%s) + ${DEADLINE_H:-12} * 3600 ))
log "armed: waiting for production to be stopped"
down=no
while [ "$(date +%s)" -lt "$end" ]; do
  if ! production; then
    sleep 60  # a restart (a supervisor, a config change) brings it back within a minute
    if ! production; then down=yes; break; fi
  fi
  sleep 30
done
[ $down = yes ] || { log "deadline passed: the check was not started"; exit 1; }
log "production is down: starting the window"
# Stopping OCR while it is still loading leaves its llama-server stuck inside the container,
# holding GPU memory (2026-09-25): wait for its /health first, and kill a server that outlives the
# stop.
stop_ocr(){
  if [ "$(systemctl --user is-active $OCR_UNIT)" = active ]; then
    for _ in $(seq 1 24); do curl -sf http://127.0.0.1:8144/health >/dev/null && break; sleep 5; done
  fi
  systemctl --user stop $OCR_UNIT; sleep 3
  if pgrep -f "$OCR_PAT" >/dev/null; then
    pkill -f "$OCR_PAT"; sleep 10; pkill -9 -f "$OCR_PAT"; log "killed an OCR server that outlived the stop"
  fi; }
OCR_WAS=$(systemctl --user is-active $OCR_UNIT)
restore(){ [ -n "${GOV:-}" ] && kill "$GOV" 2>/dev/null
  if [ "$OCR_WAS" = active ]; then systemctl --user start $OCR_UNIT; sleep 2; fi
  log "OCR -> $(systemctl --user is-active $OCR_UNIT) (was $OCR_WAS)"
  log "window exit"; }
trap restore EXIT
log "stopping OCR (was $OCR_WAS)"
stop_ocr
source ~/fttrain/gttwait.sh; gttwait >>"$LOG" 2>&1
# Anchored on the interpreter: a looser pattern also matches the shells that launch it.
PATTERN='^([^ ]*/)?python[0-9.]* +check_packed_rows\.py' \
  nohup ~/fttrain/thermostat.sh >>$L/packed-rows-thermostat-$STAMP.log 2>&1 </dev/null &
GOV=$!
log "check started"
toolbox run -c llama-rocm-unlimited-build bash -lc "
  source ~/ftgguf/bin/activate
  export PYTHONPATH=\$HOME/src/transformers5-qwen3.5-recipe:\$HOME/src/aiter \
         PYTORCH_ROCM_ARCH=gfx1151 GPU_ARCHS=gfx1151 \
         QWEN35_ATTN_IMPL=kernels-community/aiter-flash-attn \
         TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 HIP_FORCE_DEV_KERNARG=1 \
         PYTORCH_ALLOC_CONF=expandable_segments:True
  cd $D && python check_packed_rows.py $L/packed-rows-check-$STAMP.json $TARGET_C $PILOT
" </dev/null >>$L/packed-rows-check-$STAMP.log 2>&1
log "check exited (rc $?; the report is the result)"
gttwait >>"$LOG" 2>&1
log "window done"
