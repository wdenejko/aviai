#!/bin/bash
# MTP-acceptance leg of the battery (ADR-001 Gate 2: acceptance drop <= 5 points). Serves I-Mini +
# the Gate-2 LoRA with the Qwen3.6 MTP head as the draft (--spec-type draft-mtp, default 3 draft
# tokens), ONE slot (the single-user regime the production server runs), and measures acceptance on
# the same 200 prompts at LoRA scale 0 and 1 (dsbench.battery.mtp). OCR is stopped for the window.
# Launch detached, after the main window: nohup setsid battery_mtp.sh RUN_DIR </dev/null &
set -u
RUN=$1
STAMP=$(date +%Y%m%d-%H%M%S)
LOG=~/benchlab/logs/battery-mtp-$STAMP.log
MTP=${MTP:-$HOME/models/Qwen3.6-35B-A3B-MTP-GGUF/Q8_0/mtp-Qwen3.6-35B-A3B-Q8_0.gguf}
PY=~/benchlab/batteryvenv/bin/python; SRC=~/benchlab/scripts/battery/src; URL=http://127.0.0.1:8093
SRV_PAT='^/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin/llama-server .*--port 8093'
OCR_UNIT=dashi-unlimited-ocr.service
OCR_PAT='^/home/wdenejko/src/llama-unlimited-ocr/[^ ]*/llama-server '
log(){ echo "[mtp] $(date +%H:%M:%S) $*" >>"$LOG"; }
stop_ocr(){  # as in battery_window.sh: never orphan an OCR server that is still loading
  if [ "$(systemctl --user is-active $OCR_UNIT)" = active ]; then
    for _ in $(seq 1 24); do curl -sf http://127.0.0.1:8144/health >/dev/null && break; sleep 5; done
  fi
  systemctl --user stop $OCR_UNIT; sleep 3
  if pgrep -f "$OCR_PAT" >/dev/null; then
    pkill -f "$OCR_PAT"; sleep 10; pkill -9 -f "$OCR_PAT"; log "killed an OCR server that outlived the stop"
  fi; }
stop_server(){ pkill -f "$SRV_PAT"; for _ in $(seq 1 30); do pgrep -f "$SRV_PAT" >/dev/null || return 0; sleep 2; done; pkill -9 -f "$SRV_PAT"; }
restore(){ stop_server; systemctl --user start $OCR_UNIT; sleep 2
  log "OCR -> $(systemctl --user is-active $OCR_UNIT)"; }
trap restore EXIT
log "stopping OCR"; stop_ocr
source ~/fttrain/gttwait.sh; gttwait >>"$LOG" 2>&1
NP=1 CTX=32768 EXTRA_ARGS="--spec-type draft-mtp --spec-draft-model $MTP" \
  nohup ~/benchlab/scripts/battery/battery_server.sh >>~/benchlab/logs/battery-mtp-server-$STAMP.log 2>&1 </dev/null &
for _ in $(seq 1 120); do curl -sf $URL/health >/dev/null && break; sleep 5; done
curl -sf $URL/health >/dev/null || { log "server never became healthy"; exit 1; }
log "server up with MTP draft $MTP"
(cd $SRC && PYTHONPATH=. $PY -m dsbench.battery.mtp --items-dir $RUN/items --out $RUN/mtp.json \
   --per-bench 50 --workers 1) >>"$LOG" 2>&1
log "done"
