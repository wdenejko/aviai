#!/bin/bash
# One GPU window of the acceptance battery. Launch detached:
#   nohup setsid ~/benchlab/scripts/battery/battery_window.sh RUN_DIR "bench:state ..." \
#       </dev/null >/dev/null 2>&1 &
# Stops the OCR user unit for the window and restarts it on exit (as gate2_ab.sh), waits for GTT
# to drain, runs the thermal governor on the server, optionally runs the parity check first
# (PARITY=1: bare-base server, then the battery server), then runs each generation pass in order.
set -u
RUN=$1; PLAN=$2
STAMP=$(date +%Y%m%d-%H%M%S)
LOG=~/benchlab/logs/battery-window-$STAMP.log
PY=~/benchlab/batteryvenv/bin/python
SRC=~/benchlab/scripts/battery/src
URL=http://127.0.0.1:8093
SRV_PAT='^/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin/llama-server .*--port 8093'
log(){ echo "[window] $(date +%H:%M:%S) $*" >>"$LOG"; }
stop_server(){ pkill -f "$SRV_PAT"; for _ in $(seq 1 30); do pgrep -f "$SRV_PAT" >/dev/null || return 0; sleep 2; done; pkill -9 -f "$SRV_PAT"; }
start_server(){  # $1 = NOLORA value
  NOLORA=$1 nohup ~/benchlab/scripts/battery/battery_server.sh >>~/benchlab/logs/battery-server-$STAMP.log 2>&1 </dev/null &
  for _ in $(seq 1 120); do curl -sf $URL/health >/dev/null && { log "server up (NOLORA=$1)"; return 0; }; sleep 5; done
  log "server never became healthy"; return 1; }
restore(){ stop_server; [ -n "${GOV:-}" ] && kill "$GOV" 2>/dev/null
  systemctl --user start dashi-unlimited-ocr.service; sleep 2
  log "OCR -> $(systemctl --user is-active dashi-unlimited-ocr.service)"; }
trap restore EXIT
log "stopping OCR; plan: $PLAN"
systemctl --user stop dashi-unlimited-ocr.service; sleep 3
source ~/fttrain/gttwait.sh; gttwait >>"$LOG" 2>&1
PATTERN="$SRV_PAT" nohup ~/fttrain/thermostat.sh >>~/benchlab/logs/battery-thermostat-$STAMP.log 2>&1 </dev/null &
GOV=$!
if [ "${PARITY:-0}" = 1 ]; then
  start_server 1 || exit 1
  (cd $SRC && PYTHONPATH=. $PY -m dsbench.battery.parity --tag nolora --out $RUN/parity.json) >>"$LOG" 2>&1
  stop_server; gttwait >>"$LOG" 2>&1
fi
start_server 0 || exit 1
# Global default = base, so a client that names no state (pi) can't get the adapter by accident.
curl -s -X POST $URL/lora-adapters -H "Content-Type: application/json" -d '[{"id":0,"scale":0.0}]' >/dev/null
curl -s $URL/lora-adapters >>"$LOG"; echo >>"$LOG"
if [ "${PARITY:-0}" = 1 ]; then
  (cd $SRC && PYTHONPATH=. $PY -m dsbench.battery.parity --tag lora --steps 0,1,0 --out $RUN/parity.json \
     && PYTHONPATH=. $PY -m dsbench.battery.parity --compare --out $RUN/parity.json) >>"$LOG" 2>&1 \
     || { log "PARITY FAILED: scale 0 is not the base model on this server; stopping"; exit 1; }
  log "parity ok"
fi
for step in $PLAN; do
  bench=${step%%:*}; state=${step##*:}
  if [ "$bench" = hold ]; then
    # hold:NAME keeps the server up for a client that runs elsewhere (the Mac-side pi harness):
    # signal RUN/hold/NAME.ready, wait for RUN/hold/NAME.done, give up after HOLD_MAX seconds.
    mkdir -p $RUN/hold; touch $RUN/hold/$state.ready; log "hold $state: waiting for $state.done"
    for _ in $(seq 1 $(( ${HOLD_MAX:-36000} / 30 ))); do [ -f $RUN/hold/$state.done ] && break; sleep 30; done
    log "hold $state released ($( [ -f $RUN/hold/$state.done ] && echo done || echo timeout))"
    rm -f $RUN/hold/$state.ready; continue
  fi
  log "generate $bench $state"
  (cd $SRC && PYTHONPATH=. $PY -m dsbench.battery.generate --items $RUN/items/$bench.jsonl \
     --state $state --out $RUN/gen/$bench.$state.jsonl --workers 8) >>"$LOG" 2>&1
done
log "plan done"
