#!/bin/bash
# One GPU window of Revision 2's single-turn generation (ADR-004 Revision 2, action items 3, 4 and
# 6). Launch detached, once the owner has agreed to the window:
#   ARM=1 MAX_HOURS=9 nohup setsid ~/benchlab/scripts/rev2-gen/window.sh RUN "tools sql code replay" \
#       </dev/null >/dev/null 2>&1 &
# For each NAME in the plan, in order, the bare base (no LoRA) answers RUN/items/NAME.jsonl into
# RUN/gen/NAME.jsonl: thinking on, one reply a prompt, through dsbench.sftgen.reasoning_pilot
# generate --block 8192, so no reply runs past what its training row can hold. Every step resumes:
# an item an earlier window answered is skipped, so a second window with the same plan continues
# the first. The replies are checked afterwards (patches/README.md, the Revision 2 runbook).
# A step splice:FROM:TO builds RUN/items/TO.jsonl from FROM's items and replies (dsbench.sftgen.
# prefill splice): Target A's recall items, written after the plain phase that they cut. It runs
# again in every window, from all the replies there are, so a resumed plain phase feeds it too.
# With LORA=adapter.gguf the server loads that adapter (battery_server.sh), and a step NAME:STATE
# answers RUN/items/NAME.jsonl into RUN/gen/NAME.STATE.jsonl with every request naming the state's
# scale: base 0, adapter 1. This is the Target A test (sftgen/target_a_eval.py), base and adapter
# paired on one server as the battery's states are. Its items are a test's, not training rows, so
# no block applies: a reply gets the item's budget (12,288), up to MAX_TOKENS (default 12,288). Run
# one state's step after the other: requests at different scales never batch together. PARITY=1
# first checks that scale 0 is the bare base on this server, as battery_window.sh does.
# As battery_window.sh does: production is the owner's to stop, so the window refuses to start
# beside it, and with ARM=1 waits for it to stop (up to DEADLINE_H hours, default 12); OCR is
# stopped once it has finished loading and restarted on exit only if it was running when the
# window began; gttwait before the server; the thermal governor anchored on it. MAX_HOURS ends the
# window: a step still running then is stopped part-way and the rest are not run.
set -u
RUN=$1; PLAN=$2
D=~/benchlab/scripts/rev2-gen
STAMP=$(date +%Y%m%d-%H%M%S)
LOG=~/benchlab/logs/rev2-gen-window-$STAMP.log
PY=~/benchlab/batteryvenv/bin/python
URL=http://127.0.0.1:8093
SRV_PAT='^/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin/llama-server .*--port 8093'
OCR_UNIT=dashi-unlimited-ocr.service
OCR_PAT='^/home/wdenejko/src/llama-unlimited-ocr/[^ ]*/llama-server '
log(){ echo "[window] $(date +%H:%M:%S) $*" >>"$LOG"; }
production(){ pgrep -af '^[^ ]*/llama-server( |$)' | grep -qv 'llama-unlimited-ocr/'; }
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
[ -n "${MAX_HOURS:-}" ] && STOP=$(( $(date +%s) + MAX_HOURS * 3600 ))
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
stop_server(){ pkill -f "$SRV_PAT"; for _ in $(seq 1 30); do pgrep -f "$SRV_PAT" >/dev/null || return 0; sleep 2; done; pkill -9 -f "$SRV_PAT"; }
OCR_WAS=$(systemctl --user is-active $OCR_UNIT)
restore(){ stop_server; [ -n "${GOV:-}" ] && kill "$GOV" 2>/dev/null
  if [ "$OCR_WAS" = active ]; then systemctl --user start $OCR_UNIT; sleep 2; fi
  log "OCR -> $(systemctl --user is-active $OCR_UNIT) (was $OCR_WAS)"; }
trap restore EXIT
mkdir -p "$RUN/gen"
log "stopping OCR (was $OCR_WAS); plan: $PLAN"
stop_ocr
source ~/fttrain/gttwait.sh; gttwait >>"$LOG" 2>&1
PATTERN="$SRV_PAT" nohup ~/fttrain/thermostat.sh >>~/benchlab/logs/rev2-gen-thermostat-$STAMP.log 2>&1 </dev/null &
GOV=$!
# 8 slots of a prompt and a reply capped at the block (8,192 + 16 tokens), or of a test item's
# short prompt and its 12,288-token budget, fit the server's default 131,072-token pool.
start_server(){  # $1 = NOLORA: 1 = the bare base; 0 = the base with LORA loaded
  if [ "$1" = 1 ]; then NOLORA=1 NP=8 nohup ~/benchlab/scripts/battery/battery_server.sh >>~/benchlab/logs/rev2-gen-server-$STAMP.log 2>&1 </dev/null &
  else NOLORA=0 LORA="$LORA" NP=8 nohup ~/benchlab/scripts/battery/battery_server.sh >>~/benchlab/logs/rev2-gen-server-$STAMP.log 2>&1 </dev/null &
  fi
  for _ in $(seq 1 120); do curl -sf $URL/health >/dev/null && return 0; sleep 5; done
  log "server never became healthy"; return 1; }
if [ -z "${LORA:-}" ]; then
  # The bare base: the rows are its own replies. (battery_server.sh would load its default
  # adapter if NOLORA weren't 1.)
  start_server 1 || exit 1
  log "server up (base, no LoRA)"
else
  [ -f "$LORA" ] || { log "no adapter at $LORA"; exit 1; }
  if [ "${PARITY:-0}" = 1 ]; then
    start_server 1 || exit 1
    (cd $D/src && PYTHONPATH=. $PY -m dsbench.battery.parity --tag nolora --out $RUN/parity.json) >>"$LOG" 2>&1
    stop_server; gttwait >>"$LOG" 2>&1
  fi
  start_server 0 || exit 1
  # A request that names no scale gets the global one, which the adapter's loading leaves at 1:
  # make it 0, so only a request that asks for the adapter gets it.
  curl -s -X POST $URL/lora-adapters -H "Content-Type: application/json" -d '[{"id":0,"scale":0.0}]' >/dev/null
  curl -s $URL/lora-adapters >>"$LOG"; echo >>"$LOG"
  log "server up (base + LoRA $LORA, global scale 0)"
  if [ "${PARITY:-0}" = 1 ]; then
    (cd $D/src && PYTHONPATH=. $PY -m dsbench.battery.parity --tag lora --steps 0,1,0 --out $RUN/parity.json \
       && PYTHONPATH=. $PY -m dsbench.battery.parity --compare --out $RUN/parity.json) >>"$LOG" 2>&1 \
       || { log "PARITY FAILED: scale 0 is not the base model on this server; stopping"; exit 1; }
    log "parity ok"
  fi
fi
for name in $PLAN; do
  if [ "${name%%:*}" = splice ]; then
    from=${name#splice:}; from=${from%%:*}; to=${name##*:}
    log "splice $from -> $to"
    (cd $D/src && PYTHONPATH=. $PY -m dsbench.sftgen.prefill splice --items $RUN/items/$from.jsonl \
       --gen $RUN/gen/$from.jsonl --out $RUN/items/$to.jsonl) >>"$LOG" 2>&1 \
       || { log "splice $from -> $to failed: the rest not run"; break; }
    continue
  fi
  left=""
  if [ -n "${STOP:-}" ]; then
    left=$(( STOP - $(date +%s) ))
    [ "$left" -gt 0 ] || { log "time limit: $name and the rest not run"; break; }
  fi
  if [ "$name" != "${name%%:*}" ]; then  # NAME:STATE, the Target A test
    items=${name%%:*}; state=${name#*:}
    case $state in base) scale=0 ;; adapter) scale=1 ;; *) log "unknown state in $name: the rest not run"; break ;; esac
    [ -n "${LORA:-}" ] || { log "$name needs LORA: the rest not run"; break; }
    log "generate $items as $state (LoRA scale $scale)"
    (cd $D/src && PYTHONPATH=. ${left:+timeout $left} $PY -m dsbench.sftgen.reasoning_pilot generate \
       --items $RUN/items/$items.jsonl --out $RUN/gen/$items.$state.jsonl --lora-scale $scale \
       --max-tokens ${MAX_TOKENS:-12288} --workers 8) >>"$LOG" 2>&1
  else
    log "generate $name"
    (cd $D/src && PYTHONPATH=. ${left:+timeout $left} $PY -m dsbench.sftgen.reasoning_pilot generate \
       --items $RUN/items/$name.jsonl --out $RUN/gen/$name.jsonl --block 8192 --workers 8) >>"$LOG" 2>&1
  fi
  rc=$?
  [ $rc = 124 ] && { log "time limit: $name stopped part-way, the rest not run"; break; }
  [ $rc = 0 ] || log "generate $name exited $rc"
done
log "plan done"
