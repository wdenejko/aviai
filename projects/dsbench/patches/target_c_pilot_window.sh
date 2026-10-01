#!/bin/bash
# GPU window for Target C's agentic pilot (ADR-004 Revision 2, action item 5). Launch detached,
# after the owner has stopped production:
#   nohup setsid ~/benchlab/scripts/target-c-pilot/window.sh </dev/null >/dev/null 2>&1 &
# The box only serves: the bare base (I-Mini, no LoRA), 8 slots, thinking on per request. The agent
# loop runs on the Mac, next to the sandbox it operates (patches/target_c_pilot_mac.sh), through an
# SSH tunnel to this server. So the window holds, as battery_window.sh's hold step does: it touches
# hold/ready, then waits for the Mac to touch hold/done, or for HOLD_MAX seconds (default 4 hours).
# Like target_a_prefill_window.sh, it refuses to start while production (any llama-server but
# OCR's) runs, and handles OCR as ~/benchlab/RULES.md says: stopped once it has finished loading,
# and restarted on exit only if it was running when the window began.
set -u
D=~/benchlab/scripts/target-c-pilot
STAMP=$(date +%Y%m%d-%H%M%S)
L=~/benchlab/logs; LOG=$L/target-c-window-$STAMP.log
URL=http://127.0.0.1:8093
SRV_PAT='^/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin/llama-server .*--port 8093'
OCR_UNIT=dashi-unlimited-ocr.service
OCR_PAT='^/home/wdenejko/src/llama-unlimited-ocr/[^ ]*/llama-server '
mkdir -p "$L" "$D/hold"
log(){ echo "[window] $(date +%H:%M:%S) $*" >>"$LOG"; }
production(){ pgrep -af '^[^ ]*/llama-server( |$)' | grep -qv 'llama-unlimited-ocr/'; }
if production; then log "a llama-server other than OCR's is running: not starting"; exit 1; fi
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
stop_server(){ pkill -f "$SRV_PAT"; for _ in $(seq 1 30); do pgrep -f "$SRV_PAT" >/dev/null || return 0; sleep 2; done; pkill -9 -f "$SRV_PAT"; }
OCR_WAS=$(systemctl --user is-active $OCR_UNIT)
restore(){ stop_server; [ -n "${GOV:-}" ] && kill "$GOV" 2>/dev/null
  if [ "$OCR_WAS" = active ]; then systemctl --user start $OCR_UNIT; sleep 2; fi
  log "OCR -> $(systemctl --user is-active $OCR_UNIT) (was $OCR_WAS)"; log "window exit"; }
trap restore EXIT
log "stopping OCR (was $OCR_WAS)"
stop_ocr
source ~/fttrain/gttwait.sh; gttwait >>"$LOG" 2>&1
# 8 slots share one KV pool. A loop resends its whole trajectory each turn, up to ~20k tokens, so
# 8 of them fit 196k.
NOLORA=1 CTX=196608 NP=8 nohup $D/battery_server.sh >>$L/target-c-server-$STAMP.log 2>&1 </dev/null &
for _ in $(seq 1 120); do curl -sf $URL/health >/dev/null && break; sleep 5; done
curl -sf $URL/health >/dev/null || { log "server never became healthy"; exit 1; }
log "server up (base, no LoRA)"
# An hour or more of generation: under the thermal governor, anchored on this server.
PATTERN="$SRV_PAT" nohup ~/fttrain/thermostat.sh >>$L/target-c-thermostat-$STAMP.log 2>&1 </dev/null &
GOV=$!
# Smoke test: the server defaults to thinking off; the request must switch it on, and the trace
# must come back in reasoning_content.
curl -s $URL/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Is 91 prime? Answer yes or no."}],"max_tokens":1024,"temperature":0.6,"top_p":0.95,"top_k":20,"chat_template_kwargs":{"enable_thinking":true}}' >$L/target-c-smoke-$STAMP.json
if ! python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["choices"][0]["message"]; sys.exit(0 if m.get("reasoning_content") else 1)' $L/target-c-smoke-$STAMP.json; then
  log "smoke test: no reasoning in the reply, thinking not engaged; stopping"; exit 1; fi
log "smoke test ok: thinking engaged"
touch $D/hold/ready
log "hold: waiting for the Mac (hold/done), at most ${HOLD_MAX:-14400}s"
for _ in $(seq 1 $(( ${HOLD_MAX:-14400} / 30 ))); do [ -f $D/hold/done ] && break; sleep 30; done
log "hold released ($( [ -f $D/hold/done ] && echo done || echo timeout))"
log "window done"
