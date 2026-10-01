#!/bin/bash
# GPU window for Target C's volume run (ADR-004 Revision 2, action item 5). Armed by
# target_c_volume_arm.sh, or launched detached once the owner has stopped production:
#   nohup setsid ~/benchlab/scripts/target-c-volume/window.sh </dev/null >/dev/null 2>&1 &
# The pilot's window (target_c_pilot_window.sh), held longer and safe to repeat:
# - The box only serves: the bare base (I-Mini, no LoRA), 8 slots, thinking on per request. The
#   agent loops run on the Mac, next to the sandbox they operate (target_c_volume_mac.sh),
#   through an SSH tunnel to this server.
# - The hold lasts HOLD_MAX seconds, 5 hours by default: the quota run was simulated at 3.4 to
#   4.1 hours of generation. Its deadline goes into hold/<stamp>/ready, and the Mac stops
#   starting runs early enough for the last ones to finish before it.
# - The hold ends early when the Mac is done, when the Mac hasn't started NOCLIENT_MAX seconds
#   after the server came up (default 1200), when the server has had no request for IDLE_MAX
#   seconds since (default 900), or when the server is gone (hold.sh wait). The box is out of
#   production only while it works.
# - A run that doesn't fill in one window continues in a second one, armed the same way. Each
#   window holds in its own directory (hold.sh), and the Mac resumes from its output files.
# To end the hold by hand: touch ~/benchlab/scripts/target-c-volume/hold/<stamp>/done
# Production and OCR as in the other windows: it refuses to start while a llama-server other than
# OCR's runs. OCR is stopped once it has finished loading, and restarted on exit only if it was
# running when the window began.
set -u
D=~/benchlab/scripts/target-c-volume
STAMP=$(date +%Y%m%d-%H%M%S)
L=~/benchlab/logs; LOG=$L/target-c-volume-window-$STAMP.log
URL=http://127.0.0.1:8093
SRV_PAT='^/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin/llama-server .*--port 8093'
OCR_UNIT=dashi-unlimited-ocr.service
OCR_PAT='^/home/wdenejko/src/llama-unlimited-ocr/[^ ]*/llama-server '
HOLD_MAX=${HOLD_MAX:-18000}
mkdir -p "$L"
log(){ echo "[window] $(date +%H:%M:%S) $*" >>"$LOG"; }
production(){ pgrep -af '^[^ ]*/llama-server( |$)' | grep -qv 'llama-unlimited-ocr/'; }
if production; then log "a llama-server other than OCR's is running: not starting"; exit 1; fi
H=$D/hold/$STAMP
mkdir -p "$H"
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
  log "OCR -> $(systemctl --user is-active $OCR_UNIT) (was $OCR_WAS)"
  touch "$H/closed"; log "window exit"; }
trap restore EXIT
log "stopping OCR (was $OCR_WAS)"
stop_ocr
source ~/fttrain/gttwait.sh; gttwait >>"$LOG" 2>&1
# 8 slots share one KV pool. A loop resends its whole trajectory each turn, up to ~20k tokens, so
# 8 of them fit 196k.
NOLORA=1 CTX=196608 NP=8 nohup $D/battery_server.sh >>$L/target-c-volume-server-$STAMP.log 2>&1 </dev/null &
for _ in $(seq 1 120); do curl -sf $URL/health >/dev/null && break; sleep 5; done
curl -sf $URL/health >/dev/null || { log "server never became healthy"; exit 1; }
log "server up (base, no LoRA)"
# Hours of generation: under the thermal governor, anchored on this server.
PATTERN="$SRV_PAT" nohup ~/fttrain/thermostat.sh >>$L/target-c-volume-thermostat-$STAMP.log 2>&1 </dev/null &
GOV=$!
# Smoke test: the server defaults to thinking off; the request must switch it on, and the trace
# must come back in reasoning_content.
curl -s $URL/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Is 91 prime? Answer yes or no."}],"max_tokens":1024,"temperature":0.6,"top_p":0.95,"top_k":20,"chat_template_kwargs":{"enable_thinking":true}}' >$L/target-c-volume-smoke-$STAMP.json
if ! python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["choices"][0]["message"]; sys.exit(0 if m.get("reasoning_content") else 1)' $L/target-c-volume-smoke-$STAMP.json; then
  log "smoke test: no reasoning in the reply, thinking not engaged; stopping"; exit 1; fi
log "smoke test ok: thinking engaged"
echo $(( $(date +%s) + HOLD_MAX )) >"$H/ready"
log "hold $STAMP: until the Mac is done, at most ${HOLD_MAX}s"
why=$("$D/hold.sh" wait "$H" "$SRV_PAT" "$URL")
log "hold released ($why)"
log "window done"
