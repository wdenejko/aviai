#!/bin/bash
# Arms Target A's hint-conditioned pilot: starts its GPU window (target_a_hints_window.sh, deployed
# as window.sh) once the owner has stopped production, and exits. The owner stops production;
# this only waits for it. OCR doesn't hold the window back: the window stops and restores it.
# Launch detached:
#   nohup setsid ~/benchlab/scripts/target-a-hints-pilot/arm.sh </dev/null >/dev/null 2>&1 &
# Cancel: pkill -f target-a-hints-pilot/arm.sh
# Gives up after DEADLINE_H hours (default 12) without starting anything.
set -u
D=~/benchlab/scripts/target-a-hints-pilot
L=~/benchlab/logs; LOG=$L/target-a-hints-arm-$(date +%Y%m%d-%H%M%S).log
mkdir -p "$L"
log(){ echo "[arm] $(date +%H:%M:%S) $*" >>"$LOG"; }
idle(){ ! pgrep -af '^[^ ]*/llama-server( |$)' | grep -qv 'llama-unlimited-ocr/'; }
end=$(( $(date +%s) + ${DEADLINE_H:-12} * 3600 ))
log "armed: waiting for production to be stopped"
while [ "$(date +%s)" -lt "$end" ]; do
  if idle; then
    sleep 60  # a restart (a supervisor, a config change) brings it back within a minute
    if idle; then log "production is down: starting the window"; exec "$D/window.sh"; fi
  fi
  sleep 30
done
log "deadline passed: the window was not started"
