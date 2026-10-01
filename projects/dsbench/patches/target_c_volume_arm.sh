#!/bin/bash
# Arms Target C's volume run: starts its GPU window (target_c_volume_window.sh, deployed as
# window.sh) once the owner has stopped production, and exits. The owner stops production; this
# only waits for it. OCR doesn't hold the window back: the window stops and restores it.
# Start the Mac side (target_c_volume_mac.sh) first: a window nobody uses gives up after 20
# minutes, with production already down.
# Launch detached; HOLD_MAX (seconds, default 5 hours) passes through to the window:
#   nohup setsid ~/benchlab/scripts/target-c-volume/arm.sh </dev/null >/dev/null 2>&1 &
#   HOLD_MAX=10800 nohup setsid ~/benchlab/scripts/target-c-volume/arm.sh </dev/null >/dev/null 2>&1 &
# Cancel: pkill -f target-c-volume/arm.sh
# Gives up after DEADLINE_H hours (default 12) without starting anything.
set -u
D=~/benchlab/scripts/target-c-volume
L=~/benchlab/logs; LOG=$L/target-c-volume-arm-$(date +%Y%m%d-%H%M%S).log
mkdir -p "$L"
log(){ echo "[arm] $(date +%H:%M:%S) $*" >>"$LOG"; }
idle(){ ! pgrep -af '^[^ ]*/llama-server( |$)' | grep -qv 'llama-unlimited-ocr/'; }
end=$(( $(date +%s) + ${DEADLINE_H:-12} * 3600 ))
log "armed: waiting for production to be stopped (hold ${HOLD_MAX:-18000}s)"
while [ "$(date +%s)" -lt "$end" ]; do
  if idle; then
    sleep 60  # a restart (a supervisor, a config change) brings it back within a minute
    if idle; then log "production is down: starting the window"; exec "$D/window.sh"; fi
  fi
  sleep 30
done
log "deadline passed: the window was not started"
