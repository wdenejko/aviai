#!/bin/bash
# Mac side of Revision 2's held-out probe and dsbench (ADR-004 Revision 2, action item 10). The
# base and the final adapter go through the owner's pi harness: k = 5, thinking on (pi's `high`
# sends enable_thinking to the server), and pi's own settings for the dashi-qwen36 model
# (temperature 0, a reply cap of 12,288 tokens, checked below).
#
# The box holds a battery window: battery_window.sh RUN "hold:probe", with LORA set to the adapter
# and the global scale at 0. This script waits for RUN/hold/probe.ready, tunnels localhost:18080
# (where pi's dashi-qwen36 provider points) to the box's server, and runs each step of PLAN:
#   probe:base probe:adapter dsbench:base dsbench:adapter     (the default)
# pi can't name a LoRA state per request, so before each step the script sets the SERVER's adapter
# scale and reads it back (dsbench.battery.dsbench_suite set-scale): 0 for base, 1 for adapter. The
# probe goes first, so a window cut short still has both of its states. Each step writes
# reports/agentic-runs/<stamp>-rev2-<suite>-<state>.{json,md}.
#
# On the way out, however it ends, the script sets the scale back to 0, closes the tunnel and
# releases the hold, which ends the window. To stop early, kill this script. Killing a runner alone
# only ends that step.
#
# From projects/dsbench, with the sandbox up and the Mac kept awake:
#   docker compose -f sandbox/docker-compose.yml up -d --no-build clickhouse workspace
#   RUN_BOX=<the box run directory> nohup caffeinate -is patches/rev2_probe_mac.sh \
#       >>reports/agentic-runs/rev2-probe-mac.log 2>&1 &
# A drop in the tunnel costs the runs it interrupts (pi's model_error), not the window: the tunnel
# restarts itself.
set -u
cd "$(dirname "$0")/.." || exit 1
RUN_BOX=${RUN_BOX:?set RUN_BOX to the box run directory}
PLAN=${PLAN:-probe:base probe:adapter dsbench:base dsbench:adapter}
WAIT_H=${WAIT_H:-12}
URL=http://127.0.0.1:18080
TUNNEL='-L 18080:127.0.0.1:8093'
log(){ echo "[mac] $(date +%H:%M:%S) $*"; }

# Before waiting: a run is comparable only at the cap the plan assumes, the tunnel needs its port,
# and the agent's tools need the sandbox.
cap=$(uv run --package dsbench python -c \
  "from dsbench.agentic.pi_runner import pi_reply_cap; print(pi_reply_cap('dashi-qwen36', 'qwen36'))")
[ "$cap" = 12288 ] || { log "pi's reply cap for dashi-qwen36 is $cap, not 12288: not starting"; exit 1; }
if lsof -nP -iTCP:18080 -sTCP:LISTEN >/dev/null 2>&1; then log "port 18080 is in use: not starting"; exit 1; fi
for c in avbench-clickhouse avbench-workspace; do
  docker ps --format '{{.Names}}' | grep -qx $c || { log "the sandbox's $c is not running"; exit 1; }
done
for step in $PLAN; do
  case $step in probe:base|probe:adapter|dsbench:base|dsbench:adapter) ;;
    *) log "unknown step $step: not starting"; exit 1 ;; esac
done

log "plan: $PLAN; waiting for $RUN_BOX/hold/probe.ready"
end=$(( $(date +%s) + WAIT_H * 3600 ))
until ssh dashi "test -f $RUN_BOX/hold/probe.ready"; do
  [ "$(date +%s)" -lt "$end" ] || { log "no hold within $WAIT_H hours"; exit 1; }
  sleep 60
done

# shellcheck disable=SC2086  # TUNNEL is meant to split into its flag and its spec
# -q: a refused forward (the server not up yet) would print a line per request into the log.
( while :; do ssh -q -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 $TUNNEL dashi; sleep 5; done ) &
TUN=$!
release(){
  if uv run python -m dsbench.battery.dsbench_suite set-scale --base-url $URL --scale 0 >/dev/null 2>&1
  then log "scale back to 0"
  else log "could not set the scale back to 0 (a window posts 0 when its server starts)"
  fi
  kill $TUN 2>/dev/null; pkill -f -- "$TUNNEL dashi" 2>/dev/null
  if ssh dashi "touch $RUN_BOX/hold/probe.done"; then log "hold released"; else log "COULD NOT RELEASE THE HOLD"; fi; }
trap release EXIT
for _ in $(seq 1 60); do curl -sf $URL/health >/dev/null && break; sleep 2; done
curl -sf $URL/health >/dev/null || { log "no server through the tunnel"; exit 1; }

for step in $PLAN; do
  suite=${step%%:*}; state=${step#*:}
  scale=0; [ "$state" = adapter ] && scale=1
  uv run python -m dsbench.battery.dsbench_suite set-scale --base-url $URL --scale $scale >/dev/null \
    || { log "could not set the scale to $scale: the rest not run"; exit 1; }
  log "$suite as $state (adapter scale $scale)"
  if [ "$suite" = probe ]; then
    uv run --package dsbench python -m dsbench.sftgen.probe.runner --provider dashi-qwen36 \
      --model qwen36 --thinking high --repeat 5 --label rev2-probe-$state
  else
    uv run --package dsbench dsbench-pi-run --provider dashi-qwen36 --model qwen36 \
      --thinking high --repeat 5 --label rev2-dsbench-$state
  fi
  log "$suite as $state finished (exit $?)"
done
log "plan done"
