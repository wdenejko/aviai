#!/bin/bash
# Mac side of Revision 2 against Revision 2.1 in one window (ADR-004, decision 10). Each adapter
# was tested in its own window, and between windows the base's own dsbench results move by up to
# 3 runs in 5 on a problem (reports/gate-evals/20261009-rev2-1-probe-dsbench.md). So both adapters
# go on one server and through pi in one window, on the problems that moved, with more runs.
#
# The box holds a battery window whose server loads both adapters: battery_window.sh RUN
# "hold:h2h" with LORA = Revision 2.1 (adapter id 0) and LORA2 = Revision 2 (id 1), and PARITY=1,
# which checks each adapter. This script waits for RUN/hold/h2h.ready, tunnels localhost:18080
# (pi's dashi-qwen36 provider) to the server, and checks that the server loaded Revision 2.1 and
# then Revision 2 (REV21, REV2). The scales name adapters by load order, so a swap of LORA and LORA2
# would swap the states' names. Then it runs PLAN, a list of states, each one block of dsbench
# through pi: the problems IDS, REPEAT runs each, thinking on.
#
# pi can't name an adapter per request, so before each block the script sets the server's scales
# and reads them back (dsbench_suite set-scale --scales): rev21 = 1,0, rev2 = 0,1, base = 0,0.
# set-scale also empties every slot's prompt cache. The server keeps no other copy
# (battery_server.sh: --cache-ram 0), so a block computes everything at its own scales from its
# first token. Without the erase, a block could start from a prefix the other adapter computed.
# The default PLAN alternates the adapters (Revision 2 first; Revision 2.1 twice; Revision 2
# twice; Revision 2.1), three blocks of k = 5 each, so a drift during the window (heat, the box's
# other load) falls on both about alike. Each block writes
# reports/agentic-runs/<stamp>-<LABEL>-<state>-b<n>.{json,md}. agentic/compare_runs.py then counts
# one state's blocks together (k = 15):
#   uv run python -m dsbench.agentic.compare_runs --base <the rev2 blocks> \
#       --adapter <the rev21 blocks> --out <out>.json
#
# On the way out, however it ends, the script puts both scales back to 0, closes the tunnel and
# releases the hold, which ends the window. To stop early, kill this script. A drop in the tunnel
# costs the runs it interrupts (pi's model_error), not the window: the tunnel restarts itself.
#
# From projects/dsbench, with the sandbox up and the Mac kept awake:
#   docker compose -f sandbox/docker-compose.yml up -d --no-build clickhouse workspace
#   RUN_BOX=<the box run directory> nohup caffeinate -is patches/rev2_h2h_mac.sh \
#       >>reports/agentic-runs/rev2-h2h-mac.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
RUN_BOX=${RUN_BOX:?set RUN_BOX to the box run directory}
PLAN=${PLAN:-rev2 rev21 rev21 rev2 rev2 rev21}
# The five problems whose runs moved between the two adapters' windows (decision 10).
IDS=${IDS:-da_cancel_dow,da_weekend_delay,de_carrier_ontime,da_utc_peak_hour,da_all_flights_avg_delay}
REPEAT=${REPEAT:-5}
WAIT_H=${WAIT_H:-12}
LABEL=${LABEL:-h2h}
REV21=${REV21:-/home/wdenejko/benchlab/runs/2026-10-08-qwen36-rev2-1-gate-final/lora.gguf}
REV2=${REV2:-/home/wdenejko/benchlab/runs/2026-10-06-qwen36-rev2-gate-final/lora.gguf}
URL=http://127.0.0.1:18080
TUNNEL='-L 18080:127.0.0.1:8093'
log(){ echo "[mac] $(date +%H:%M:%S) $*"; }
# A state's scales, by adapter id: Revision 2.1 is id 0 (LORA), Revision 2 id 1 (LORA2).
scales(){ case $1 in rev21) echo 1,0 ;; rev2) echo 0,1 ;; base) echo 0,0 ;; *) return 1 ;; esac; }

# Before waiting: a run is comparable only at the cap the plan assumes, the tunnel needs its port,
# and the agent's tools need the sandbox (as patches/rev2_probe_mac.sh).
cap=$(uv run --package dsbench python -c \
  "from dsbench.agentic.pi_runner import pi_reply_cap; print(pi_reply_cap('dashi-qwen36', 'qwen36'))")
[ "$cap" = 12288 ] || { log "pi's reply cap for dashi-qwen36 is $cap, not 12288: not starting"; exit 1; }
if lsof -nP -iTCP:18080 -sTCP:LISTEN >/dev/null 2>&1; then log "port 18080 is in use: not starting"; exit 1; fi
for c in avbench-clickhouse avbench-workspace; do
  docker ps --format '{{.Names}}' | grep -qx $c || { log "the sandbox's $c is not running"; exit 1; }
done
for state in $PLAN; do
  scales "$state" >/dev/null || { log "unknown state $state: not starting"; exit 1; }
done

log "plan: $PLAN; problems $IDS, $REPEAT runs a block; waiting for $RUN_BOX/hold/h2h.ready"
end=$(( $(date +%s) + WAIT_H * 3600 ))
until ssh dashi "test -f $RUN_BOX/hold/h2h.ready"; do
  [ "$(date +%s)" -lt "$end" ] || { log "no hold within $WAIT_H hours"; exit 1; }
  sleep 60
done

# shellcheck disable=SC2086  # TUNNEL is meant to split into its flag and its spec
# -q: a refused forward (the server not up yet) would print a line per request into the log.
( while :; do ssh -q -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 $TUNNEL dashi; sleep 5; done ) &
TUN=$!
release(){
  if uv run python -m dsbench.battery.dsbench_suite set-scale --base-url $URL --scales 0,0 \
       >/dev/null 2>&1
  then log "scales back to 0,0"
  else log "could not set the scales back to 0,0 (a window posts 0 when its server starts)"
  fi
  kill $TUN 2>/dev/null; pkill -f -- "$TUNNEL dashi" 2>/dev/null
  if ssh dashi "touch $RUN_BOX/hold/h2h.done"; then log "hold released"; else log "COULD NOT RELEASE THE HOLD"; fi; }
trap release EXIT
for _ in $(seq 1 60); do curl -sf $URL/health >/dev/null && break; sleep 2; done
curl -sf $URL/health >/dev/null || { log "no server through the tunnel"; exit 1; }
loaded=$(curl -sf $URL/lora-adapters | uv run python -c \
  'import json, sys; print(" ".join(a["path"] for a in json.load(sys.stdin)))')
[ "$loaded" = "$REV21 $REV2" ] \
  || { log "the server loaded [$loaded], not Revision 2.1 then Revision 2: not run"; exit 1; }
log "adapters: 0 = $REV21, 1 = $REV2"

n=0
for state in $PLAN; do
  n=$((n + 1)); s=$(scales "$state")
  uv run python -m dsbench.battery.dsbench_suite set-scale --base-url $URL --scales "$s" >/dev/null \
    || { log "could not set the scales to $s and empty the slots: the rest not run"; exit 1; }
  log "block $n: $state (scales $s, slots emptied)"
  uv run --package dsbench dsbench-pi-run --provider dashi-qwen36 --model qwen36 \
    --thinking high --repeat "$REPEAT" --ids "$IDS" --label "$LABEL-$state-b$n"
  log "block $n: $state finished (exit $?)"
done
log "plan done"
