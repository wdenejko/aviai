#!/bin/bash
# Mac side of Target C's volume run (ADR-004 Revision 2, action item 5). It waits for the box's
# window (target_c_volume_window.sh) to reach its hold, tunnels localhost:18080 to the box's
# server, and runs the generator's quota mode: each of the 7 task families until QUOTA of its rows
# are kept (default 22, so 154 rows for the mixture's 150), the base as the agent with thinking
# on, 8 loops at once. Then it releases the hold, which ends the window.
#
# Start it BEFORE arming the box. It checks the sandbox, the tunnel's port, the box scripts and
# the tasks' oracle gate first; a window nobody uses keeps production down for 20 minutes. From
# projects/dsbench, keeping the Mac awake:
#   docker compose -f sandbox/docker-compose.yml up -d --no-build clickhouse workspace
#   nohup caffeinate -is patches/target_c_volume_mac.sh >/dev/null 2>&1 &
# Then watch OUT/mac.log; OUT/generate.log has every run.
#
# One window or two. Every pass runs with --resume on the same OUT (default
# data/sft/rev2_target_c, git-ignored, relative to projects/dsbench). A second window, started
# the same way, continues where the first stopped, past the run indices it used.
#
# Within a window, the generator stops after a run of model or harness errors: the server or the
# sandbox went away. If both answer again, another pass resumes, PASSES in all at most, while the
# hold lasts. No run starts in the last DRAIN_MIN minutes before the hold's deadline (default 30;
# the pilot's longest run took 11).
#
# To stop early: pkill -f target_c_volume_mac.sh, then pkill -f sftgen.ml_delivery_trajectories.
# The script exits once the generator has, and releases the hold on the way out. (Killing only the
# generator looks like a crash, and another pass would start.)
#
# D (where the box keeps the window's scripts and holds) is set from the environment only to
# rehearse this against a stand-in.
set -u
cd "$(dirname "$0")/.." || exit 1
D=${D:-/home/wdenejko/benchlab/scripts/target-c-volume}
OUT=${OUT:-data/sft/rev2_target_c}
QUOTA=${QUOTA:-22}
BLOCK=${BLOCK:-8192}
OFFSET=${OFFSET:-200}  # past the run indices Gate 2 (0-62) and the pilot (101-103) used
DRAIN_MIN=${DRAIN_MIN:-30}
PASSES=${PASSES:-3}
WAIT_H=${WAIT_H:-12}
URL=http://127.0.0.1:18080
mkdir -p "$OUT"
log(){ echo "[mac] $(date +%H:%M:%S) $*" >>"$OUT/mac.log"; }
box(){ ssh -o BatchMode=yes -o ConnectTimeout=10 dashi "$@"; }
# Through the tunnel, for up to 6 minutes: it reconnects within seconds, and the thermal governor
# can pause the server for a minute or two.
server_up(){ for _ in $(seq 1 24); do curl -sf -m 10 $URL/health >/dev/null && return 0; sleep 5; done
             return 1; }
sandbox_up(){ curl -sf -m 10 http://localhost:8123/ping >/dev/null &&
              docker exec avbench-workspace true 2>/dev/null; }

log "preflight: QUOTA=$QUOTA BLOCK=$BLOCK OFFSET=$OFFSET DRAIN_MIN=$DRAIN_MIN PASSES=$PASSES"
sandbox_up || { log "the sandbox is down: start ClickHouse and the workspace"; exit 1; }
if lsof -nP -iTCP:18080 -sTCP:LISTEN >/dev/null 2>&1; then
  log "localhost:18080 is taken: close the other tunnel first"; exit 1; fi
box "test -x $D/window.sh -a -x $D/arm.sh -a -x $D/hold.sh -a -x $D/battery_server.sh" ||
  { log "the window's scripts are not on the box in $D"; exit 1; }
# Setup, reference and check on each task's base dataset, and each agent login's access.
uv run python -m dsbench.sftgen.ml_tasks >>"$OUT/preflight.log" 2>&1 ||
  { log "the tasks' oracle gate failed: see preflight.log"; exit 1; }

# The hold's deadline is in the box's clock; the passes count in the Mac's.
box_now=$(box date +%s) || { log "the box doesn't answer"; exit 1; }
skew=$(( box_now - $(date +%s) ))
log "waiting for the window's hold, at most $WAIT_H hours (box clock minus Mac clock: ${skew}s)"
give_up=$(( $(date +%s) + WAIT_H * 3600 ))
while :; do
  read -r H DEADLINE STATE <<<"$(box "$D/hold.sh state" 2>/dev/null)"
  if [ "${STATE:-}" = open ] && [ "${DEADLINE:--}" != - ]; then
    DEADLINE=$(( DEADLINE - skew ))
    [ "$DEADLINE" -gt $(( $(date +%s) + (DRAIN_MIN + 10) * 60 )) ] && break
  fi
  [ "$(date +%s)" -lt "$give_up" ] || { log "no window within $WAIT_H hours: giving up"; exit 1; }
  sleep 60
done

# A tunnel that restarts itself: the generator waits out a drop (ml_delivery_trajectories._call).
TUNNEL=(ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15
        -o ServerAliveCountMax=4 -L 18080:127.0.0.1:8093 dashi)
( while :; do "${TUNNEL[@]}"; sleep 5; done ) &
TUN=$!
# The loop first, so it can't reconnect, then its ssh, found by its command line: a pkill of this
# script kills the loop too, and leaves the ssh behind.
release(){ kill "$TUN" 2>/dev/null; pkill -f "${TUNNEL[*]}" 2>/dev/null
           box "touch $H/done" && log "hold released"; }
trap release EXIT
trap 'exit 1' INT TERM HUP  # a signal runs the release above too
server_up || { log "no server through the tunnel"; exit 1; }
box "touch $H/started"
log "hold $(basename "$H"): generating, no new run after $(date -r $(( DEADLINE - DRAIN_MIN * 60 )) +%H:%M)"

pass=0
while [ "$pass" -lt "$PASSES" ]; do
  pass=$((pass + 1))
  left=$(( (DEADLINE - $(date +%s)) / 60 - DRAIN_MIN ))
  if [ "$left" -lt 10 ]; then log "pass $pass: $left minutes to start runs in, too few"; break; fi
  REPORT=$OUT/report-$(date +%Y%m%d-%H%M%S).json
  log "pass $pass: starting runs for $left minutes"
  uv run python -m dsbench.sftgen.ml_delivery_trajectories --quota "$QUOTA" --block "$BLOCK" \
      --resume --run-offset "$OFFSET" --max-minutes "$left" --thinking --max-tokens 8192 \
      --workers 8 --base-url $URL/v1 --model base --verbose \
      --out "$OUT/trajectories.jsonl" --fail-out "$OUT/failed.jsonl" --report "$REPORT" \
      >>"$OUT/generate.log" 2>&1
  rc=$?
  stop=$(uv run python -c 'import json, sys; print(json.load(open(sys.argv[1]))["stop"])' \
         "$REPORT" 2>/dev/null) || stop="no report"
  log "pass $pass: exit $rc, stopped on: $stop"
  case "$stop" in
    "quotas filled"|"time limit"|"max runs reached"*) break ;;
  esac
  if ! { server_up && sandbox_up; }; then
    log "the server or the sandbox is still down: no more passes"; break; fi
done
kept=$(uv run python -c 'import collections, json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print(len(rows), dict(sorted(collections.Counter(r["meta"]["task"] for r in rows).items())))' \
       "$OUT/trajectories.jsonl" 2>/dev/null) || kept="none"
log "rows kept in all: $kept"
