#!/bin/bash
# Mac side of Target C's agentic pilot (ADR-004 Revision 2, action item 5). Waits for the box
# window (target_c_pilot_window.sh) to reach its hold, tunnels localhost:18080 to the box's
# server, and runs the base as the agent on the Target C tasks: 7 tasks x 3 datasets (run indices
# 101-103, each passed by the oracle without a model), thinking on, 8 loops at once. Then it
# releases the hold, which ends the window.
# Run from projects/dsbench, with the sandbox's ClickHouse and workspace up, and keep the Mac
# awake:
#   docker compose -f sandbox/docker-compose.yml up -d --no-build clickhouse workspace
#   OUT=<dir> nohup caffeinate -is patches/target_c_pilot_mac.sh >/dev/null 2>&1 &
set -u
D=/home/wdenejko/benchlab/scripts/target-c-pilot
OUT=${OUT:?set OUT to a local output directory}
URL=http://127.0.0.1:18080
mkdir -p "$OUT"
log(){ echo "[mac] $(date +%H:%M:%S) $*" >>"$OUT/mac.log"; }
log "waiting for the box window's hold"
ready(){ ssh -o BatchMode=yes -o ConnectTimeout=10 dashi "test -f $D/hold/ready" 2>/dev/null; }
until ready; do sleep 60; done
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L 18080:127.0.0.1:8093 dashi &
TUN=$!
release(){ kill $TUN 2>/dev/null; ssh -o BatchMode=yes dashi "touch $D/hold/done"
           log "hold released"; }
trap release EXIT
sleep 5
curl -sf $URL/health >/dev/null || { log "no server through the tunnel"; exit 1; }
log "tunnel up: generating"
uv run python -m dsbench.sftgen.ml_delivery_trajectories --thinking --workers 8 \
    --reps 3 --run-offset 101 --max-tokens 8192 --base-url $URL/v1 --model base --verbose \
    --out "$OUT/trajectories.jsonl" --fail-out "$OUT/failed.jsonl" --report "$OUT/report.json" \
    >>"$OUT/generate.log" 2>&1
log "generate exit $?: $(tail -2 "$OUT/generate.log" | head -1)"
