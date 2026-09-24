#!/bin/bash
# Mac side of the battery's agentic-suite phase (the owner's dsbench v2 suite via the pi harness).
# Waits for the box window to reach its `hold:dsbench` step, tunnels localhost:18080 to the battery
# server (the port pi's `dashi-qwen36` provider already points at), then runs the suite for base
# and for the adapter, k=5 and thinking off, as the Gate-0 baseline did at k=5. pi can't name a
# LoRA state per request, so each pass first sets the SERVER's adapter scale and reads it back.
# Finally it converts both runs into battery scores and releases the hold.
# Run from projects/dsbench with the ClickHouse sandbox up (sandbox/README.md).
set -u
RUN_BOX=${RUN_BOX:-/home/wdenejko/benchlab/runs/2026-09-24-gate2-battery}
MIRROR=${MIRROR:?set MIRROR to the local mirror of the run dir}
URL=http://127.0.0.1:18080
until ssh dashi "test -f $RUN_BOX/hold/dsbench.ready"; do sleep 60; done
ssh -N -o ExitOnForwardFailure=yes -L 18080:127.0.0.1:8093 dashi &
TUN=$!
release(){ uv run python -m dsbench.battery.dsbench_suite set-scale --base-url $URL --scale 0
           kill $TUN; ssh dashi "touch $RUN_BOX/hold/dsbench.done"; }
trap release EXIT
sleep 5
for state in base adapter; do
  scale=0; [ $state = adapter ] && scale=1
  uv run python -m dsbench.battery.dsbench_suite set-scale --base-url $URL --scale $scale || exit 1
  uv run --package dsbench dsbench-pi-run --provider dashi-qwen36 --model qwen36 \
      --thinking off --repeat 5 --label battery-$state
  latest=$(ls -t reports/agentic-runs/*-battery-$state.json | head -1)
  uv run python -m dsbench.battery.dsbench_suite convert --pi-run "$latest" --state $state \
      --run-dir "$MIRROR"
done
