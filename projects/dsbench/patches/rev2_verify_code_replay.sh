#!/bin/bash
# The box's checks after Revision 2's generation (patches/README.md, "The checks"): code and replay
# through dsbench.sftgen.replay_verify. Code's programs run in the battery's podman sandbox (no
# network, read-only root, no capabilities); replay's checks run no code.
#   nohup setsid ~/benchlab/scripts/rev2-gen/verify_code_replay.sh RUN >LOG 2>&1 </dev/null &
set -u
RUN=$1
cd ~/benchlab/scripts/rev2-gen/src || exit 1
for step in code replay; do
  echo "[verify] $(date +%H:%M:%S) $step"
  PYTHONPATH=. ~/benchlab/batteryvenv/bin/python -m dsbench.sftgen.replay_verify \
    --items "$RUN/items/$step.jsonl" --gen "$RUN/gen/$step.jsonl" \
    --out "$RUN/verified/$step.jsonl" --run-dir "$RUN"
  echo "[verify] $(date +%H:%M:%S) $step exited $?"
done
