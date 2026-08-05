#!/usr/bin/env bash
# chain_notam_all.sh — run the two Phase-7 runs back-to-back on dashi (Session 22).
#
# One iGPU, so the runs must be serial: phase_notam (NOTAM-only adapter, base+FT on both NOTAM
# tasks) THEN phase_all (the three-product adapter on all four evals). Chaining here (rather than
# two separate launches) avoids any chance of them overlapping on the GPU. phase_notam runs first
# because it also produces the NOTAM BASE numbers that the phase_all comparison leans on.
set -uo pipefail
cd ~/scripts/avtext
echo "=== CHAIN START $(date) ==="

echo "### RUN 1/2: phase_notam ###"
bash ./phase_notam.sh; echo "phase_notam exit=$?"

echo "### RUN 2/2: phase_all ###"
bash ./phase_all.sh;   echo "phase_all exit=$?"

echo "=== CHAIN DONE $(date) ==="
