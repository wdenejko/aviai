#!/usr/bin/env bash
# chain_v2.sh — corrected Phase-7 run order on dashi (Session 22).
#
# Replaces chain_notam_all.sh after the first run exposed two eval flaws (--limit skewed NOTAM
# extraction to 4/11 categories; raw output was discarded so invalids weren't diagnosable). The
# single-task NOTAM adapter is already trained (reused), so:
#   1. eval_notam_bs.sh — base + single-task adapter on the FULL NOTAM sets (surfaces the core
#      base-vs-FT comparison first, ~half a day, no training).
#   2. phase_all.sh     — train the three-product adapter, eval it on all four evals (NOTAM full).
# Serial on the one iGPU.
set -uo pipefail
cd ~/scripts/avtext
echo "=== CHAIN_V2 START $(date) ==="

echo "### RUN 1/2: eval_notam_bs (base + single, FULL sets) ###"
bash ./eval_notam_bs.sh; echo "eval_notam_bs exit=$?"

echo "### RUN 2/2: phase_all (train all-products + eval all four) ###"
bash ./phase_all.sh;      echo "phase_all exit=$?"

echo "=== CHAIN_V2 DONE $(date) ==="
