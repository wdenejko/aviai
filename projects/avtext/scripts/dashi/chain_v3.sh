#!/usr/bin/env bash
# chain_v3.sh — corrected Phase-7 run with chunked NOTAM eval (Session 22).
# single-task NOTAM (chunked) -> base NOTAM (chunked) -> all-products (train + eval all four).
set -uo pipefail
cd ~/scripts/avtext
echo "=== CHAIN_V3 START $(date) ==="
echo "### 1/3: single-task NOTAM (chunked ext + full cls) ###"
bash ./eval_notam_chunked.sh gemma-4-e4b-notam-r16 ~/ft/gemma-4/gemma-4-e4b-notam-r16-f16.gguf; echo "single exit=$?"
echo "### 2/3: base NOTAM (chunked ext + full cls) ###"
EXTTOK=256 bash ./eval_notam_chunked.sh gemma-4-e4b-notam-base; echo "base exit=$?"
echo "### 3/3: all-products (train + eval all four) ###"
bash ./phase_all_v2.sh; echo "phase_all_v2 exit=$?"
echo "=== CHAIN_V3 DONE $(date) ==="
