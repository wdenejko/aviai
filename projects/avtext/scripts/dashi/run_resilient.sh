#!/usr/bin/env bash
# run_resilient.sh <trainer args...> — loop train_lora_peft.py with --resume so an intermittent
# gfx1151 GPU fault (AOTriton attention page fault, S24) costs at most --save-steps steps, not the
# night. Pass --save-steps N in the args. Run inside the ROCm toolbox with CC/CXX exported.
set -uo pipefail
MAX=${MAX_ATTEMPTS:-8}
for i in $(seq 1 "$MAX"); do
  echo "=== attempt $i/$MAX $(date) ==="
  if "$HOME/fttorch/bin/python" "$HOME/scripts/avtext/train_lora_peft.py" "$@" --resume; then
    echo "=== DONE $(date) ==="; exit 0
  fi
  echo "=== attempt $i failed — retrying in 30s ==="; sleep 30
done
echo "=== GAVE UP after $MAX attempts ==="; exit 1
