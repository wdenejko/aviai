#!/usr/bin/env bash
# run_resilient.sh <trainer args...> — loop train_lora_peft.py with --resume so an intermittent
# gfx1151 GPU fault costs at most --save-steps steps, not the night. Pass --save-steps N in the args.
# Run inside the ROCm toolbox with CC/CXX exported.
#
# gttwait: on this APU, GPU memory of a just-exited (or page-faulted, still in the coredump handler)
# process lingers in the amdgpu GTT pool for tens of seconds. Launching the next GPU process too
# soon starts it in a nearly-full pool and it OOMs on its first big allocation (S24: this
# invalidated a whole experiment). So wait for the pool to drain before every launch.
set -uo pipefail
GTT=/sys/class/drm/card0/device/mem_info_gtt_used
gttwait () { for i in $(seq 1 60); do [ "$(cat $GTT 2>/dev/null || echo 0)" -lt 12884901888 ] && return 0; sleep 5; done; echo "  (gtt did not drain in 5 min; continuing)"; }
MAX=${MAX_ATTEMPTS:-8}
for i in $(seq 1 "$MAX"); do
  gttwait
  echo "=== attempt $i/$MAX $(date) | gtt_used $(( $(cat $GTT) / 1073741824 )) GiB ==="
  if "$HOME/fttorch/bin/python" "$HOME/scripts/avtext/train_lora_peft.py" "$@" --resume; then
    echo "=== DONE $(date) ==="; exit 0
  fi
  echo "=== attempt $i failed — waiting for GPU memory to drain, then retrying ==="; sleep 20
done
echo "=== GAVE UP after $MAX attempts ==="; exit 1
