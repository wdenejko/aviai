#!/bin/bash
# Acceptance-battery server: production llama-server (build-v2) + I-Mini base + the Gate-2 LoRA.
# The battery's requests name their state ("lora": [{"id": 0, "scale": 0|1}]). A request that names
# none gets the server's global scale, and --lora-init-without-apply does NOT make that 0: it only
# skips applying the adapter at startup (GET /lora-adapters still reports scale 1.0). So
# battery_window.sh posts scale 0 right after the server starts, making "no state" mean base.
# Thinking is off by default for any client (the pi harness included). NP slots share one unified
# KV pool of CTX tokens (--kv-unified), so a long request isn't capped at CTX/NP. 8 slots: the
# measured optimum (battery_tput_sweep.sh: 104 tok/s at 8, ~40-80 at 16). Vulkan's mat-vec kernels
# serve batches of <= 8 tokens (mul_mat_vec_max_cols); past that, MoE decode drops to the slower
# general matmul path.
# Loopback only. NOLORA=1 serves the bare base, for the parity check (dsbench.battery.parity), and
# loads neither adapter.
export PATH=$PATH:/usr/sbin:/sbin
B=~/src/llama-qwen4exp-src/build-v2/bin
LORA=${LORA:-$HOME/benchlab/runs/2026-09-23-qwen36-gate2-train/lora-gguf/gate2-correct-f32.gguf}
ARGS=(-m ~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf --alias qwen36-battery
      --host 127.0.0.1 --port 8093 -ngl 99 -c "${CTX:-131072}" --parallel "${NP:-8}" --kv-unified
      --flash-attn on --metrics
      --chat-template-kwargs '{"enable_thinking":false}'
      --slot-save-path /tmp/battery-slots)  # enables POST /slots/N?action=erase (generate.py)
# LORA2: a second adapter, to serve two adapters from one server (ids in load order: LORA 0, LORA2
# 1). A request's "lora" list puts every adapter it doesn't name at 0, and an adapter at 0 is left
# out of the compute graph, so each adapter is served as a server with it alone would serve it.
# --cache-ram 0: a server that serves more than one state keeps no RAM copy of prompts. The server
# loads the closest copy into a slot by tokens alone, whichever scales computed it. In every window
# checked that served two states until 2026-10-09, some requests began from the other state's
# prompt KV (parity.py's docstring). A slot's own cache stays: the server drops it when a request
# names other scales, and set-scale (dsbench_suite.py) erases every slot when it changes the global
# ones. Without the RAM copy the server also stops clearing idle slots before each request, so
# their cells stay in the pool and greedy replies vary with them; parity.py empties the server.
if [ "${NOLORA:-0}" != 1 ]; then
  ARGS+=(--lora "$LORA")
  [ -n "${LORA2:-}" ] && ARGS+=(--lora "$LORA2")
  ARGS+=(--lora-init-without-apply --cache-ram 0)
fi
# shellcheck disable=SC2206  # EXTRA_ARGS is a flag list and is meant to split on spaces
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)
mkdir -p /tmp/battery-slots  # /tmp is cleared on reboot, and llama-server refuses a missing dir
podman start llama-vulkan-wdenejko >/dev/null 2>&1 || true
exec toolbox run --container llama-vulkan-wdenejko env LD_LIBRARY_PATH=$B $B/llama-server "${ARGS[@]}"
