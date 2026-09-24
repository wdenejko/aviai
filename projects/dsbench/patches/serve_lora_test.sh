#!/bin/bash
# Production llama-server (build-v2) + the Gate-2 LoRA, loopback only, on a spare port.
export PATH=$PATH:/usr/sbin:/sbin
B=~/src/llama-qwen4exp-src/build-v2/bin
exec toolbox run --container llama-vulkan-wdenejko env LD_LIBRARY_PATH=$B $B/llama-server \
  -m ~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf \
  --lora ~/gate2/lora-gguf/gate2-correct-f32.gguf \
  --alias qwen36-gate2 --host 127.0.0.1 --port 8093 \
  -ngl 99 -c 8192 -ub 512 --parallel 1 --flash-attn on --metrics
