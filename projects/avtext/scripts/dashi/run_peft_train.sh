#!/usr/bin/env bash
# run_peft_train.sh <trainer-args...> — run train_lora_peft.py inside the ROCm toolbox, which
# supplies the system libs (libatomic) + gfx1151 device the TheRock torch wheel needs. The venv
# (~/fttorch, TheRock torch 2.12+rocm7.13) lives on the shared home; the toolbox provides the rest.
set -uo pipefail
TB=${TB:-llama-rocm-7.2.4_2}
exec toolbox run --container "$TB" bash -lc "cd ~/fttrain && ~/fttorch/bin/python ~/scripts/avtext/train_lora_peft.py $*"
