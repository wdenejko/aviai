#!/bin/bash
# Rebuild torch-ggml-ops (branch seq4096) into the ~/ftgguf editable install: the MMQ bundle
# (152 + 38 seq-4096 keys = 190 kernels) and _C.abi3.so with the regenerated exact-key table.
# CPU only (no GPU is needed to compile). Toolbox and flags as at Gate 0: no ROCM_PATH or PATH
# override, which would poison TheRock torch, and the venv NOT activated: its bin/ puts TheRock's
# hipcc/amdclang++ first on PATH, the toolchain then pairs them with the system llvm-readobj, whose
# LLVM-style header has no 'Flags:' line, and every artifact fails the gfx1151 check. Gate 0's
# working kernels were compiled by the system ROCm 7.2.4 clang, which this reproduces. The seq-2048 build is backed up in
# ~/benchlab/scratch/torch-ggml-ops-bundle-seq2048/. Launch detached:
#   nohup setsid ~/benchlab/scripts/build_ggml_ops_seq4096.sh </dev/null >/dev/null 2>&1 &
set -u
LOG=~/benchlab/logs/build-ggml-ops-seq4096-$(date +%Y%m%d-%H%M%S).log
nice -n 10 toolbox run -c llama-rocm-unlimited-build bash -lc '
  PY=~/ftgguf/bin/python
  cd ~/src/torch-ggml-ops
  export PYTORCH_ROCM_ARCH=gfx1151 GPU_ARCHS=gfx1151
  echo "branch: $(git branch --show-current); hipcc: $(which hipcc)"
  time $PY -m pip install --no-build-isolation --no-deps -e .
  echo "kernels: $(ls torch_ggml_ops/kernels/gfx1151 | wc -l)"
  # the inventory checks of test_mmq_deployment_bundle / test_public_deployment_correctness,
  # without pytest (not in the training venv)
  $PY -c "
import torch, torch_ggml_ops
from tools.mmq_deployment_cases import operation_counts, public_deployment_cases
cases = public_deployment_cases()
print(\"import ok; cases:\", len(cases), dict(sorted(operation_counts().items())))
print(\"seq-4096 cases:\", sum(c.rows in (4096, 32768) for c in cases))
"
' >"$LOG" 2>&1
echo "[build] exit $? at $(date +%H:%M:%S)" >>"$LOG"
