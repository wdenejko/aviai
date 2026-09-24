#!/bin/bash
# Gate-2 fixed-M loss eval: base vs Gate-1 (masked) vs Gate-2 adapter, one model load, same blocks.
# Follows gate1_ab.sh (OCR user unit stopped for the window, restarted on exit). Launch detached.
set -u
G=~/gate2; LOG=$G/ab.log; : > "$LOG"
R=~/src/transformers5-qwen3.5-recipe
restore(){ systemctl --user start dashi-unlimited-ocr.service 2>>"$LOG"; sleep 2
           echo "[window] $(date +%H:%M:%S) OCR -> $(systemctl --user is-active dashi-unlimited-ocr.service)" >>"$LOG"; }
trap restore EXIT
echo "[window] $(date +%H:%M:%S) stopping OCR" >>"$LOG"
systemctl --user stop dashi-unlimited-ocr.service 2>>"$LOG"; sleep 3
source ~/fttrain/gttwait.sh; gttwait
toolbox run -c llama-rocm-unlimited-build bash -lc "
  source ~/ftgguf/bin/activate
  export PYTHONPATH=\$HOME/src/aiter PYTORCH_ROCM_ARCH=gfx1151 GPU_ARCHS=gfx1151 PYTHONUNBUFFERED=1 \
         QWEN35_ATTN_IMPL=kernels-community/aiter-flash-attn \
         ADAPTERS='gate1=$R/out_qwen36_35b/final,gate2=$G/out_qwen36_35b_gate2/final' \
         EVAL_BUCKETS=$G/eval_buckets_gate2.jsonl EVAL_OUT=$G/gate2_ab_result.json \
         TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 HIP_FORCE_DEV_KERNARG=1 PYTORCH_ALLOC_CONF=expandable_segments:True
  python ~/eval_adapter_ab.py
" </dev/null 2>&1 | grep --line-buffered -vE "Autotuning kernel|num_warps:|Triton autotuning|with key as|finished after|best config|MIOpen|UserWarning|warnings.warn|causal_conv1d|kernel mapping found|Loading weights|Fetching|recompiles|guards.py" >>"$LOG"
echo "[window] $(date +%H:%M:%S) A/B done" >>"$LOG"
