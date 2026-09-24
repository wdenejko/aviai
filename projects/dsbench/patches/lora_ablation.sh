#!/bin/bash
# Ablation: a correctly exported adapter must beat deliberately broken exports on identical text.
# Uses build-v2-85cc-bak (perplexity tool and libllama built together); build-v2's perplexity binary
# predates its libllama and segfaults. OCR stays up: a 13 GB model fits beside it.
export PATH=$PATH:/usr/sbin:/sbin
B=~/src/llama-qwen4exp-src/build-v2-85cc-bak/bin
M=~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf
L=~/gate2/lora-gguf; P=~/gate2/ppl; OUT=$P/ablation.tsv
printf "text\tstate\tppl\tpm\n" > $OUT
for text in targetA_heldout targetC_heldout general_heldout; do
  for state in base correct no_vperm swap_gate_up; do
    lora=""; [ "$state" != base ] && lora="--lora $L/gate2-$state-f32.gguf"
    log=$P/$text.$state.log
    toolbox run --container llama-vulkan-wdenejko env LD_LIBRARY_PATH=$B $B/llama-perplexity \
      -m $M $lora -f $P/$text.txt -c 2048 -b 2048 -ub 512 -ngl 99 </dev/null >$log 2>&1
    est=$(grep -aoE "PPL = [0-9.]+ \+/- [0-9.]+" $log | tail -1)
    ppl=$(echo "$est" | awk '{print $3}'); pm=$(echo "$est" | awk '{print $5}')
    printf "%s\t%s\t%s\t%s\n" "$text" "$state" "${ppl:-FAIL}" "${pm:-}" >> $OUT
  done
done
echo DONE >> $OUT
