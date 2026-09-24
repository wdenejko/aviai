#!/bin/bash
# Isolated V-permutation test: GDN-only adapters, correct vs no_vperm, on the same three texts.
export PATH=$PATH:/usr/sbin:/sbin
B=~/src/llama-qwen4exp-src/build-v2-85cc-bak/bin
M=~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf
L=~/gate2/lora-gguf; P=~/gate2/ppl; OUT=$P/gdn_ablation.tsv
printf "text\tstate\tppl\tpm\n" > $OUT
for text in targetA_heldout targetC_heldout general_heldout; do
  for state in gdn-correct gdn-no_vperm; do
    log=$P/$text.$state.log
    toolbox run --container llama-vulkan-wdenejko env LD_LIBRARY_PATH=$B $B/llama-perplexity \
      -m $M --lora $L/$state-f32.gguf -f $P/$text.txt -c 2048 -b 2048 -ub 512 -ngl 99 </dev/null >$log 2>&1
    est=$(grep -aoE "PPL = [0-9.]+ \+/- [0-9.]+" $log | tail -1)
    printf "%s\t%s\t%s\t%s\n" "$text" "$state" "$(echo "$est" | awk '{print $3}')" "$(echo "$est" | awk '{print $5}')" >> $OUT
  done
done
echo DONE >> $OUT
