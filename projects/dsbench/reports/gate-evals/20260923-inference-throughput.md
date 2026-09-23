# Why the agentic benchmark is not runnable: inference throughput on the GGUF+LoRA stack

**Date:** 2026-09-23. Goal was to run the dsbench agentic suite on the fine-tuned adapter. The
plumbing all works; throughput does not. This records the measurements so the question is settled.

## What was built and verified working
- OpenAI-compatible server for the GGUF base + recipe LoRA (`dsbench.sftgen.serve_adapter`),
  serving base (LoRA B=0) or any adapter, with Qwen `<tool_call>` parsing into OpenAI `tool_calls`.
- Harness integration: `DSBENCH_STREAM=0` (streaming exists only because the llama.cpp fork 500s on
  malformed tool-call JSON; a server that parses its own calls has no such gate).
- ClickHouse + workspace sandbox restored, aviation data intact (607,577 flights).
- SSH tunnel for access (dashi runs firewalld; no firewall rule was added).

## Three shape-specialised tables block inference, in sequence
The recipe is specialised to the TRAINING geometry at every layer. Each fails closed:

| layer | table | blocks |
|---|---|---|
| dense projections | torch-ggml-ops MMQ bundle | `M=48`, `M=1` — any non-training row count |
| expert base | torch-ggml-ops grouped MMQ | `r=384`, `r=8` (rows = tokens x top_k) |
| expert LoRA | `moe_gmm_configs.py` AITER GMM | `M=128, N=4` — untuned prompt/decode shapes |

All three now have generic fallbacks (see
`patches/recipe-fast_lora-mmq-generic-fallback.patch` and
`patches/recipe-fast_moe_lora-inference-fallbacks.patch`). The LoRA-GMM fallback deliberately engages
only when autograd is off: a tuned-config miss during TRAINING is a real bug, not something to hide.

**Result: inference now WORKS at any prompt length and any context length.** Before this, generation
crashed outright, and the fixed-window workaround capped the whole conversation at 2048 tokens
(only 30% of agentic conversations fit).

## But throughput did not improve

| approach | s/token |
|---|---|
| fixed 2048-token window (MMQ, padded) | 1.81 |
| KV-cached decode + generic dequant | **1.62** |

Only 10% apart, because **both cost roughly one full pass over the model's weights per generated
token**:
- fixed-window runs MMQ over 2048 padded rows to produce 1 useful token;
- cached decode avoids the padding but the generic path must dequantize the whole weight to
  multiply a single row — which is exactly the work MMQ exists to avoid.

This is a floor, not a tuning problem. Decoding faster requires **M=1 MMQ kernels that operate
directly on packed weights** (a GEMV-style kernel family). The bundle's kernels are WMMA-tiled
(`MIWaveTile [4,4]`, `WorkGroup [32,4,1]`); M=1 is not a catalog entry, it is a new kernel family.

## Cost at 1.62 s/token
| scope | runtime (base + adapter) |
|---|---|
| 7-problem subset, k=1 | **6.3 h** |
| full 23 problems, k=1 | **20.7 h** |

Measured evidence: a single `da_hub_delay` turn exceeded the harness's 3x300s budget
(`ReadTimeout` at 912s, 0 tool calls).

## Conclusion
The agentic benchmark is **possible but not economical** on this stack. Options, in order of value:
1. Write M=1 packed GEMV kernels — the real fix, a kernel-authoring project.
2. Run the 7-problem subset overnight (~6.3h) accepting a length-biased k=1 result.
3. Keep assistant-only loss as the Gate-1/Gate-2 measurement and revisit when a faster runtime exists.
