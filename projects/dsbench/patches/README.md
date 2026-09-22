# Patches to the upstream training recipe

These apply to `woct0rdho/transformers5-qwen3.5-recipe`, cloned on the box at
`~/src/transformers5-qwen3.5-recipe`. They are kept here so the box is reproducible; upstream is not
vendored into this repo.

## `recipe-fast_lora-mmq-generic-fallback.patch`
The `torch-ggml-ops` MMQ bundle is generated for the TRAINING geometry only (dense M in
{2048, 8192, 32768}; grouped-pair r in {16384, 65536, 262144}). Anything else raises
`unsupported exact deployment key`. That makes ordinary generation impossible: prefill presents
M = prompt length and each decode step M = 1.

Regenerating the bundle cannot fix this in general — prefill M is unbounded, so you would need a
kernel per possible prompt length. Instead this patch lets a *dense* projection fall back to the
generic compiled-dequant base forward (`base(x)`, the same path the GDN projections already use)
when a shape has no compiled kernel, caching the decision so it costs one exception per shape.

Training is unaffected: M=2048 is compiled, so it still takes the MMQ path.

NOTE: this only covers the dense path. The MoE expert path (`fast_moe_lora.py`, `grouped_mmq_pair`)
has no generic counterpart in the recipe, and swapping to a stock transformers experts
implementation would silently drop the expert LoRA. For faithful generation use the fixed-2048-token
window decoder (`dsbench.sftgen.gen_fixed_window`) instead, which keeps every op on a compiled shape.
