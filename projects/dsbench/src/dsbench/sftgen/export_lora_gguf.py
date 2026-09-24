"""Export the recipe's PEFT LoRA adapter to a llama.cpp LoRA GGUF (arch `qwen35moe`).

Serving the adapter under llama.cpp is what unblocks ADR-001's Gate-2 acceptance battery: the
training stack generates at 1.62 s/token, llama.cpp at tens of tokens per second. Runtime LoRA
(`llama-server --lora`) is chosen over merge-and-requantize because it serves exactly the function
that was trained -- the quantized base plus the delta. Re-quantizing merged weights back to the
~3-bit APEX I-Mini recipe would add rounding noise comparable in size to a rank-4 delta.

llama.cpp's own `convert_lora_to_gguf.py` is not usable here: the expert LoRA uses this recipe's
layout (`lora_A`/`lora_B` over the fused gate+up projection, `lora_A_down`/`lora_B_down`), which
stock PEFT and the converter do not know. The mapping below was established by reading both
stacks, and each non-trivial step can be checked by an ablation (see `--variant`).

Per layer N, adapter tensor -> GGUF base tensor (`<name>.lora_a` / `<name>.lora_b` in the adapter):

    linear_attn.in_proj_qkv   -> attn_qkv        B rows 4096.. (the V block) reordered, see below
    linear_attn.in_proj_z     -> attn_gate       B rows reordered
    linear_attn.out_proj      -> ssm_out         A columns reordered
    self_attn.{q,k,v,o}_proj  -> attn_{q,k,v,output}
    mlp.experts lora_A/B      -> ffn_gate_exps + ffn_up_exps   B split, A shared
    mlp.experts lora_*_down   -> ffn_down_exps
    mlp.shared_expert.*_proj  -> ffn_{gate,up,down}_shexp

Everything else passes straight through: PEFT stores A as [r, in] and B as [out, r], which is the
orientation llama.cpp's loader validates, including the 3-D expert tensors ([E, r, in] and
[E, out, r]).

1. Value-head order (GDN layers). llama.cpp's converter reorders value heads from grouped-by-key
   to tiled order so ggml can broadcast them (`conversion/qwen.py::_reorder_v_heads`). The
   training stack UNDOES that on load (`TiledToGroupedRows`, `TiledToGroupedInputs`), so the model
   -- and every LoRA factor trained on it -- sees grouped order. For `out_proj` the base weight
   stays packed and its *input* is permuted inside `GgufLinear.forward`, but the recipe's fused
   LoRA path feeds the factors the un-permuted input, so `lora_A` there is grouped too.
   Export applies llama.cpp's forward reorder to each of the three.

2. Expert gate/up. The recipe applies one A to the fused projection and splits the output with
   `gate_delta, up_delta = gate_up_delta.chunk(2, dim=-1)`: gate is the FIRST half. llama.cpp stores
   `ffn_gate_exps` and `ffn_up_exps` separately, so B is split there and A duplicated for both.

Runs on the box, where the adapter and the `gguf` package live (the training venv has both):

    python -m dsbench.sftgen.export_lora_gguf --adapter .../final --out lora.gguf
"""

from __future__ import annotations

import argparse
import re

import numpy as np

# Qwen3.6-35B-A3B geometry (config.json): 16 key heads, 32 value heads, 128-dim heads.
NUM_K_HEADS = 16
V_HEADS_PER_K = 2
HEAD_V_DIM = 128
QK_ROWS = 2 * NUM_K_HEADS * 128  # rows of in_proj_qkv before its V block: q then k
EXPERT_FF = 512                  # moe_intermediate_size
NUM_EXPERTS = 256
LORA_ALPHA = 4.0

VARIANTS = ("correct", "no_vperm", "swap_gate_up")

_KEY = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.(.+)\."
    r"(lora_A|lora_B|lora_A_down|lora_B_down)\.weight$")

_DENSE = {
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.out_proj": "ssm_out",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.shared_expert.gate_proj": "ffn_gate_shexp",
    "mlp.shared_expert.up_proj": "ffn_up_shexp",
    "mlp.shared_expert.down_proj": "ffn_down_shexp",
}


def tiled_index(num_k: int = NUM_K_HEADS, per_k: int = V_HEADS_PER_K,
                head_dim: int = HEAD_V_DIM) -> np.ndarray:
    """Gather index taking grouped order to tiled: `tiled = grouped[index]`.

    Built exactly as llama.cpp's `_reorder_v_heads` reshapes: [key head, value-per-key, dim]
    with the first two axes swapped. The training stack's `TiledToGroupedRows` gathers with
    `argsort` of this same array, so the two are inverses.
    """
    index = np.arange(num_k * per_k * head_dim).reshape(num_k, per_k, head_dim)
    return index.swapaxes(0, 1).reshape(-1)


def grouped_to_tiled(x: np.ndarray, axis: int, offset: int = 0) -> np.ndarray:
    """Reorder `x` along `axis` from grouped to tiled, leaving the first `offset` entries alone."""
    index = np.concatenate([np.arange(offset), offset + tiled_index()])
    if index.size != x.shape[axis]:
        raise ValueError(f"axis {axis} has {x.shape[axis]} entries, expected {index.size}")
    return np.take(x, index, axis=axis)


def export_tensors(key: str, value: np.ndarray, variant: str = "correct"
                   ) -> list[tuple[str, np.ndarray]]:
    """Map one adapter tensor to the GGUF adapter tensor(s) it becomes.

    `variant` exists only to build deliberately wrong exports for the ablation: `no_vperm` skips
    the value-head reorder, `swap_gate_up` puts the gate half of B on `up` and vice versa. A correct
    export must beat both on held-out perplexity.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    match = _KEY.match(key)
    if not match:
        raise ValueError(f"unrecognised adapter tensor: {key}")
    layer, module, kind = int(match[1]), match[2], match[3]
    blk = f"blk.{layer}"

    if module == "mlp.experts":
        if value.shape[0] != NUM_EXPERTS:  # llama.cpp's loader does not check the expert axis
            raise ValueError(f"{key}: {value.shape[0]} experts, expected {NUM_EXPERTS}")
        if kind == "lora_A":
            return [(f"{blk}.ffn_gate_exps.weight.lora_a", value),
                    (f"{blk}.ffn_up_exps.weight.lora_a", value)]
        if kind == "lora_B":
            gate, up = value[:, :EXPERT_FF, :], value[:, EXPERT_FF:, :]
            if variant == "swap_gate_up":
                gate, up = up, gate
            return [(f"{blk}.ffn_gate_exps.weight.lora_b", gate),
                    (f"{blk}.ffn_up_exps.weight.lora_b", up)]
        suffix = "lora_a" if kind == "lora_A_down" else "lora_b"
        return [(f"{blk}.ffn_down_exps.weight.{suffix}", value)]

    if module not in _DENSE:
        raise ValueError(f"no GGUF mapping for module {module!r} ({key})")
    suffix = "lora_a" if kind == "lora_A" else "lora_b"
    if variant != "no_vperm":
        if module == "linear_attn.in_proj_qkv" and kind == "lora_B":
            value = grouped_to_tiled(value, axis=0, offset=QK_ROWS)
        elif module == "linear_attn.in_proj_z" and kind == "lora_B":
            value = grouped_to_tiled(value, axis=0)
        elif module == "linear_attn.out_proj" and kind == "lora_A":
            value = grouped_to_tiled(value, axis=1)
    return [(f"{blk}.{_DENSE[module]}.weight.{suffix}", value)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the recipe LoRA to a llama.cpp LoRA GGUF")
    parser.add_argument("--adapter", required=True, help="dir holding adapter_model.safetensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    parser.add_argument("--variant", choices=VARIANTS, default="correct")
    parser.add_argument("--only", default="",
                        help="regex on adapter keys: export only matching tensors (for "
                             "ablations that isolate one part of the adapter)")
    args = parser.parse_args()

    import gguf  # the box's training venv has both; kept local so tests need neither
    from safetensors.torch import load_file

    state = load_file(f"{args.adapter.rstrip('/')}/adapter_model.safetensors")
    dtype = np.float32 if args.dtype == "f32" else np.float16
    writer = gguf.GGUFWriter(args.out, arch="qwen35moe")
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, LORA_ALPHA)
    written = 0
    keys = [k for k in sorted(state) if not args.only or re.search(args.only, k)]
    if not keys:
        raise SystemExit(f"--only {args.only!r} matched no adapter tensor")
    for key in keys:
        value = state[key].float().numpy()
        for name, tensor in export_tensors(key, value, args.variant):
            writer.add_tensor(name, np.ascontiguousarray(tensor.astype(dtype)))
            written += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"{len(keys)} adapter tensors -> {written} GGUF tensors | variant={args.variant} "
          f"dtype={args.dtype} -> {args.out}")


if __name__ == "__main__":
    main()
