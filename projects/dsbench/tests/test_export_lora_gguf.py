"""Unit tests for the LoRA -> llama.cpp GGUF exporter (pure numpy; no model, no gguf, no torch).

The two reference functions below re-implement the operations of the two real stacks, copied from:
  * llama.cpp  conversion/qwen.py::_reorder_v_heads   (HF grouped -> GGUF tiled, at conversion)
  * transformers-gguf  TiledToGroupedRows / PermuteRows  (GGUF tiled -> HF grouped, at load)
so the exporter is checked against what each stack actually does, not against itself.
"""
from __future__ import annotations

import numpy as np
import pytest
from dsbench.sftgen.export_lora_gguf import (
    EXPERT_FF,
    NUM_EXPERTS,
    QK_ROWS,
    export_tensors,
    grouped_to_tiled,
    tiled_index,
)

K, PER_K, D = 16, 2, 128
RNG = np.random.default_rng(0)


def llamacpp_reorder_v_heads(t: np.ndarray, dim: int) -> np.ndarray:
    """conversion/qwen.py::_reorder_v_heads, verbatim in numpy."""
    shape = list(t.shape)
    new_shape = shape[:dim] + [K, PER_K, D] + shape[dim + 1:]
    t = t.reshape(new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.transpose(perm).reshape(shape)


def recipe_tiled_to_grouped_rows(t: np.ndarray, offset: int = 0) -> np.ndarray:
    """TiledToGroupedRows + PermuteRows.convert, verbatim in numpy."""
    tiled_from_grouped = np.arange(K * PER_K * D).reshape(K, PER_K, D).swapaxes(0, 1).reshape(-1)
    perm = np.argsort(tiled_from_grouped)
    if offset:
        return np.concatenate([t[:offset], t[offset:][perm]], axis=0)
    return t[perm]


def test_tiled_index_matches_llamacpp_converter():
    grouped = RNG.standard_normal((K * PER_K * D, 4))
    assert np.array_equal(grouped[tiled_index()], llamacpp_reorder_v_heads(grouped, 0))


def test_export_reorder_is_the_exact_inverse_of_the_recipe_load():
    """Round trip: GGUF (tiled) -> recipe load (grouped) -> export (tiled) must be the identity."""
    tiled = RNG.standard_normal((K * PER_K * D, 4))
    assert np.array_equal(grouped_to_tiled(recipe_tiled_to_grouped_rows(tiled), axis=0), tiled)


def test_qkv_reorders_only_the_value_block():
    grouped = RNG.standard_normal((QK_ROWS + K * PER_K * D, 4))
    tiled = grouped_to_tiled(grouped, axis=0, offset=QK_ROWS)
    assert np.array_equal(tiled[:QK_ROWS], grouped[:QK_ROWS])  # q and k rows never move
    assert np.array_equal(recipe_tiled_to_grouped_rows(tiled, offset=QK_ROWS), grouped)


def test_row_reorder_preserves_the_lora_output_up_to_head_order():
    """B[T] @ (A x) must equal (B @ A x)[T]: llama.cpp sees the same delta, in tiled order."""
    a, b = RNG.standard_normal((4, 2048)), RNG.standard_normal((K * PER_K * D, 4))
    x = RNG.standard_normal(2048)
    np.testing.assert_allclose(grouped_to_tiled(b, axis=0) @ (a @ x),
                               grouped_to_tiled(b @ (a @ x), axis=0), rtol=1e-12)


def test_column_reorder_keeps_out_proj_exact_on_tiled_input():
    """out_proj consumes the value axis: x_tiled @ A_tiled.T must equal x_grouped @ A.T."""
    a = RNG.standard_normal((4, K * PER_K * D))
    x_grouped = RNG.standard_normal(K * PER_K * D)
    x_tiled = grouped_to_tiled(x_grouped, axis=0)
    np.testing.assert_allclose(grouped_to_tiled(a, axis=1) @ x_tiled, a @ x_grouped, rtol=1e-12)


def _key(layer: int, module: str, kind: str) -> str:
    return f"base_model.model.model.layers.{layer}.{module}.{kind}.weight"


def test_expert_gate_up_split_puts_gate_first_and_shares_a():
    a = RNG.standard_normal((NUM_EXPERTS, 4, 2048))
    b = RNG.standard_normal((NUM_EXPERTS, 2 * EXPERT_FF, 4))
    out_a = dict(export_tensors(_key(0, "mlp.experts", "lora_A"), a))
    out_b = dict(export_tensors(_key(0, "mlp.experts", "lora_B"), b))
    assert out_a["blk.0.ffn_gate_exps.weight.lora_a"] is a
    assert out_a["blk.0.ffn_up_exps.weight.lora_a"] is a
    assert np.array_equal(out_b["blk.0.ffn_gate_exps.weight.lora_b"], b[:, :EXPERT_FF, :])
    assert np.array_equal(out_b["blk.0.ffn_up_exps.weight.lora_b"], b[:, EXPERT_FF:, :])
    swapped = dict(export_tensors(_key(0, "mlp.experts", "lora_B"), b, "swap_gate_up"))
    assert np.array_equal(swapped["blk.0.ffn_gate_exps.weight.lora_b"], b[:, EXPERT_FF:, :])


def test_no_vperm_variant_really_skips_the_reorder():
    b = RNG.standard_normal((K * PER_K * D, 4))
    (_, kept), = export_tensors(_key(0, "linear_attn.in_proj_z", "lora_B"), b, "no_vperm")
    assert np.array_equal(kept, b)


def test_full_adapter_layout_maps_every_tensor():
    """The real adapter: 30 GDN + 10 full-attention layers, MoE everywhere -> 660 in, 740 out."""
    shapes = {"linear_attn.in_proj_qkv": ((4, 2048), (QK_ROWS + 4096, 4)),
              "linear_attn.in_proj_z": ((4, 2048), (4096, 4)),
              "linear_attn.out_proj": ((4, 4096), (2048, 4)),
              "self_attn.q_proj": ((4, 2048), (8192, 4)), "self_attn.k_proj": ((4, 2048), (512, 4)),
              "self_attn.v_proj": ((4, 2048), (512, 4)), "self_attn.o_proj": ((4, 4096), (2048, 4)),
              "mlp.shared_expert.gate_proj": ((4, 2048), (512, 4)),
              "mlp.shared_expert.up_proj": ((4, 2048), (512, 4)),
              "mlp.shared_expert.down_proj": ((4, 512), (2048, 4))}
    inputs, names = 0, []
    for layer in range(40):
        attn = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
        gdn = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")
        shared = ("mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
                  "mlp.shared_expert.down_proj")
        mods = (attn if layer % 4 == 3 else gdn) + shared
        for m in mods:
            for kind, shape in zip(("lora_A", "lora_B"), shapes[m], strict=True):
                exported = export_tensors(_key(layer, m, kind), np.zeros(shape))
                names += [n for n, _ in exported]
                inputs += 1
        for kind, shape in (("lora_A", (256, 4, 2048)), ("lora_B", (256, 1024, 4)),
                            ("lora_A_down", (256, 4, 512)), ("lora_B_down", (256, 2048, 4))):
            exported = export_tensors(_key(layer, "mlp.experts", kind), np.zeros(shape))
            names += [n for n, _ in exported]
            inputs += 1
    assert inputs == 660 and len(names) == 740 and len(set(names)) == 740


def test_rejects_what_it_does_not_understand():
    with pytest.raises(ValueError, match="no GGUF mapping"):
        export_tensors(_key(0, "mlp.gate", "lora_A"), np.zeros((4, 2048)))
    with pytest.raises(ValueError, match="experts"):  # llama.cpp itself would not catch this
        export_tensors(_key(0, "mlp.experts", "lora_A"), np.zeros((128, 4, 2048)))
