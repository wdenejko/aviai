"""Packed rows for the Qwen3.5-MoE recipe: each row in a block attends only to itself.

Revision 2's blocks hold several whole rows (`tokenize_masked.pack_thinking`). Without boundaries,
three paths carry one row into the next (reports/gate-evals/20260929-thinking-rendering-packing.md):
- the 10 full-attention layers attend across the whole block;
- the 30 GatedDeltaNet layers carry their recurrent state from row to row;
- their 4-tap convolution mixes a row's first 3 tokens with the previous row's last 3.

The dataset's `seq_lens` (each row with its separator, then the padding) gives the boundaries:
- `segment_kwargs` turns them into the model's packed-sequence arguments. `cu_seq_lens_q` and
  `cu_seq_lens_k` send flash attention down its variable-length path, and the GatedDeltaNet passes
  `cu_seq_lens_q` to FLA's chunk rule as `cu_seqlens`. Position ids restart at 0 in every row, as
  each conversation does when it is served.
- `configure_qwen35_packed_conv` gives the convolution the same boundaries. The box has no
  `causal-conv1d` package, so transformers runs its torch fallback, which ignores them; FLA's
  `causal_conv1d` takes `cu_seqlens`. Without `cu_seq_lens_q`, the original runs, so unpacked
  inputs are untouched.
- Flash attention is the only attention path that uses the boundaries: the recipe's
  `kernels-community/aiter-flash-attn`. An all-ones attention mask becomes no mask, and the
  explicit `cu_seq_lens_*` send it down `flash_attn_varlen_func`.

FLA's chunk rule flattens the batch, so a packed batch holds one block: the recipe trains with
`per_device_train_batch_size=1`. Used by the patched collator
(patches/recipe-train-packed-rows.patch) and by patches/packing/check_packed_rows.py, which tests
the whole path on the GPU.
"""

from __future__ import annotations

import torch


def segment_kwargs(seq_lens: list[int]) -> dict:
    """The model's packed-sequence arguments for one block whose segments have these lengths."""
    if not seq_lens or min(seq_lens) < 1:
        raise ValueError(f"segments must be positive lengths, got {seq_lens}")
    lengths = torch.tensor(seq_lens, dtype=torch.int32)
    cu_seq_lens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(lengths, 0)
    longest = int(lengths.max())
    return {
        "position_ids": torch.cat([torch.arange(n) for n in seq_lens]).unsqueeze(0),
        "cu_seq_lens_q": cu_seq_lens,
        "cu_seq_lens_k": cu_seq_lens,
        "max_length_q": longest,
        "max_length_k": longest,
    }


def add_segments(batch: dict, seq_lens: list[list[int]] | None) -> dict:
    """The collator's step: one block's segments, checked against its tokens, into `batch`."""
    if seq_lens is None or seq_lens[0] is None:
        return batch
    if len(seq_lens) != 1:
        raise ValueError("packed rows need one block a batch: FLA's varlen rule flattens it")
    segments = [int(n) for n in seq_lens[0]]
    if sum(segments) != batch["input_ids"].shape[1]:
        raise ValueError(f"segments sum to {sum(segments)}, the block is "
                         f"{batch['input_ids'].shape[1]} tokens")
    batch.update(segment_kwargs(segments))
    return batch


def configure_qwen35_packed_conv() -> bool:
    """Route the GatedDeltaNet convolution through FLA's `causal_conv1d` when the call carries
    `cu_seq_lens_q`; any other call keeps transformers' own function."""
    from fla.modules.convolution import causal_conv1d
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen

    original = qwen.causal_conv1d_fn
    if getattr(original, "_packed_rows", False):
        return True

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        cu_seq_lens = kwargs.get("cu_seq_lens_q")
        if cu_seq_lens is None:
            return original(hidden_states, weight, bias, activation=activation, **kwargs)
        # transformers passes [batch, channels, time]; FLA takes [batch, time, channels].
        out, _ = causal_conv1d(hidden_states.transpose(1, 2), weight=weight, bias=bias,
                               activation=activation, cu_seqlens=cu_seq_lens)
        return out.transpose(1, 2)

    causal_conv1d_fn._packed_rows = True
    qwen.causal_conv1d_fn = causal_conv1d_fn
    return True
