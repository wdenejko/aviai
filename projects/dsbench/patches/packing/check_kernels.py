"""Do the kernels that packed rows switch to compute what the ones Gate 1 and 2 used compute, and
do their backward passes keep to the boundaries too?

check_packed_rows.py found each packed row bit for bit equal to the same row alone, both with
boundaries, and the block's loss equal to the rows' weighted loss. Two things it can't see:
- A block's first row without boundaries came out 29% away from the same row with them (relative
  L2 of the last hidden state). Nothing comes before that row, and only the kernels differ:
  - attention: aiter's `flash_attn_func` (no boundaries) or `flash_attn_varlen_func`
    (boundaries), called through transformers' `_flash_attention_forward`, as the model does;
  - the GatedDeltaNet rule: FLA's `chunk_gated_delta_rule` without or with `cu_seqlens`;
  - the convolution: transformers' torch fallback or FLA's `causal_conv1d`.
- The block's gradients came out 1.2-1.7% away from the rows' weighted gradients, where the
  forward was exact. A backward pass that ignored the boundaries would pass gradient from one row
  into the one before it, and show up only there.

Each kernel runs here on random input at the model's shapes (attention: 16 query heads, 2
key-value heads, head size 256; the rule: 32 heads of 128; the convolution: 8,192 channels,
width 4), on T tokens with a boundary at SPLIT, off the kernels' block edges:
- forward: attention on each path against float32 references on the first segment, causal and
  not; the rule and the convolution with boundaries against without on the first segment, and
  on the second against the second run alone;
- backward: a loss on the second segment's output only. With boundaries, no gradient may reach
  the first segment's inputs (`bwd_into_first`, the largest absolute value: 0 exactly); without
  them, some does. The second segment's input gradients are compared with the segment run alone.

On the box, in the ftgguf toolbox, on the GPU (a minute or two, mostly autotuning):

    python check_kernels.py OUT.json
"""

from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F

T, SPLIT = 1024, 600
DEVICE = "cuda:0"
ATTN = "kernels-community/aiter-flash-attn"


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def largest(grads: list[torch.Tensor], positions: slice, dim: int = 1) -> float:
    """The largest absolute gradient at these positions, over the inputs."""
    def at(g: torch.Tensor) -> torch.Tensor:
        index = [slice(None)] * g.dim()
        index[dim] = positions
        return g[tuple(index)]

    return max(float(at(g).abs().max()) for g in grads)


def second_only(run, inputs: list[torch.Tensor], cut, weight: torch.Tensor) -> list[torch.Tensor]:
    """Input gradients of a loss on the output's `cut` alone."""
    leaves = [x.detach().clone().requires_grad_() for x in inputs]
    (run(*leaves)[cut].float() * weight).sum().backward()
    return [x.grad for x in leaves]


def reference(q, k, v, causal: bool) -> torch.Tensor:
    """float32 attention in [B, T, H, D], key-value heads repeated for the query heads."""
    groups = q.shape[2] // k.shape[2]
    q_, k_, v_ = (x.float().transpose(1, 2) for x in (q, k.repeat_interleave(groups, 2),
                                                        v.repeat_interleave(groups, 2)))
    return F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal).transpose(1, 2)


def attention() -> dict:
    from transformers.integrations.hub_kernels import load_and_register_attn_kernel
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

    load_and_register_attn_kernel(ATTN)
    q = torch.randn(1, T, 16, 256, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, T, 2, 256, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(1, T, 2, 256, device=DEVICE, dtype=torch.bfloat16)
    cu = torch.tensor([0, SPLIT, T], dtype=torch.int32, device=DEVICE)

    def plain(q, k, v):
        return _flash_attention_forward(q, k, v, None, q.shape[1], is_causal=True,
                                        attn_implementation=ATTN)

    def varlen(q, k, v):
        return _flash_attention_forward(
            q, k, v, None, q.shape[1], is_causal=True, attn_implementation=ATTN,
            cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=SPLIT, max_length_k=SPLIT)

    first, second = slice(0, SPLIT), slice(SPLIT, T)
    out_plain, out_varlen = plain(q, k, v), varlen(q, k, v)
    causal = reference(q[:, first], k[:, first], v[:, first], causal=True)
    full = reference(q[:, first], k[:, first], v[:, first], causal=False)
    second_causal = reference(q[:, second], k[:, second], v[:, second], causal=True)
    weight = torch.randn(1, T - SPLIT, 16, 256, device=DEVICE)
    cut = (slice(None), second)
    g_varlen = second_only(varlen, [q, k, v], cut, weight)
    g_plain = second_only(plain, [q, k, v], cut, weight)
    g_alone = second_only(plain, [x[:, second] for x in (q, k, v)], (slice(None),), weight)
    return {
        "plain_vs_causal": rel(out_plain[:, first], causal),
        "plain_vs_noncausal": rel(out_plain[:, first], full),
        "varlen_vs_causal": rel(out_varlen[:, first], causal),
        "varlen_vs_noncausal": rel(out_varlen[:, first], full),
        "varlen_second_vs_causal_alone": rel(out_varlen[:, second], second_causal),
        # position 0 attends to itself only, if causal
        "plain_pos0_vs_causal": rel(out_plain[:, :1], causal[:, :1]),
        "varlen_pos0_vs_causal": rel(out_varlen[:, :1], causal[:, :1]),
        "bwd_into_first_varlen": largest(g_varlen, first),
        "bwd_into_first_plain": largest(g_plain, first),
        "bwd_second_varlen_vs_alone": max(rel(a[:, second], b)
                                          for a, b in zip(g_varlen, g_alone, strict=True)),
    }


def gated_delta_rule() -> dict:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    q, k, v = (torch.randn(1, T, 32, 128, device=DEVICE, dtype=torch.bfloat16) for _ in range(3))
    g = F.logsigmoid(torch.randn(1, T, 32, device=DEVICE, dtype=torch.float32))
    beta = torch.rand(1, T, 32, device=DEVICE, dtype=torch.bfloat16)
    inputs = [q, k, v, g, beta]
    cu = torch.tensor([0, SPLIT, T], dtype=torch.int32, device=DEVICE)

    def plain(*args):
        return chunk_gated_delta_rule(*args, use_qk_l2norm_in_kernel=True)[0]

    def varlen(*args):
        return chunk_gated_delta_rule(*args, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)[0]

    first, second = slice(0, SPLIT), slice(SPLIT, T)
    out_plain, out_varlen = plain(*inputs), varlen(*inputs)
    out_alone = plain(*(x[:, second] for x in inputs))
    weight = torch.randn(1, T - SPLIT, 32, 128, device=DEVICE)
    cut = (slice(None), second)
    g_varlen = second_only(varlen, inputs, cut, weight)
    g_plain = second_only(plain, inputs, cut, weight)
    g_alone = second_only(plain, [x[:, second] for x in inputs], (slice(None),), weight)
    return {
        "varlen_vs_plain_first": rel(out_varlen[:, first], out_plain[:, first]),
        "varlen_second_vs_alone": rel(out_varlen[:, second], out_alone),
        "plain_second_vs_alone": rel(out_plain[:, second], out_alone),
        "bwd_into_first_varlen": largest(g_varlen, first),
        "bwd_into_first_plain": largest(g_plain, first),
        "bwd_second_varlen_vs_alone": max(rel(a[:, second], b)
                                          for a, b in zip(g_varlen, g_alone, strict=True)),
    }


def convolution() -> dict:
    from fla.modules.convolution import causal_conv1d
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen

    x = torch.randn(1, 8192, T, device=DEVICE, dtype=torch.bfloat16)  # transformers: [B, D, T]
    conv_weight = torch.randn(8192, 4, device=DEVICE, dtype=torch.bfloat16) * 0.5  # frozen
    cu = torch.tensor([0, SPLIT, T], dtype=torch.int32, device=DEVICE)

    def torch_conv(x):
        return qwen.causal_conv1d_fn(x, conv_weight, None, activation="silu")

    def fla_conv(x, cu_seqlens=cu):
        out, _ = causal_conv1d(x.transpose(1, 2), weight=conv_weight, bias=None,
                               activation="silu", cu_seqlens=cu_seqlens)
        return out.transpose(1, 2)

    first, second = slice(0, SPLIT), slice(SPLIT, T)
    out_torch, out_fla = torch_conv(x), fla_conv(x)
    out_alone = torch_conv(x[:, :, second])
    weight = torch.randn(1, 8192, T - SPLIT, device=DEVICE)
    cut = (slice(None), slice(None), second)
    g_fla = second_only(fla_conv, [x], cut, weight)
    g_torch = second_only(torch_conv, [x], cut, weight)
    g_alone = second_only(torch_conv, [x[:, :, second]], (slice(None),), weight)
    g_fla_alone = second_only(lambda t: fla_conv(t, None), [x[:, :, second]], (slice(None),),
                              weight)
    return {
        "fla_vs_torch_first": rel(out_fla[:, :, first], out_torch[:, :, first]),
        "fla_second_vs_torch_alone": rel(out_fla[:, :, second], out_alone),
        "torch_second_vs_alone": rel(out_torch[:, :, second], out_alone),
        "bwd_into_first_fla": largest(g_fla, first, dim=2),
        "bwd_into_first_torch": largest(g_torch, first, dim=2),
        "bwd_second_fla_vs_fla_alone": rel(g_fla[0][:, :, second], g_fla_alone[0]),
        "bwd_second_fla_vs_torch_alone": rel(g_fla[0][:, :, second], g_alone[0]),
    }


def main() -> None:
    torch.manual_seed(0)
    report = {"T": T, "split": SPLIT}
    for name, fn in (("attention", attention), ("gated_delta_rule", gated_delta_rule),
                     ("convolution", convolution)):
        report[name] = fn()
        print(name, json.dumps(report[name], indent=1), flush=True)
    with open(sys.argv[1], "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
