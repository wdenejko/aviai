# Packed rows: each row in a block trains as if it were alone

ADR-004 Revision 2, action item 1, its last point. Run 2026-10-02 on dashi, in one GPU window of
under 9 minutes (`patches/packing/window.sh` around `check_packed_rows.py`), with CPU checks before
it and a kernel test during and after it (`check_kernels.py`). Outputs:
- the numbers are in `20261002-packed-rows-check.json`;
- the raw outputs are in `~/benchlab/runs/2026-10-02-qwen36-packed-rows/`.

## Summary

- **With its rows' boundaries, a block trains each row exactly as if it were alone.**
  - Each packed row's last hidden state equals its row alone, bit for bit, on all four rows.
  - The block's loss is the rows' loss weighted by their labelled tokens: 0.15419 both, equal to
    7 digits.
  - The block's LoRA gradients are the rows' weighted gradients to within 1.2-1.7% (relative L2).
    That is the rounding of a backward pass run at another scale: no kernel passes any gradient
    across a boundary.
- **Without boundaries, the way the retrain would have run:**
  - rows 2 to 4 come out 39-71% away from themselves alone, and their first 16 tokens 85-114%;
  - the block's loss is 0.221 instead of 0.154;
  - the gradients point elsewhere: cosine 0.25-0.46 with the rows' own, by parameter group.

  These figures also carry the rounding difference described two points down.
- **No kernel computes anything else, forward or backward.** Tested on random inputs at the
  model's shapes:
  - attention: aiter's plain and variable-length paths give the same output, and both are causal;
  - the GatedDeltaNet rule: with boundaries, the same as without on a first segment, bit for bit;
  - the convolution: FLA's and transformers' torch fallback differ by rounding, 0.3%;
  - backward, with a loss on a second segment only: with boundaries, exactly no gradient reaches
    the first segment through any of the three; without them, all three pass some.
- **A block's first row also moves without boundaries, by 29%, though nothing comes before it.**
  Only the convolution computes it differently, and only by rounding. That 0.3% grows to 29%
  through 40 layers that each route a token to 8 of 256 experts. It is a property of the network,
  not of packing, but this check didn't measure it directly.
- **Speed and memory.**
  - A warm packed step (forward and backward, 8,192 tokens) took 23.6 s. The seq-4096 report's
    warm 8,192-token step took 44.9 s, on rows glued without boundaries.
  - With boundaries, attention covers each row instead of the whole block, and the convolution
    runs on FLA. This check doesn't separate the two. Its one step without boundaries (76.3 s)
    also paid for first calls.
  - Peak memory was 17.2 GiB reserved, against 17.8 there.

## What was built

- **The dataset.** `tokenize_masked.pack_thinking` records each block's segments: each row with
  its separator, then the padding (`stats["block_seq_lens"]`). `build_masked_dataset` writes them
  as `seq_lens`.
- **The collator.** `patches/recipe-train-packed-rows.patch` and `patches/packing/packed_rows.py`
  turn `seq_lens` into the model's packed-sequence arguments:
  - position ids that restart at each row, as each conversation's do when it is served;
  - `cu_seq_lens_q/k` and `max_length_q/k`.
- **The convolution.** The box has no `causal-conv1d` package, so transformers runs its torch
  fallback, which ignores boundaries. `configure_qwen35_packed_conv` sends the convolution to
  FLA's `causal_conv1d` when the boundaries are given.
- **A guard.** The recipe refuses `seq_lens` under any attention but flash's. Only flash
  attention's variable-length path uses the boundaries, so rows would otherwise attend to each
  other again, silently.
- **Traced in the box's code** (transformers-gguf `fdd5e1a`, FLA 0.5.2):
  - the recipe keeps unused columns, so `seq_lens` reaches the collator;
  - the attention mask is all ones, so flash attention gets no mask, and the explicit
    `cu_seq_lens_*` send it down `flash_attn_varlen_func`;
  - the decoder passes its arguments to the GatedDeltaNet, which passes `cu_seq_lens_q` to the
    rule as `cu_seqlens` and to the convolution;
  - the loss reads only the arguments it knows.

## The checks

**On the CPU, on the box.**
- **The patch** applies to the box's own `train_qwen3_5_35b.py`.
- **The collator** (`check_collator.py`): a Revision 2 dataset built from the reasoning pilot's
  199 rows (56 blocks, 190 rows, 97.7% full) runs through the collator, block by block. The
  collator is compiled from the patched recipe's source. On every block:
  - the segments are the build report's and cover the block;
  - position ids restart at 0 at each segment;
  - a separator precedes every boundary;
  - the labels are the dataset's.

  It also refuses a batch of two blocks, and leaves a block without `seq_lens` as it was.

**On the GPU: one block, three ways** (`check_packed_rows.py`).
- **The model** is set up as the recipe sets it up for training: the GGUF base, rank-4 LoRA with
  B = 0, `kernels-community/aiter-flash-attn`, FLA, and gradient checkpointing.
- **The block** holds 4 real thinking-on rows and 157 tokens of padding. None of the boundaries
  falls on an edge of FLA's 64-token chunks.

  | Row | Tokens | Labelled | Loss alone |
  |---|---:|---:|---:|
  | Target C `C-mlc_churn_rare-221` (tool calls, several turns) | 2,978 | 1,188 | 0.1435 |
  | Target C `C-mlc_widget_defect-215` | 3,182 | 1,153 | 0.1084 |
  | reasoning pilot `A-weekday-numbering-clickhouse-support_tickets-200013` (one turn) | 1,029 | 938 | 0.1939 |
  | reasoning pilot `A-month-bucket-clickhouse-iot_readings-329661` | 842 | 753 | 0.1917 |

- **The three ways:**
  - `packed`: with the boundaries;
  - `today`: without them;
  - `alone`: each row by itself at the start of a block, with its boundaries, and padding after.
- **Two passes each:**
  - a forward pass, for the last hidden state at every row token;
  - one training step, for the loss and the LoRA-B gradients. LoRA-A's gradients are zero while
    B is.
- **Noise:** the packed block run twice. Forward and backward are deterministic: it is exactly 0
  both times.

## Results

**Hidden states** (relative L2 of each row's last hidden state against its row alone; `head` is
the row's first 16 tokens):

| Row | Packed | Packed, head | Without boundaries | Without, head |
|---|---:|---:|---:|---:|
| 1 `C-mlc_churn_rare-221` | 0 | 0 | 0.286 | 0.144 |
| 2 `C-mlc_widget_defect-215` | 0 | 0 | 0.706 | 1.14 |
| 3 `A-weekday-numbering-...-200013` | 0 | 0 | 0.386 | 0.847 |
| 4 `A-month-bucket-...-329661` | 0 | 0 | 0.409 | 0.887 |

- Every packed row is equal to its row alone: the largest difference is 0.
- Without boundaries, the rows after the first are furthest off at their start. That is where
  the previous row's state and the convolution's last 3 tokens arrive first.

**Loss:**
- packed and packed again: 0.15418851;
- the rows alone, weighted by labelled tokens: 0.15418853;
- without boundaries: 0.22087.

**LoRA-B gradients**, by parameter group, against the rows' gradients weighted the same way:

| Group | Tensors | Packed, relative L2 | Packed, largest difference | Without boundaries, relative L2 | Without, cosine |
|---|---:|---:|---:|---:|---:|
| attention | 40 | 0.0139 | 3.8e-5 | 1.34 | 0.30 |
| GatedDeltaNet | 90 | 0.0148 | 1.2e-4 | 1.27 | 0.25 |
| experts | 80 | 0.0166 | 1.5e-5 | 1.22 | 0.32 |
| shared expert | 120 | 0.0124 | 1.7e-5 | 1.12 | 0.46 |

The JSON's cosines are computed in float32 over up to 126M elements and are off by up to 2%
(some exceed 1). The relative L2 and the largest differences are exact.

### Kernels on their own (`check_kernels.py`)

- **The setup.**
  - Random inputs at the model's shapes:
    - attention: 16 query heads, 2 key-value heads, head size 256;
    - the rule: 32 heads of 128;
    - the convolution: 8,192 channels, width 4, bf16 weights, as the model loads them.
  - 1,024 tokens with a boundary at 600.
  - Attention goes through transformers' `_flash_attention_forward`, as the model calls it.
- **When it ran.** A forward-only version ran during the window's last steps. The full run came 7
  minutes after the window, with production still down. Its tensors are a few megabytes, and it
  took 30 s.

| | Attention | GatedDeltaNet rule | Convolution |
|---|---|---|---|
| Forward, first segment, with vs without boundaries | the same (both 0.20% from float32 causal, 207% from non-causal) | 0 | 0.31% (FLA vs torch) |
| Forward, second segment vs alone, with boundaries | 0.20% from float32 causal | 0 | 0.31% vs torch alone |
| Forward, second segment vs alone, without | (sees the first) | 3.4% | 6.0% |
| Backward into the first segment, with boundaries (largest) | **0** | **0** | **0** |
| Backward into the first segment, without | 1.09 | 0.116 | 5.84 |
| Backward, second segment vs alone, with boundaries | 0 | 6.6e-7 | 0 (vs FLA alone), 0.27% (vs torch) |

Everything else in the model works token by token. So no path runs from one row into another,
in either direction, and the gradients' 1.2-1.7% is rounding:
- the block's backward runs at the loss scale 1/N, and a row's at 1/n;
- bf16 rounds the two differently;
- the block also sums all its rows into one bf16 gradient, where the reference adds four rounded
  ones.

### The first row without boundaries

Nothing comes before the first row, so its 29% isn't a leak.
- **Attention and the GatedDeltaNet rule** compute it the same way on both paths.
- **The convolution differs by rounding.** The torch fallback rounds to bf16 after the
  convolution and again after its activation; FLA's kernel computes in fp32 and rounds once. The
  model loads the convolution's weights in bf16: the GGUF stores them in F32, but the model keeps
  no module in fp32.
- **The rest is the network.** A 0.3% difference in each of 30 convolutions, through 40 layers
  that route each token to 8 of 256 experts, is enough to tip close routing choices.
- **Not measured directly.** That would take one run of the first row with only the convolution
  switched.
- **It doesn't matter for the retrain.** The retrain runs every block on FLA's convolution, so it
  is one consistent rounding. It is also a caution for any comparison that switches kernels: on
  this model, small rounding differences show up as large differences in hidden states.

### Time and memory

| Pass (8,192 tokens) | Seconds |
|---|---:|
| Forward, packed, first (autotunes FLA's variable-length kernels) | 56.4 |
| Forward, without boundaries, first | 14.3 |
| Forward, a row alone (warm) | 5.4 |
| Forward, packed again (warm) | 5.3 |
| Step, packed, first (autotunes the backward) | 113.3 |
| Step, without boundaries, first | 76.3 |
| Step, a row alone (warm) | 25.2-29.2 |
| Step, packed again (warm) | 23.6 |

- The forward-only kernel test overlapped the last warm steps (23.6 to 29.2 s).
- **The runs without boundaries were each the first of their kind:**
  - transformers' torch convolution goes through MIOpen, whose tuning database is unreadable on
    the box (it warns), and its first call compiles;
  - the non-varlen Triton kernels compile on first use.

  So 76.3 s against 23.6 s measures nothing.
- **The seq-4096 report's warm 8,192-token step** took 44.9 s (182 tokens a second, the
  optimizer included). That was rows glued without boundaries, attention over the whole block,
  and the torch convolution.
  - It put the superlinear cost on attention's quadratic term. With boundaries, attention's cost
    follows the rows' lengths, not the block's, so the retrain's rate depends on its mix of rows.
  - Its first steps will give the rate.
- **Memory:** 17.0 GiB allocated at the peak, 17.2 GiB reserved (the seq-4096 audit at 8,192:
  17.8 GiB reserved).

## What it means for the retrain

1. **Revision 2 trains with the boundaries.**
   - Build with `build_masked_dataset.py`, which writes `seq_lens`.
   - Apply `recipe-train-packed-rows.patch` with `packed_rows.py` in the recipe directory, and
     keep batch size 1.
   - Attention must be flash's (`QWEN35_ATTN_IMPL=kernels-community/aiter-flash-attn`, as
     `gate2_train.sh` sets it). The recipe now refuses anything else.
2. **The logged loss means what it says.** It is the mean over the block's labelled tokens, and
   it equals the rows' own losses weighted the same way.
3. **Action item 1 is done.** Also from this work: Target C's selection now counts the separator
   that follows a row in its block, so a row can be at most `block - 1` tokens, as packing
   requires. No kept row changes.

**Limits:**
- **One block:** four rows at 8,192 tokens, at the start of training (the base, LoRA B = 0). Only
  LoRA-B gradients exist there. The kernels' isolation was shown on input gradients, which
  carries to any later state.
- **The kernel test** used random inputs, 1,024 tokens and one boundary.
- **The first row's 29%** is explained by rounding through the network, not measured.
- **The speed difference** is not measured.
