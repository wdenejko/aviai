# Thinking-on rendering and packing: labels from the served prompt, llama.cpp's tokenization

**Date:** 2026-09-29 · raw numbers: `20260929-thinking-rendering-packing.json` · code:
`src/dsbench/sftgen/tokenize_masked.py`, `src/dsbench/sftgen/build_masked_dataset.py` · tests:
`tests/test_tokenize_masked.py` · box scripts: `patches/thinking-packing/` · run dir:
`~/benchlab/runs/2026-09-29-qwen36-thinking-packing/` on dashi

**Status: ADR-004 Revision 2's rendering and packing are built, and validated on the reasoning
pilot's real traces. One finding changes how Gate 1 and Gate 2 read.**
- **The training tokenizer did not split text the way the server does.** transformers' GGUF
  converter leaves `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>` and
  `</tool_response>` as BPE pieces; llama.cpp keeps each one whole. Every Gate 1 and Gate 2
  training row carried the split form. Now fixed: on the pilot's 200 prompts, HF counted exactly
  2 tokens more than llama-server on every one, and now counts the same.
- **Labels start where the served prompt ends.** Turns before the last user query are context
  only, and rows are bin-packed whole into 8192-token blocks. On the pilot: 190 of 199 finished
  rows in 56 blocks, 97.7% full, with no label invariant broken.
- **Open: rows in one block still see each other.** The model code can reset the GatedDeltaNet
  state at a row boundary if it is told where the boundaries are, but the recipe doesn't tell it.
  Settling this needs a GPU check.

## The tokenizer: what the server reads, and what training read

A GGUF records each vocabulary entry's type. llama.cpp never breaks a token of type CONTROL or
USER_DEFINED into pieces: USER_DEFINED always, CONTROL whenever it parses special tokens, which the
server does for every prompt. This GGUF has 33 such tokens:
- 27 CONTROL: the ChatML markers, vision, audio and fill-in-the-middle tokens;
- 6 USER_DEFINED: `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`,
  `</tool_response>`.

transformers builds its tokenizer from the same GGUF, but registers only 4 of the 33 as added
tokens, the ChatML markers among them. The rest go through BPE:

| Text | llama.cpp | The training tokenizer of Gate 1 and Gate 2 |
|---|---|---|
| `<think>` | 1 token (248068) | 3: `<th` `ink` `>` |
| `</think>` | 1 token (248069) | 3: `</` `think` `>` |
| `<tool_call>` | 1 token (248058) | 4: `<` `tool` `_call` `>` |
| an assistant turn with reasoning (sample) | 12 tokens | 16 |
| a tool call and its response (sample) | 42 tokens | 54 |

**The check against the server.** The reasoning pilot recorded each prompt's token count from
llama-server's own usage block. Its prompts end in the thinking-on opener `<think>\n`. The same
prompts, rendered with the same template and counted by HF:

| HF tokenizer | HF count minus llama-server's, per prompt |
|---|---|
| as transformers' GGUF converter builds it | **+2 on all 200** |
| with the GGUF's 33 atomic tokens registered (`match_llama_tokenization`) | **0 on all 200** |

The +2 is the opener: `<think>` in three pieces instead of one. After the fix, HF and llama.cpp
agree on every prompt, including code, system prompts and SQL schemas. Registering the tokens
changes how text is split, not which ids exist: the vocabulary stays at 248,320 entries, and
`<think>` keeps id 248068.

**What it meant for Gate 1 and Gate 2.** In the Gate-2 mixture, the `<think>` of 27,690 trained
assistant turns and the tool-call tags of its 896 tool rows were token sequences the served model
never reads or writes. Its thinking-off prompt ends in the empty block with `<think>` and `</think>`
as single tokens. The adapter had been trained on that block with both tags in BPE pieces.

That fits the think-leak: the adapter opened a block of its own after the served one. But nothing
here proves the link. The battery report's mechanism, `<think>` as a trained token, stands, and no
run separates the two. Gate 2's held-out losses were also measured on the split form
(`eval_adapter_ab.py` uses the Gate 1/2 path), so they don't compare one to one with anything
tokenized the new way.

## The labels

`thinking_record` builds one row. Each rule answers a Gate-2 failure (ADR-004 Revision 2):

- **The label starts where the served prompt ends.** For each trained turn, the conversation up to
  that turn is rendered with the thinking-on generation prompt, which is the text the server sends.
  That text must be a prefix of the full rendering, as characters and as tokens, and the label
  starts right after it. The row is rejected if a token straddles that edge, or if the prompt
  tokenizes differently alone than inside the row.
- **Only assistant turns after the last user query are trained**, by the template's own rule for
  which turns keep their reasoning. In a tool loop that is every turn; in a chat it is the last
  one. Earlier turns stay in the prompt without a label.
- **A trained turn with empty reasoning rejects its row.** The template renders it as
  `<think>\n\n</think>`, and its label would teach closing the block at once.

On the pilot's 199 finished rows (one reply had hit the 16,384-token cap and is left out, as the
mixture's assembly will leave such replies out), with the real tokenizer and template:

| Check | Result |
|---|---|
| Rows rejected | 0 (every pilot reply has reasoning) |
| Trained span follows `<think>` `\n` and ends on `<|im_end|>`, with one `</think>` inside | 199 of 199 |
| Row length minus (llama-server's prompt + generated tokens) | +1 for 190 rows, 0 for 7, −1 and +2 once each |

The +1 is the template's newline after `<|im_end|>`. It is part of the row, but no label. The
other 9 rows differ by one or two tokens, because the model can sample a non-canonical split of its
own text, and the template trims whitespace.

## Packing

`bin_pack` is first-fit decreasing. Rows go longest first, each into the first block with room,
and never split. Each row is followed by `<|endoftext|>`, the document boundary the model knows
from pretraining. The block's tail is padded with the same token, and neither separator nor
padding gets a label. `num_tokens` stays the block length. The recipe's collator makes it the
attention mask, and a mask of all ones keeps the tuned attention path; padding at a block's end
can't reach the tokens before it, because every layer is causal.

| Pilot, 8192-token blocks | |
|---|---:|
| Rows | 199 |
| Too long (over 8,191 tokens with the separator) | 9 (3 Target A, 6 opencoder) |
| Rows packed | 190 |
| Blocks | 56 |
| Fill | 97.7% |
| Trained tokens, of all block tokens | 87.0% |
| Rows per block | 2-9 (24 blocks hold 2) |

The 9 long rows are the ones the pilot report counted: 190 of 200 fit at 8192, where the 200th is
the capped reply. `build_masked_dataset.py` ran end to end on the same records: 56 blocks, saved
dataset and report. Its `--legacy` mode still rebuilds the Gate 1/2 way.

## Open: rows in a block see each other

Code reading on the box, in transformers 5.17.0.dev0 and the recipe's collator:

| Path | Can it stop at a row boundary? | Today |
|---|---|---|
| GatedDeltaNet recurrent state (30 of 40 layers) | Yes: the layer passes `cu_seq_lens_q` to FLA's chunk rule as `cu_seqlens` | Not given one; the state flows from row to row |
| GatedDeltaNet short convolution (4 taps) | No: the torch fallback ignores sequence boundaries | A row's first 3 tokens mix in the previous row's last 3 |
| Full attention (10 of 40 layers) | Needs the attention kernel's variable-length entry, not verified for the aiter kernel | Attends across rows |
| The recipe's collator | Passes neither position ids nor sequence lengths | — |

So today a packed row sees the rows before it in its block, across an `<|endoftext|>`. The first
row of each block sees nothing before it: 56 of the pilot's 190. The convolution leak touches only
the first 3 tokens of a row, which are prompt tokens (`<|im_start|>`, the role, a newline) with no
label.

Next, both steps needing a GPU window:
- a check that a block run with its row lengths gives the same logits as its rows run alone;
- a collator patch that passes the lengths.

## What it means for the retrain

1. **Build with the defaults.** `build_masked_dataset.py` now means 8192-token blocks, the matched
   tokenizer and Revision 2's labels.
2. **Don't compare losses across the tokenizer change.** A Gate-2 loss and a retrain loss are
   measured on different token sequences for the same text.
3. **One rendering item is left** (ADR-004 Revision 2, action item 1): rows in a block seeing each
   other.

## Ops notes

- **`~/ftgguf`'s Python runs only inside the `llama-rocm-unlimited-build` toolbox**, even for CPU
  work such as tokenizing: its torch needs `libatomic.so.1`, and the host lacks it.
- **`llama-tokenize` from the qwen4exp `build-v2` segfaults on this GGUF** (exit 139). The
  server's own usage counts from the pilot served as the reference instead.
- **A byte-level vocabulary spells a newline `Ċ`.** `convert_tokens_to_ids("\n")` is not id 198;
  ask the tokenizer (`tok("\n")`). The first run of the validation tripped on exactly this.
