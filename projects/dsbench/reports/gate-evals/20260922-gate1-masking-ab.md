# A/B: assistant-only loss masking vs full-block loss

**Date:** 2026-09-22 · Same pilot, same hyper-parameters, same 390-step budget. Only the loss mask
differs. Both adapters evaluated in ONE model load, on identical blocks, through the identical
packed-MMQ path, with base = freshly injected LoRA (B=0).

## The bug this tests
Every pilot row carries `loss_mask_roles: ["assistant"]` (ADR-004 designed it that way) but the
recipe's collator ignored the field and computed loss over the whole packed block. Measured on the
tokenised pilot, only **53.7% of tokens are assistant-authored** — so **46.3% of the gradient was
spent learning to predict prompts, tool output and system text the model never has to generate.**

Locating the assistant spans needed care. `return_assistant_tokens_mask` is unusable (the template
has no `{% generation %}` block; it returns an all-zero mask), and prefix rendering breaks because
the template appends an empty `<think></think>` to whichever assistant turn is *last* in the list it
is given — so a non-final assistant turn is not a prefix of the full render (11.9% of records).
Parsing the ChatML structure and mapping spans through the fast tokenizer's offset mapping handles
all of it: **0 fallbacks**.

## Results

| bucket | metric | base | unmasked | masked |
|---|---|---|---|---|
| targetA_heldout | full | 1.330 | **0.147** (-88.9%) | 0.497 (-62.6%) |
| targetA_heldout | **assistant** | 0.685 | 0.128 (-81.4%) | **0.119 (-82.6%)** |
| targetC_heldout | full | 3.777 | **0.432** (-88.6%) | 1.373 (-63.6%) |
| targetC_heldout | **assistant** | 0.895 | 0.275 (-69.3%) | **0.220 (-75.4%)** |
| tulu3_heldout | full | 2.855 | 1.267 (-55.6%) | 1.422 (-50.2%) |
| tulu3_heldout | **assistant** | 2.402 | 1.175 (-51.1%) | 1.174 (-51.1%) |

## Reading

1. **On full-block loss the unmasked adapter looks far better — and that is exactly the trap.** It
   was trained to predict prompts and tool text, so it wins a metric that includes them. This is the
   metric Gate 1 originally reported.
2. **On assistant-only loss — what the model actually has to generate — masking wins on both
   targets**: targetA 0.119 vs 0.128, targetC **0.220 vs 0.275 (20% better)**.
3. **targetC gains most, and the token census explains why**: targetC blocks are only **8.9%**
   assistant tokens, so the unmasked run spent over 90% of that bucket's gradient on tool-output
   JSON. Gate 1's headline "targetC 3.53 -> 0.34 (-90%)" was largely measuring tool-output
   memorisation, not reasoning.
4. **General capability is untouched**: tulu3_heldout assistant loss is a dead tie (1.1746 vs
   1.1736). The fix costs nothing.

## The global shift is real, and it survives
General never-trained assistant text still improves **-51.1%** in both runs. So a large part of the
gain is broad adaptation, not target learning. Subtracting it gives the target-specific excess:

| | targetA | targetC |
|---|---|---|
| unmasked | 30.3 pp | 18.2 pp |
| **masked** | **31.5 pp** | **24.3 pp** |

Masking improves the target-specific component, most clearly for targetC (+6.1 pp). But the honest
summary remains: roughly two thirds of the apparent improvement is a global shift shared with data
the model was never trained on.

## Consequences
- **Adopt assistant-only masking for Gate 2.** It is strictly better on the metric that matters,
  free on general data, and it removes a systematic inflation from every number we report.
- **Report assistant-only loss, not full-block loss.** Full-block loss on a bucket that is 91%
  prompt/tool text mostly measures format memorisation.
- Loss still cannot separate broad adaptation from capability. The agentic benchmark remains the
  measurement that can.
