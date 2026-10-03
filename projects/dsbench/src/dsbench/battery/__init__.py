"""The ADR-001 before/after acceptance battery: public benchmarks, base vs adapter, paired per item.

Gate 2 accepts the adapter only if it gains on the target domain (DS-1000, BIRD-dev, the owner's
dsbench suite, tool-call format tests) without regressing general skills (IFEval, MMLU-Pro, GPQA:
<= 1 point; LiveCodeBench, HumanEval+: <= 2 points). This package runs that battery.

Design choices, and why:

* ONE server, TWO states. llama-server loads the base GGUF plus the LoRA GGUF once and every request
  names its state (`"lora": [{"id": 0, "scale": 0 | 1}]`). Base and adapter therefore share the
  binary, the quant, the chat template and the sampler, so the only difference between the two
  columns is the adapter itself. A pass runs one state only, so no batch ever mixes the two. (ADR
  step 4 adds a third, `adapter_half`: the adapter at scale 0.5, paired with the same base pass.)
* Greedy decoding, thinking off. Greedy makes each item's outcome a property of the model and not
  of a sampling seed, which is what the paired statistics below assume. Thinking stays off because
  that is how the adapter was trained. The price is that absolute scores sit below Qwen's published
  thinking-mode numbers; the report says so rather than comparing against them.
* Paired statistics. Each item is scored for both states and compared with an exact McNemar test on
  the discordant pairs. Pairing removes the between-item variance that makes an unpaired ~200-item
  comparison noisy (ADR-001: ~3 points of 1-sigma noise).
* An A/A control. A second base pass (`base_rep`) measures how often the harness itself flips an
  item (batching changes float summation order, which can change a greedy token). Adapter flips
  are only meaningful above that floor.
* Reference scorers, not re-implementations. IFEval uses Google's `instruction_following_eval`,
  LiveCodeBench its own `testing_util.run_test`, BFCL its own `ast_checker`, and DS-1000 and
  HumanEval+ their published test programs. Each is fetched at a pinned commit (see `prepare.py`).
  Only the glue (prompts, answer extraction, bookkeeping) is ours, and that glue is unit-tested.
* Model code never runs on the host. Anything that executes generated code (HumanEval+, DS-1000,
  LiveCodeBench, BIRD) runs in a rootless podman container with no network, a read-only root
  filesystem and memory/pid caps (`sandbox_exec.py`, `sandbox/battery/Containerfile`).

Pipeline: `prepare` (datasets -> items) -> `generate` (items x state -> generations) -> `score`
(generations -> per-item pass/fail) -> `stats`/`report` (paired deltas + ADR verdicts).

ADR-004 Revision 2 is served with thinking on, so its checks sample with thinking on instead: the
mini-battery on checkpoints (`mini`), and the whole battery on the base and the final candidate
(`full`), each item with the same seed in every state but the A/A pass.
"""
