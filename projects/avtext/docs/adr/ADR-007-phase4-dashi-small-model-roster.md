# ADR-007: Phase 4 on dashi — small-model roster + llama.cpp serving

- **Status:** Accepted
- **Date:** 2026-07-30
- **Builds on:** ADR-003 (target model), ADR-005 (compute inventory), ADR-006 (frozen eval)
- **Amends:** ADR-005's "MacBook eval + GMKtec candidate finetune" — Phase 4 now runs **entirely on dashi**

## Context

Two decisions converged. (1) Phase 4 (finetune) needs more compute headroom than the M5, and the GMKtec
(`dashi`, Strix Halo, 123 GiB) is the always-on box the user already serves models on. (2) The study is
strongest as a **small-model *size sweep***, not a single model — "does aviation decode scale with size, and
does finetuning rescue the small end?" is a better question than "does finetuning help E4B?"

`dashi` reality (probed 2026-07-30): Fedora 43, AMD Ryzen AI Max+ 395, GPU driven by **Vulkan** (ROCm not on
the host; it lives inside `kyuz0/amd-strix-halo-toolboxes` **toolbox containers**). Models served via
`toolbox run … llama-server` on **:8080** (the only firewall-open port; the M5 reaches it over the LAN).
`hf` present; `*-Unsloth` model dirs indicate an existing finetuning workflow.

## Decision

- **Run Phase 4 (train + eval) entirely on dashi**, served by **llama.cpp** (GGUF, Q8_0). Eval stays the frozen
  `eval/v1`; the M5 drives the harness over the LAN against `dashi:8080` (`~/serve.sh` swaps the served model).
- **Roster (size sweep), all Q8_0 GGUF:** Gemma 3 270M · Qwen2.5 0.5B · Gemma 3 1B · Gemma-3n E2B · Gemma-3n E4B.
  Gemma is the controlled ladder; Qwen-0.5B is one cross-family point (it earned its place — see below).
- **The finetune anchor is the dashi/llama.cpp/Q8 E4B baseline (69% value-acc), NOT the M5 MLX number (77.8%).**
  The two stacks differ by ~9 points on the same model; a paired before/after must not cross stacks.
- **Baseline eval re-done on dashi first** (this ADR's companion: `reports/size_sweep_v1.md`), before any training.

## Consequences

- **+** A self-consistent Phase-4 pipeline: same box, same serving, same eval for base and finetuned — the McNemar
  comparison is clean.
- **+** The sweep confirmed the core finding is *robust across five models*: the unit-conversion gap (inHg→hPa,
  m/s→kt) is **0–1% at every size, including 4B** — the finetuning lever is real and universal.
- **+** Discovered a genuine cross-family result: at ~0.5B, Gemma-270M is reckless (67% hallucination) while
  Qwen-0.5B is conservative (abstains) — hence keeping the Qwen point.
- **−** Serving-stack dependence: llama.cpp's Gemma-3n implementation may be lossy vs MLX (the 9-point E4B gap).
  Documented; does not affect the internal dashi before/after, but flagged for a chat-template ablation.
- **−** Displaces the user's Laguna server on :8080 during eval runs (reversible — `~/run-laguna-dflash-vulkan.sh`).
- **Open (next ADR):** the training stack. `dashi` is AMD → no CUDA/MLX; PyTorch needs ROCm. The user's Unsloth
  dirs suggest a working path; to be settled when planning the finetune.
