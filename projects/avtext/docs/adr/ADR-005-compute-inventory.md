# ADR-005: Compute inventory — MacBook (eval) + GMKtec EVO-X2 (candidate finetune); no OVH

- **Status:** Accepted
- **Date:** 2026-07-29
- **Relates to:** ADR-003 (which assumed cloud fine-tune from the 24 GB MacBook)

## Context

ADR-003 set "fine-tune in the cloud, evaluate locally" based on a 24 GB MacBook.
A second, much more capable machine has since been confirmed — and the plan's
references to an OVH VPS are wrong: **there is no OVH VPS.**

## Inventory

| Machine | Specs | Role |
|---|---|---|
| MacBook (M5) | Apple M5, 24 GB unified, macOS, `llama-cli` present | Local GGUF **eval** (Q4 E4B ~4–5 GB fits easily); day-to-day dev |
| **GMKtec EVO-X2** | AMD Ryzen AI Max+ 395 "Strix Halo", 32 CPUs, Radeon 8060S (RDNA 3.5), **123 GiB unified RAM**, Fedora 43; `ssh dashi` (192.168.0.131) | Home server: heavy local inference; **candidate local fine-tune host** |

Installed on the GMKtec today: python3, git. **Not yet:** ROCm, llama.cpp, ollama,
docker.

## Decision

- **No OVH VPS** anywhere in this project. The bulk-data fallback host and any
  always-on service (e.g. the collector as a systemd timer) target the
  **GMKtec (`dashi`)** — a natural fit now that the repo is private (no GitHub
  Actions minutes spent).
- Keep ADR-003's default **for now** — fine-tune in the cloud, eval locally — but
  treat the GMKtec as a **first-class candidate for local fine-tuning**, decided in
  Phase 4. Its 123 GiB unified memory makes even bf16 LoRA of E4B comfortable; the
  open question is tooling maturity (AMD **ROCm** PEFT is less turnkey than
  CUDA/Unsloth). Order of operations: prove local inference (llama.cpp via
  ROCm/Vulkan) first, then evaluate a local fine-tune path.

## Consequences

- **+** A powerful, always-on, no-hourly-cost box for inference and possibly training.
- **+** The collector can run on `dashi` instead of GitHub Actions.
- **−** AMD/ROCm setup is required before the GPU is usable for ML; until then it's a
  strong CPU/inference box only.
- Revisit ADR-003's cloud-finetune default in Phase 4 with a measured local-vs-cloud
  comparison.
