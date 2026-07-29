# eval/ — frozen evaluation sets

Unlike `data/`, these files **are committed** — they're small and freezing them
is the whole point.

- `v1/*.jsonl` — the frozen eval slices. Once frozen, they don't change; a new
  version means a new directory (`v2/`), never an edit in place.
- Each set ships with a **content hash** so a run report can prove exactly which
  bytes it was scored against.

Two rules that keep these honest (enforced in `src/avtext/tasks/`):

1. **Time split** — every item's observation time is strictly *after* the
   model's Jan-2025 data cutoff. No item the base model could have memorized.
2. **Station split** — stations used in test never appear in training.

- `v1/` — **frozen 2026-07-29** (620 records). Built by `harness/freeze.py` from the
  consensus-labelled corpus; split policy in ADR-006; details in `v1/README.md`.

Note: rules 1–2 below are realised with a small, data-forced twist (ADR-006). Rule 2
(station split) holds for the `unseen_station` slice; rule 1 (time split) is enforced
as a hard floor on *every* item, and additionally defines the `unseen_time` slice.
