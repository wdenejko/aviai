# reports/runs/ — one directory per scored run

Committed. Each `<run_id>/` holds the results JSON + `report.md`.

**Reproducibility contract** — every report records enough to re-derive any
number from a clean clone:

- model hash (base checkpoint or GGUF)
- adapter hash (for fine-tuned runs)
- prompt-template id
- decode params (temperature, constrained-decoding on/off, seed)
- eval-set content hash

If two people (or future-you) run the same `run_id`, they get the same numbers.
That property is the difference between a benchmark and a vibe.
