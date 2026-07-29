# aviai

A personal **learning sandbox for applied LLM work** — building datasets, eval
harnesses, fine-tuning, RAG, and whatever else is worth learning by doing.

It's structured as a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
one shared environment and lockfile at the root, one self-contained package per
topic under `projects/`.

## Projects

| Project | What it is | Status |
|---|---|---|
| [`avtext`](projects/avtext/) | Aviation-text LLM lab: build a trustworthy METAR/TAF decoding dataset, a benchmark/eval harness, then measure whether LoRA fine-tuning a small open model (Gemma 4 E4B) beats the base model on aviation-specific tasks. | 🚧 scaffolding |

## Getting started

```bash
uv sync              # creates .venv with Python 3.12 + all workspace deps
uv run pytest        # run tests across all projects
uv run ruff check .  # lint
```

Each project has its own `README.md`, `docs/adr/` (decision records), and
`docs/LEARNING_LOG.md` (the running log of what was learned — the real output of
a learning project).
