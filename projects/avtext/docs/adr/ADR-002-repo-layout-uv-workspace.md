# ADR-002: aviai is a uv-workspace sandbox; avtext is its first subproject

- **Status:** Accepted
- **Date:** 2026-07-29

## Context

The master plan assumed a single-purpose repo (`avtext-lab`). But `aviai` is
intended as a **multi-topic learning sandbox** — aviation-text today; RAG and
other fine-tuning studies later. Two goals pull against each other: each topic
wants its own dependencies and decision history (self-contained), yet the sandbox
should be one thing to clone, one environment to activate.

## Decision

Structure `aviai` as a **[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)**:

- Root `pyproject.toml` declares `[tool.uv.workspace] members = ["projects/*"]`
  and is **not itself a package** — it only owns shared dev tooling (ruff,
  pytest, hypothesis) and the single lockfile.
- Each topic is a self-contained package under `projects/`. The first is
  `projects/avtext/` (package `avtext`), following the plan's blueprint.
- Directory named `avtext` (not `avtext-weather`/`metar-*`): the plan warns to
  keep the name neutral to the whole aviation-text scope, since NOTAM/PIREP
  arrive later.

## Consequences

- **+** One `uv sync`, one `.venv`, consistent tooling across every topic.
- **+** New topic = new `projects/<x>/`; nothing else moves.
- **−** Workspace members share a single resolved dependency set — two projects
  cannot pin conflicting versions of the same library. Acceptable for a personal
  sandbox; escape hatch is to split a project into its own repo if it ever bites.
- The master plan was moved to `projects/avtext/docs/research-repo-plan.md` to
  keep the subproject self-contained.

## Alternatives considered

- *One shared `src/` for all topics (monorepo).* Rejected: couples unrelated
  learning projects and their deps.
- *Separate repo per topic.* Rejected: loses the single-clone, single-env
  sandbox ergonomics that motivated `aviai`.
