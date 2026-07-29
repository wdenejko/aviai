# avtext — task runner.
#
# uv commands work from anywhere in the workspace; the `--package avtext` flag
# pins them to this project. Targets that aren't built yet fail loudly with a
# pointer to the phase where they get implemented, so this file doubles as a
# map of the work ahead.

.DEFAULT_GOAL := help
.PHONY: help sync test lint fmt probe backfill collect harness report

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ── Works today ──────────────────────────────────────────────────────────────
sync:  ## Create/update the workspace .venv from the lockfile
	uv sync

test:  ## Run tests
	uv run pytest

lint:  ## Lint with ruff
	uv run ruff check .

fmt:  ## Auto-format + fix with ruff
	uv run ruff check --fix .
	uv run ruff format .

probe:  ## [Phase 1] Scope probe: parser failure/disagreement rate on a live AWC sample
	uv run python -m avtext.probe

# ── Coming in later phases (fail loudly until implemented) ───────────────────
backfill:  ## [Phase 1] One-off, idempotent, checksummed history pull (IEM)
	@echo "Not implemented yet — Phase 1 (ingest/iem.py). See docs/research-repo-plan.md §2.1" && exit 1

collect:  ## [Phase 1] Snapshot current AWC caches (raw + official decode)
	@echo "Not implemented yet — Phase 1 (ingest/awc.py). See docs/research-repo-plan.md §2.2" && exit 1

harness:  ## [Phase 3] Score a model against frozen eval/v1 -> reports/runs/<id>/
	@echo "Not implemented yet — Phase 3 (harness/). Usage will be: make harness MODEL=..." && exit 1

report:  ## [Phase 3] Render the before/after markdown report
	@echo "Not implemented yet — Phase 3 (harness/report.py)." && exit 1
