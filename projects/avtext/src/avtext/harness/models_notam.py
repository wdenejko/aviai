"""NOTAM model backend — a raw /completion caller for both NOTAM tasks (Phase 7).

Same seam + template discipline as models_taf, but NOTAM needs the PROMPT to vary per record
(extraction is category-aware; classification isn't), so this exposes a low-level `complete(prompt)`
rather than a fixed `predict(raw)` — the runner builds the right prompt per task/category.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx

from avtext.harness.models import GEMMA4_TURN_WRAP


def notam_completer(
    base_url: str, *, wrapper: str = GEMMA4_TURN_WRAP, temperature: float = 0.0,
    max_tokens: int = 512, timeout: float = 300.0, cache_prompt: bool = False,
) -> Callable[[str], str]:  # fmt: skip
    """Return `complete(prompt_body) -> text`: wraps the body in the byte-for-byte training turn
    wrapper and POSTs to the raw /completion endpoint (sidesteps minja drift for hard finetunes).

    cache_prompt defaults False for NOTAM: prompts change prefix whenever the category changes, so
    reusing the server's prompt cache across requests thrashes the KV and — observed on this ROCm
    build over a long extraction run — eventually corrupts it into <unused49> reserved-token spam.
    False makes every request reprocess its prompt fresh (a little slower, but stable)."""
    client = httpx.Client(base_url=base_url, timeout=timeout)

    def complete(prompt_body: str) -> str:
        resp = client.post(
            "/completion",
            json={
                "prompt": wrapper.format(prompt_body),
                "n_predict": max_tokens,
                "temperature": temperature,
                "cache_prompt": cache_prompt,
            },
        )
        resp.raise_for_status()
        return resp.json().get("content", "")

    return complete
