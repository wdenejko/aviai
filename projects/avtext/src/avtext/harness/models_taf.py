"""TAF model backends behind the Predictor contract (Phase 6 runner).

Mirror of harness/models.py for TAF: same OpenAI-compatible-HTTP seam, same CI-safety (no
inference libs in the package), but wired to the TAF prompt/parser and a larger token budget —
a decoded TAF is a nested header + many change groups, so 256 tokens (fine for METAR) truncates
it; 1024 is the default. `completion_predictor_taf` exists for the same reason as its METAR twin:
a hard-finetuned Gemma-4 overfits the exact training wrapper, so we send it byte-for-byte to the
raw /completion endpoint rather than trusting the server's chat-template render.
"""

from __future__ import annotations

import httpx

from avtext.harness.models import GEMMA4_TURN_WRAP
from avtext.harness.prompt_taf import format_prompt_taf, parse_prediction_taf


def http_predictor_taf(
    base_url: str, model: str, *, temperature: float = 0.0, max_tokens: int = 1024,
    timeout: float = 300.0,
):  # fmt: skip
    """A TAF Predictor that POSTs to an OpenAI-compatible /v1/chat/completions endpoint."""
    client = httpx.Client(base_url=base_url, timeout=timeout)

    def predict(raw: str) -> dict | None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": format_prompt_taf(raw)}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        return parse_prediction_taf(resp.json()["choices"][0]["message"]["content"])

    return predict


def completion_predictor_taf(
    base_url: str, wrapper: str = GEMMA4_TURN_WRAP, *, temperature: float = 0.0,
    max_tokens: int = 1024, timeout: float = 300.0,
):  # fmt: skip
    """A TAF Predictor that POSTs the byte-for-byte training wrapper to the raw /completion endpoint
    (sidesteps minja template drift for hard-finetuned models — see models.completion_predictor)."""
    client = httpx.Client(base_url=base_url, timeout=timeout)

    def predict(raw: str) -> dict | None:
        resp = client.post(
            "/completion",
            json={
                "prompt": wrapper.format(format_prompt_taf(raw)),
                "n_predict": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return parse_prediction_taf(resp.json().get("content", ""))

    return predict
