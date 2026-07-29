"""Model backends behind the Predictor contract (Phase 3 runner).

CI-safe by design: the harness reaches a model through an OpenAI-compatible HTTP endpoint
via httpx (already a dependency). WHICH server answers — `mlx_vlm.server` for the local
Gemma-4-E4B, ollama, llama-server, LM Studio, or a frontier API — is not the harness's
concern; only the endpoint is. That keeps Mac-only inference libraries (mlx) out of the
installed package, so CI on Linux stays green and the finetuned model later plugs into the
exact same seam.

`predict(raw)` returns the canonical field dict (via prompt.parse_prediction) or None when
the server errors or the output has no usable JSON — which the scorer reads as abstention.
"""

from __future__ import annotations

import httpx

from avtext.harness.prompt import format_prompt, parse_prediction


def http_predictor(
    base_url: str,
    model: str,
    *,
    temperature: float = 0.0,  # deterministic decode — the eval must be reproducible
    max_tokens: int = 256,
    timeout: float = 180.0,
):
    """A Predictor that POSTs to an OpenAI-compatible /v1/chat/completions endpoint."""
    client = httpx.Client(base_url=base_url, timeout=timeout)

    def predict(raw: str) -> dict | None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": format_prompt(raw)}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return parse_prediction(text)

    return predict
