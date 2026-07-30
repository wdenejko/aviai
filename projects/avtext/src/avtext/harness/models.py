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

# Gemma-4's chat turn wrapper, exactly as Unsloth's tokenizer renders it at train time.
# A hard-finetuned model overfits to this precise string; see completion_predictor for why
# reproducing it byte-for-byte (rather than trusting the server's --jinja render) matters.
GEMMA4_TURN_WRAP = "<|turn>user\n{}<turn|>\n<|turn>model\n"


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


def completion_predictor(
    base_url: str,
    wrapper: str = GEMMA4_TURN_WRAP,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: float = 180.0,
):
    """A Predictor that POSTs a pre-templated prompt to the raw `/completion` endpoint.

    Why this exists: the chat path (`/v1/chat/completions` + `--jinja`) makes llama.cpp render
    the model's chat template with its own engine (minja), which can differ subtly from how HF
    Transformers rendered the SAME template at training. A base model tolerates the drift; a hard
    LoRA-finetuned model (train loss ~0.06) overfits to the exact training prompt and falls off
    distribution — emitting turn-delimiter garbage instead of JSON. Reproducing the training
    wrapper byte-for-byte here and sending it raw sidesteps the server's template engine entirely,
    so serve == train. The server still prepends BOS. `wrapper` has one `{}` for format_prompt."""
    client = httpx.Client(base_url=base_url, timeout=timeout)

    def predict(raw: str) -> dict | None:
        resp = client.post(
            "/completion",
            json={
                "prompt": wrapper.format(format_prompt(raw)),
                "n_predict": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return parse_prediction(resp.json().get("content", ""))

    return predict
