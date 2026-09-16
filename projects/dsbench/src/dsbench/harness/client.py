"""Talk to the model over an OpenAI-compatible HTTP endpoint (llama-server, vLLM, any).

Same seam as avtext: the harness never imports an inference library, it just POSTs to
/v1/chat/completions. Point --base-url at dashi's llama-server and the served fine-tune plugs
into the exact same path as the base model, so before/after runs are comparable by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Completion:
    """A model reply plus the metadata needed to tell 'answered badly' from 'never answered'.

    finish_reason == 'length' with no extractable code is a thinking model that spent its whole
    token budget in the reasoning channel -- a serving/config artifact, not a wrong answer. The
    runner uses this to mark such attempts `truncated` instead of `no_code`. reasoning_chars is
    kept for debugging (how much the model reasoned) without storing the whole trace.
    """

    content: str
    finish_reason: str = ""
    reasoning_chars: int = 0


class ChatClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "served",
        *,
        temperature: float = 0.0,  # deterministic decode -> a reproducible score
        top_p: float = 1.0,
        max_tokens: int = 2048,
        timeout: float = 300.0,
        system: str | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system = system
        # Server-side chat-template switches, e.g. {"enable_thinking": False} to turn off a
        # reasoning model's thinking (avoids runaway-reasoning truncation on this fork, which
        # honors chat_template_kwargs but ignores the per-request `reasoning_effort` field).
        self.chat_template_kwargs = chat_template_kwargs
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def complete(self, prompt: str) -> Completion:
        """Return the reply + finish metadata; raises httpx.HTTPError on a transport error."""
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        resp = self._client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        msg = choice.get("message", {}) or {}
        return Completion(
            content=msg.get("content") or "",
            finish_reason=choice.get("finish_reason") or "",
            # Some servers (this fork included) return reasoning in a separate channel.
            reasoning_chars=len(msg.get("reasoning_content") or ""),
        )
