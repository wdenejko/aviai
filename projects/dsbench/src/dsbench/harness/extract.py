"""Pull runnable code out of a chat completion.

Models wrap answers in prose, in <think> blocks, and in Markdown fences. We strip reasoning,
take the LAST fenced block of the right language (last, because models often show a wrong first
attempt then a corrected final one), and fall back to the whole message when there is no fence.
"""

from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> spans (Qwen thinking mode) before we look for code."""
    return _THINK.sub("", text)


def extract_code(text: str, mode: str) -> str:
    """Return the model's code for the given mode ('python' or 'sql'), or '' if none found."""
    body = strip_reasoning(text or "")
    blocks = _FENCE.findall(body)

    if mode == "sql":
        langs = {"sql", "duckdb", ""}
    else:
        langs = {"python", "py", "python3", ""}

    # Prefer the last explicitly-tagged block of the right language; then any last block.
    tagged = [code for lang, code in blocks if lang.lower() in langs and lang != ""]
    if tagged:
        return tagged[-1].strip()
    if blocks:
        return blocks[-1][1].strip()

    # No fence at all. For SQL, salvage from the first SELECT/WITH; else return the trimmed text.
    if mode == "sql":
        m = re.search(r"(?is)\b(with|select)\b.*", body)
        return m.group(0).strip() if m else ""
    return body.strip()
