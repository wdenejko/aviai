"""Turn a chat reply into the thing each benchmark scores: code, a letter, a SQL query, tool calls.

Where a benchmark publishes its own extraction rule, it is copied here verbatim and says so (DS-1000
`postprocess`, LiveCodeBench `extract_code`, the MMLU-Pro regex chain). Where it doesn't, the rule
is ours and is written to be lenient in the same way for both states. That keeps the paired delta
fair even where the rule differs from someone else's harness.
"""

from __future__ import annotations

import ast
import json
import re

_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop a leading reasoning block if the template leaks one into `content` (thinking off)."""
    return _THINK.sub("", text or "", count=1)


def fenced_blocks(text: str, langs: tuple[str, ...] = ()) -> list[str]:
    """All ``` fenced blocks in order; restricted to `langs` (case-insensitive) when given."""
    out = []
    for lang, body in _FENCE.findall(text):
        if not langs or lang.lower() in langs:
            out.append(body)
    return out


# --- HumanEval+ -------------------------------------------------------------------------------


def _prompt_imports(prompt: str) -> str:
    return "\n".join(
        line for line in prompt.splitlines() if re.match(r"^(import |from \S+ import )", line)
    )


def _sanitize(code: str) -> str:
    """Keep definitions and imports; drop top-level demo code, like evalplus' `sanitize`.

    A chat model often appends `if __name__ == "__main__":` examples or bare `print(...)` /
    `assert` lines. Run as a script, those execute before the tests and can crash or hang a correct
    solution. Anything that doesn't parse is returned unchanged and fails honestly at run time.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
            ast.Assign, ast.AnnAssign)
    tree.body = [node for node in tree.body if isinstance(node, keep)]
    return ast.unparse(tree)


def humaneval_program(content: str, prompt: str, entry_point: str) -> str:
    """The solution module for one HumanEval+ problem (tests are appended by the scorer).

    Take the fenced block that defines the entry point (else the longest fenced block, else the raw
    reply). If the model wrote only a function body, fall back to prompt + body, which is the
    completion-style reading. The prompt's own imports (`from typing import List`) are prepended
    because chat models routinely drop them while keeping the annotations that need them.
    """
    text = strip_think(content)
    blocks = fenced_blocks(text, ("python", "py", "python3", "")) or [text]
    defining = [b for b in blocks if re.search(rf"^\s*def {re.escape(entry_point)}\s*\(", b, re.M)]
    code = defining[0] if defining else max(blocks, key=len)
    if not re.search(rf"^\s*def {re.escape(entry_point)}\s*\(", code, re.M):
        code = prompt + code
    return _prompt_imports(prompt) + "\n" + _sanitize(code) + "\n"


# --- DS-1000 (official postprocess, test_ds1000.py @ xlang-ai/DS-1000) --------------------------


def ds1000_code(content: str) -> str:
    code = strip_think(content)
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.split("\nEND SOLUTION")[0]
    code = code.replace("<code>", "")
    return code


# --- LiveCodeBench (official extract_code, lcb_runner/utils/extraction_utils.py) ----------------


def lcb_code(content: str) -> str:
    lines = strip_think(content).split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


# --- multiple choice ---------------------------------------------------------------------------


def mmlu_pro_letter(content: str) -> str | None:
    """MMLU-Pro's own chain (evaluate_from_api.py): 'answer is (X)', then 'Answer: X', then the
    last standalone capital A-J. First match wins at each stage, as upstream."""
    text = strip_think(content)
    m = re.search(r"answer is \(?([A-J])\)?", text)
    if m:
        return m.group(1)
    m = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if m:
        return m.group(1)
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return m.group(0) if m else None


def gpqa_letter(content: str) -> str | None:
    """simple-evals' 'Answer: X' pattern, taking the LAST occurrence (a CoT may quote an option
    as 'Answer: A' before settling). Same rule for both states."""
    hits = re.findall(r"(?i)Answer[ \t]*:[ \t]*\$?\(?([A-D])\)?\$?", strip_think(content))
    return hits[-1].upper() if hits else None


# --- BIRD --------------------------------------------------------------------------------------


def bird_sql(content: str) -> str:
    """The last ```sql block, else the last fenced block, else the reply itself."""
    text = strip_think(content)
    blocks = fenced_blocks(text, ("sql", "sqlite")) or fenced_blocks(text)
    sql = blocks[-1] if blocks else text
    return sql.strip()


# --- BFCL --------------------------------------------------------------------------------------


def bfcl_calls(tool_calls: list[dict]) -> tuple[list[dict] | None, str]:
    """OpenAI tool_calls -> BFCL's decoded form [{name: {arg: value}}], or (None, why)."""
    out = []
    for call in tool_calls or []:
        fn = call.get("function") or {}
        name, raw = fn.get("name"), fn.get("arguments") or "{}"
        if not name:
            return None, "tool call without a function name"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as e:
            return None, f"arguments are not valid JSON: {e.msg}"
        if not isinstance(args, dict):
            return None, "arguments are not a JSON object"
        out.append({name: args})
    return out, ""
