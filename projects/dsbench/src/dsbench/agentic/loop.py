"""The tool-calling ReAct loop: drive the served model against the sandbox until it finishes,
then grade the final state.

Protocol = native OpenAI tool-calling (verified working on the fork with Ornith): we send `tools`,
the model returns `tool_calls`, we execute them and feed back `tool` messages, and loop. Thinking is
OFF by default (memory `reference-fork-thinking-eval`: at temp 0 a thinking model can burn the whole
budget reasoning and never act). Determinism: temp 0, per-problem scratch DB, final-state grading.
"""
from __future__ import annotations

import json
import time

import httpx

from dsbench.agentic import tools as T
from dsbench.agentic.ch import get_client
from dsbench.agentic.schema import AgentProblem, AgentResult, GradeContext

SCHEMA_DOC = """Warehouse schema (shared database `aviation`):
- aviation.flights — one row per US flight, June 2026. Columns include:
    FlightDate (Date), Reporting_Airline (String), Origin (String, IATA), Dest (String, IATA),
    DepDelayMinutes (Float, dep delay minutes; NULL when cancelled), ArrDelayMinutes (Float),
    Cancelled (UInt8 0/1), Diverted (UInt8 0/1), Distance (Float miles),
    CarrierDelay/WeatherDelay/NASDelay/SecurityDelay/LateAircraftDelay (Float minutes).
- aviation.metar — raw METAR obs, June 2026, hub airports. Columns:
    station (String, ICAO e.g. 'KORD'), valid_utc (DateTime UTC), raw (String, raw METAR text).
- aviation.airports — dimension: iata (String), icao (String), name (String).
    Join flights.Origin/Dest (IATA) -> airports.iata; airports.icao -> metar/taf.station.
- aviation.taf — TAF forecasts for the hubs, June 2026, ONE ROW PER FORECAST PERIOD. Columns:
    station (ICAO), issued (DateTime, bulletin issue time), fx_from/fx_to (DateTime, the period),
    ftype ('Observation'|'Forecast'), is_tempo/is_amendment (UInt8), sknt/drct/gust (Float, wind),
    visibility (Float, SM), skyc/skyl (text lists), raw (change-group text), product_id (bulletin).
    count(DISTINCT product_id) = number of TAF bulletins (a bulletin spans many rows).
- aviation.notam — DEEL-AI NOTAM classification corpus. Columns: text (NOTAM E-field), label_id
    (0-12), category (class name), split ('train'|'test'), source. STATIC ~2024 corpus, NOT
    date-aligned and with no airport/time key — use for NOTAM text/classification, not joins."""

SYSTEM = """You are a data engineer/analyst operating a real aviation data warehouse (ClickHouse) \
to complete a task. Tools: run_sql (one ClickHouse statement), run_python (analysis container with \
pandas/numpy/sklearn/clickhouse_connect), finish (end the task).

Your scratch database is `{namespace}` and is the DEFAULT for run_sql, so `CREATE TABLE \
t ...` lands there. Shared data is in `aviation.*` — always query it fully-qualified. Work \
step by step: inspect, compute, and produce EXACTLY the deliverable asked for, then call \
finish (pass the value in `answer` if the task asks a question). You have a limited number \
of steps, so be efficient. Keep each run_python call to one focused, self-contained script; \
if a step would need a very long code block, split it across calls — over-long tool arguments \
can be truncated.

{schema}"""


def _reassemble_stream(r: httpx.Response) -> dict:
    """Reassemble a streamed chat completion into one non-stream-shaped `choice` dict.

    WHY stream: the fork's NON-stream endpoint runs a strict json::parse over the model's tool-call
    arguments and 500s the whole request if the model emitted one malformed/truncated tool call deep
    in a trajectory (empirically what kills ds_notam_classify ~step 15 — NOT a double-quote bug; a
    controlled probe showed quotes and 4KB args parse fine). Streaming delivers the arguments as
    deltas we concatenate ourselves, so a malformed call comes back to us as text we can turn into a
    recoverable tool error instead of aborting the run. See reference-ornith-agentic-behavior.
    """
    content = ""
    finish = None
    calls: dict[int, dict] = {}  # index -> {id, name, arguments}
    for line in r.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            continue  # keep-alive / non-JSON line
        ch = (ev.get("choices") or [{}])[0]
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
        delta = ch.get("delta") or {}
        if delta.get("content"):
            content += delta["content"]
        for tc in (delta.get("tool_calls") or []):
            idx = tc.get("index", 0)
            slot = calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function", {}) or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            slot["arguments"] += fn.get("arguments") or ""
    tool_calls = [
        {"id": c["id"], "type": "function",
         "function": {"name": c["name"], "arguments": c["arguments"]}}
        for _, c in sorted(calls.items())
    ]
    return {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
            "finish_reason": finish}


def call_model(messages: list, *, base_url: str, model: str, no_think: bool,
               temperature: float = 0.0, max_tokens: int = 4096, timeout: float = 300.0,
               attempts: int = 3, stream: bool = True) -> dict:
    # Generous max_tokens: the model emits run_python CODE as a tool-call argument. We stream by
    # default and reassemble tool-calls ourselves (see _reassemble_stream) so a single malformed
    # tool-call is a recoverable observation, not a server-side 500 that ends the whole run.
    body: dict = {
        "model": model, "messages": messages, "tools": T.TOOL_SCHEMAS, "tool_choice": "auto",
        "temperature": temperature, "max_tokens": max_tokens, "stream": stream,
    }
    if no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    last: Exception | None = None
    for i in range(attempts):
        try:
            if not stream:
                r = httpx.post(f"{base_url}/v1/chat/completions", json=body, timeout=timeout)
                r.raise_for_status()
                return r.json()["choices"][0]
            with httpx.stream("POST", f"{base_url}/v1/chat/completions", json=body,
                              timeout=timeout) as r:
                r.raise_for_status()
                return _reassemble_stream(r)
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:  # a 4xx won't fix on retry
                raise
            last = e
        except httpx.TransportError as e:  # transient network/server hiccup mid-run
            last = e
        time.sleep(2 * (i + 1))
    raise last  # type: ignore[misc]


def prepare_context(problem_id: str) -> GradeContext:
    """Drop+create a clean per-problem scratch database and return a context bound to it."""
    ns = f"prob_{problem_id}"
    admin = get_client(database="aviation")
    admin.command(f"DROP DATABASE IF EXISTS {ns}")
    admin.command(f"CREATE DATABASE {ns}")
    return GradeContext(client=get_client(database=ns), namespace=ns)


def grade(problem: AgentProblem, ctx: GradeContext) -> tuple[bool, str, str]:
    """Run the independent checker over the final state. Returns (passed, status, reason)."""
    try:
        res = problem.check(ctx)
    except Exception as e:  # noqa: BLE001 — a buggy checker must fail loud, not pass silently
        return False, "error", f"checker raised: {type(e).__name__}: {str(e)[:300]}"
    passed, reason = (res if isinstance(res, tuple) else (bool(res), ""))
    return bool(passed), ("ok" if passed else "wrong"), reason


def run_agent(problem: AgentProblem, ctx: GradeContext, *, base_url: str, model: str,
              no_think: bool = True, verbose: bool = False) -> AgentResult:
    t0 = time.time()
    messages: list = [
        {"role": "system", "content": SYSTEM.format(namespace=ctx.namespace, schema=SCHEMA_DOC)},
        {"role": "user", "content": problem.prompt},
    ]
    n_tool_calls = 0

    def result(passed, status, reason, steps):
        return AgentResult(problem.id, problem.category, problem.difficulty, passed, status,
                           reason=reason, steps=steps, tool_calls=n_tool_calls, answer=ctx.answer,
                           trajectory=messages, latency_s=round(time.time() - t0, 1))

    for step in range(1, problem.max_steps + 1):
        try:
            choice = call_model(messages, base_url=base_url, model=model, no_think=no_think)
        except Exception as e:  # noqa: BLE001
            return result(False, "model_error", f"{type(e).__name__}: {str(e)[:200]}", step)
        msg = choice.get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        if verbose:
            print(f"    step {step}: {len(tool_calls)} tool_call(s)"
                  + (f'; says: {content[:80]!r}' if content and not tool_calls else ""))

        if not tool_calls:
            if not content.strip() and choice.get("finish_reason") == "length":
                return result(False, "truncated", "model hit token cap with no answer/tool", step)
            ctx.answer = content.strip()  # final answer given in prose
            passed, status, reason = grade(problem, ctx)
            return result(passed, status, reason, step)

        # Parse each tool call's arguments up front. A malformed one (truncated / bad escaping —
        # the fine-tune target) must be RECOVERABLE, not fatal: we (a) put a sanitized, valid-JSON
        # copy into history so the server can re-render the turn without 500ing on the next request,
        # and (b) hand the model a tool error asking it to resend. parsed[i] is None iff malformed.
        parsed: list[dict | None] = []
        for tc in tool_calls:
            try:
                parsed.append(json.loads((tc.get("function", {}) or {}).get("arguments") or "{}"))
            except json.JSONDecodeError:
                parsed.append(None)
        safe_calls = [
            {"id": tc.get("id"), "type": "function",
             "function": {"name": (tc.get("function", {}) or {}).get("name"),
                          "arguments": json.dumps(a if a is not None else {})}}
            for tc, a in zip(tool_calls, parsed, strict=True)
        ]
        messages.append({"role": "assistant", "content": content, "tool_calls": safe_calls})
        for tc, args in zip(tool_calls, parsed, strict=True):
            n_tool_calls += 1
            fn = tc.get("function", {}) or {}
            name = fn.get("name")
            if args is None:
                obs = ("tool-call error: your `arguments` were not valid JSON (likely truncated or "
                       "an unescaped quote in a long value). Resend this call with valid JSON; if "
                       "the code was long, split it into smaller run_python calls.")
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": obs})
                continue
            if name == "finish":
                ans = args.get("answer")
                if ans is None or (isinstance(ans, str) and not ans.strip()):
                    ans = content.strip() or None  # no answer arg; fall back to prose
                ctx.answer = ans
                passed, status, reason = grade(problem, ctx)
                return result(passed, status, reason, step)
            if name == "run_sql":
                obs = T.run_sql(ctx.client, args.get("query", ""))
            elif name == "run_python":
                obs = T.run_python(args.get("code", ""), ctx.namespace)
            else:
                obs = f"error: unknown tool {name!r}"
            messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": obs})

    # Step budget exhausted without an explicit finish — grade whatever state exists (the agent may
    # have built the deliverable table without calling finish).
    passed, status, reason = grade(problem, ctx)
    if not passed and not reason:
        reason = f"step budget ({problem.max_steps}) exhausted without finishing"
    return result(passed, status, reason, problem.max_steps)
