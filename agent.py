"""The agent loop — the heart of the harness."""

import json
import time
from collections import Counter
from collections.abc import Callable

SYSTEM_PROMPT = """You are a careful coding assistant working inside a sandboxed workspace.
Use the tools to inspect before you change anything. When a tool returns ERROR,
read it and fix your call instead of repeating it. When the task is done, answer
with a short summary and do not call more tools."""


def assistant_to_dict(msg) -> dict:
    """Turn the assistant reply back into a message dict for the next request.

    Gemini (thinking models) attaches `extra_content.google.thought_signature`
    to each tool call. It MUST be sent back unchanged, or the next request fails
    with 400 "Function call is missing a thought_signature". model_dump() keeps
    such non-standard fields; rebuilding the dict by hand would drop them.
    """
    if hasattr(msg, "model_dump"):
        d = msg.model_dump(exclude_none=True)
    else:  # plain objects, e.g. the fake LLM in tests
        d = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                    **(
                        {"extra_content": tc.extra_content}
                        if getattr(tc, "extra_content", None)
                        else {}
                    ),
                }
                for tc in msg.tool_calls
            ]
    d["role"] = "assistant"
    d["content"] = d.get("content") or ""
    return d


def call_key(name: str, raw_args: str, result: str) -> tuple[str, str, str]:
    """Identity of a tool call for loop detection.

    Args are re-serialized with sorted keys so '{"a":1,"b":2}' and
    '{"b": 2, "a": 1}' count as the same call. The result is part of the key:
    re-running `pytest` after an edit (different output) is progress, not a loop.
    """
    try:
        args = json.dumps(json.loads(raw_args or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        args = raw_args
    return name, args, result


def run_agent(
    task: str,
    llm,
    registry,
    max_steps: int = 15,
    on_event: Callable[[str, dict], None] = lambda kind, data: None,
    loop_limit: int = 3,
) -> str:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    tools = registry.schemas()
    seen: Counter = Counter()  # call_key -> how many times it happened
    on_event("start", {"messages": messages, "tools": [t["function"]["name"] for t in tools]})

    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        msg = llm.chat(messages, tools)
        latency = time.perf_counter() - t0
        tool_calls = msg.tool_calls or []

        messages.append(assistant_to_dict(msg))
        on_event(
            "llm_call",
            {
                "step": step,
                "latency_s": round(latency, 3),
                "usage": getattr(llm, "last_usage", None),  # None for the fake LLM
                "message": messages[-1],
            },
        )

        if not tool_calls:  # model answered without tools -> done
            on_event("final", {"step": step, "content": msg.content})
            return msg.content or ""

        for tc in tool_calls:
            on_event(
                "tool_call",
                {
                    "step": step,
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": tc.function.arguments,
                },
            )
            t0 = time.perf_counter()
            result = registry.run(tc.function.name, tc.function.arguments)
            on_event(
                "tool_result",
                {
                    "step": step,
                    "id": tc.id,
                    "name": tc.function.name,
                    "result": result,
                    "latency_s": round(time.perf_counter() - t0, 3),
                },
            )
            key = call_key(tc.function.name, tc.function.arguments, result)
            seen[key] += 1
            content = result
            if seen[key] == loop_limit - 1:  # one repeat left: nudge before stopping
                # Appended to the tool result (not a separate user message) so the
                # assistant -> tool message order the API requires stays intact.
                content += (
                    f"\n\nWARNING: you have called {key[0]} with these exact arguments "
                    f"{seen[key]} times and got the same result each time. Calling it "
                    "again will stop the run. Try a different approach, or give your "
                    "final answer if you have enough information."
                )
                on_event(
                    "loop_warning",
                    {"step": step, "name": key[0], "args": key[1], "count": seen[key]},
                )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

            if seen[key] >= loop_limit:  # same call, same answer, N times: stuck
                on_event(
                    "loop_detected",
                    {"step": step, "name": key[0], "args": key[1], "count": seen[key]},
                )
                return (
                    f"Stopped: {key[0]} {key[1]} was called {seen[key]} times "
                    "with the same result (loop detected)."
                )

    on_event("max_steps", {"step": max_steps})
    return f"Stopped after {max_steps} steps without a final answer."
