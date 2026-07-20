"""Capability probe for the LLM gateway.

"OpenAI-compatible" is a spectrum: many gateways serve /chat/completions
faithfully but return `tool_calls: null`, reject strict `json_schema`, or emit
only one tool call per turn. Feature 3 (the coach-assist agent) depends on all
three, so this verifies them against the live endpoint before the agent is
designed around them.

Run:  PYTHONPATH=src python scripts/probe_gateway.py
Never prints the API key.
"""

from __future__ import annotations

import asyncio
import sys

from app.config import get_settings
from app.llm.client import LLMClient, Usage

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
POPULATION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_population",
        "description": "Get the population of a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
DEMO_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "score": {"type": "integer"}},
    "required": ["city", "score"],
    "additionalProperties": False,
}


async def main() -> int:
    settings = get_settings()
    client = LLMClient(settings)
    usage = Usage()
    failures: list[str] = []

    print(f"gateway   : {settings.base_url}")
    print(f"chat model: {settings.llm_model_name}")
    print(f"embedding : {settings.text_embedding_model_name}\n")

    async def check(label: str, coro, verify) -> None:
        try:
            result = await coro
        except Exception as exc:  # noqa: BLE001 - probe reports, never raises
            print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
            failures.append(label)
            return
        ok, detail = verify(result)
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}: {detail}")
        if not ok:
            failures.append(label)

    await check(
        "basic chat",
        client.chat(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            usage=usage,
            max_tokens=10,
        ),
        lambda text: (bool(text.strip()), repr(text)),
    )

    await check(
        "native tool calling",
        client.chat_with_tools(
            [{"role": "user", "content": "What's the weather in Hanoi?"}],
            [WEATHER_TOOL],
            usage=usage,
        ),
        lambda c: (
            bool(c.choices[0].message.tool_calls),
            f"finish_reason={c.choices[0].finish_reason} "
            f"calls={[t.function.name for t in c.choices[0].message.tool_calls or []]}",
        ),
    )

    await check(
        "parallel tool calling",
        client.chat_with_tools(
            [{"role": "user", "content": "Weather AND population of Hanoi?"}],
            [WEATHER_TOOL, POPULATION_TOOL],
            usage=usage,
        ),
        lambda c: (
            len(c.choices[0].message.tool_calls or []) >= 2,
            f"n={len(c.choices[0].message.tool_calls or [])}",
        ),
    )

    await check(
        "strict json_schema",
        client.structured(
            [{"role": "user", "content": "City Hanoi, score 7."}],
            DEMO_SCHEMA,
            usage=usage,
        ),
        lambda text: ("city" in text, repr(text)),
    )

    await check(
        "embeddings",
        client.embed(["bench press technique"], usage=usage),
        lambda vecs: (
            len(vecs[0]) == get_settings().embedding_dim,
            f"dim={len(vecs[0])} (config expects {get_settings().embedding_dim})",
        ),
    )

    await client.aclose()
    print(f"\nusage: {usage.as_dict(settings)}")

    if failures:
        print(f"\n{len(failures)} capability check(s) failed: {', '.join(failures)}")
        return 1
    print("\nall capability checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
