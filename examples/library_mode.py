"""Run a Pydantic AI agent using OGX in library mode (no server required).

Prerequisites:
    pip install pydantic-ai-ogx 'ogx[starter]'

    For local Ollama models, make sure Ollama is running (`ollama serve`)
    and you've pulled the model (`ollama pull llama3.2`).

    For cloud models, set the relevant API key as an environment variable
    (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY) so OGX can forward it via
    provider data.
"""

from __future__ import annotations

import asyncio
import os
import sys

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from ogx.core.library_client import AsyncOGXAsLibraryClient

from pydantic_ai_ogx import OgxProvider

from pydantic_ai.output import NativeOutput

MODEL = os.environ.get("OGX_MODEL", "ollama/llama3.2")


class CityInfo(BaseModel):
    name: str
    country: str
    population: int
    fun_fact: str


async def main() -> None:
    print(f"Using model: {MODEL}")
    print()

    async with AsyncOGXAsLibraryClient("starter") as ogx_client:
        provider = OgxProvider(ogx_client=ogx_client)
        model = OpenAIChatModel(MODEL, provider=provider)

        # 1. Simple text completion
        print("--- Simple completion ---")
        simple_agent = Agent(model)
        result = await simple_agent.run("What is the capital of France?")
        print(result.output)
        print()

        # 2. Structured output
        print("--- Structured output ---")
        structured_agent = Agent(model, output_type=NativeOutput(CityInfo))
        result = await structured_agent.run("Tell me about Tokyo.")
        city = result.output
        print(f"{city.name}, {city.country}")
        print(f"  Population: {city.population:,}")
        print(f"  Fun fact:   {city.fun_fact}")
        print()

        # 3. Tool use
        print("--- Tool use ---")
        tool_agent = Agent(model)

        @tool_agent.tool
        async def get_weather(ctx: RunContext[None], city: str) -> str:
            """Return the current weather for *city*."""
            return f"It's 22 °C and sunny in {city}."

        result = await tool_agent.run("What's the weather like in Berlin?")
        print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
