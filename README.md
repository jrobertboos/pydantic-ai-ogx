# pydantic-ai-ogx

[![PyPI version](https://badge.fury.io/py/pydantic-ai-ogx.svg)](https://badge.fury.io/py/pydantic-ai-ogx)
[![Python](https://img.shields.io/pypi/pyversions/pydantic-ai-ogx.svg)](https://pypi.org/project/pydantic-ai-ogx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pydantic AI provider for [OGX](https://github.com/ogx-ai/ogx) -- the open GenAI stack.

## What is OGX?

OGX is an open-source agentic API server that provides a drop-in replacement for the OpenAI API. It enables you to:

- Run AI models **anywhere**: locally with Ollama, in datacenters with vLLM, or via cloud services
- **Swap models seamlessly**: switch between Llama, GPT, Gemini, Mistral, or any model without changing application code
- **Maintain compatibility**: works with OpenAI, Anthropic, and Google GenAI SDKs

## What is this extension?

`pydantic-ai-ogx` provides an `OgxProvider` for [Pydantic AI](https://github.com/pydantic/pydantic-ai) that works with both `OpenAIChatModel` and `OpenAIResponsesModel`. It supports two modes:

1. **Server mode** -- connect to a running OGX server via HTTP (with a `base_url` or an `AsyncOgxClient`)
2. **Library mode** -- run OGX in-process via `AsyncOGXAsLibraryClient` (no server needed)

## Installation

```bash
pip install pydantic-ai-ogx
```

For library mode, also install OGX with the starter distribution:

```bash
pip install 'ogx[starter]'
```

## Quick Start

### Server Mode

Start an OGX server, then point the provider at it:

```bash
# Install & run OGX
curl -LsSf https://github.com/ogx-ai/ogx/raw/main/scripts/install.sh | bash
uv run ogx stack run starter
```

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai_ogx import OgxProvider

async def main():
    provider = OgxProvider(base_url="http://localhost:8321/v1")
    model = OpenAIChatModel("ollama/llama-3.2", provider=provider)
    agent = Agent(model)
    result = await agent.run("What is the capital of France?")
    print(result.data)

asyncio.run(main())
```

### Library Mode

No separate server -- OGX runs directly in your Python process:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai_ogx import OgxProvider
from ogx.core.library_client import AsyncOGXAsLibraryClient

async def main():
    async with AsyncOGXAsLibraryClient("starter") as ogx_client:
        provider = OgxProvider(ogx_client=ogx_client)
        model = OpenAIChatModel("ollama/llama-3.2", provider=provider)
        agent = Agent(model)
        result = await agent.run("What is the capital of France?")
        print(result.data)

asyncio.run(main())
```

## Usage

### Using `OgxProvider` directly

`OgxProvider` implements `Provider[AsyncOpenAI]`, so it plugs directly into
`OpenAIChatModel` and `OpenAIResponsesModel`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai_ogx import OgxProvider

# --- Server mode ---
provider = OgxProvider(base_url="http://localhost:8321/v1")

chat_model = OpenAIChatModel("ollama/llama-3.2", provider=provider)
responses_model = OpenAIResponsesModel("ollama/llama-3.2", provider=provider)

agent = Agent(chat_model)

# --- Library mode ---
from ogx.core.library_client import AsyncOGXAsLibraryClient

async with AsyncOGXAsLibraryClient("starter") as ogx_client:
    provider = OgxProvider(ogx_client=ogx_client)
    model = OpenAIChatModel("ollama/llama-3.2", provider=provider)
    agent = Agent(model)
    result = await agent.run("Hello!")
```

### Using the Responses API

Pass an `OgxProvider` to `OpenAIResponsesModel` to use the Responses API:

```python
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai_ogx import OgxProvider

provider = OgxProvider(base_url="http://localhost:8321/v1")
model = OpenAIResponsesModel("ollama/llama-3.2", provider=provider)
```

### Connecting to a Remote OGX Server

```python
from pydantic_ai_ogx import OgxProvider

provider = OgxProvider(
    base_url="https://ogx.example.com/v1",
    api_key="your-api-key",
)
```

### Structured Outputs

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai_ogx import OgxProvider

class City(BaseModel):
    name: str
    country: str
    population: int

provider = OgxProvider(base_url="http://localhost:8321/v1")
model = OpenAIChatModel("ollama/llama-3.2", provider=provider)
agent = Agent(model, result_type=City)

result = await agent.run("Tell me about Paris")
print(f"{result.data.name} has {result.data.population:,} people")
```

### Using Tools

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai_ogx import OgxProvider

provider = OgxProvider(base_url="http://localhost:8321/v1")
model = OpenAIChatModel("ollama/llama-3.2", provider=provider)
agent = Agent(model)

@agent.tool
async def get_weather(ctx: RunContext[None], city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny!"

result = await agent.run("What is the weather in Paris?")
```

## Configuration

### `OgxProvider` constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_url` | `str \| None` | OGX server URL. **Required** for server mode. |
| `api_key` | `str \| None` | API key. Defaults to `"not-needed"` since OGX servers typically don't require one. |
| `ogx_client` | `AsyncOgxClient \| None` | An OGX client instance. When an `AsyncOGXAsLibraryClient` (subclass) is passed, requests are dispatched in-process (library mode). A plain `AsyncOgxClient` reuses its `base_url`/`api_key` for server mode. Mutually exclusive with `base_url`, `api_key`, and `http_client`. |
| `http_client` | `httpx.AsyncClient \| None` | Custom httpx client for server mode. |

## Model Names

Model names depend on your OGX configuration:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai_ogx import OgxProvider

provider = OgxProvider(base_url="http://localhost:8321/v1")

# Ollama models via OGX provider prefix
model = OpenAIChatModel("ollama/llama3.2", provider=provider)
model = OpenAIChatModel("ollama/qwen2.5", provider=provider)

# Cloud models via OGX provider prefix
model = OpenAIChatModel("openai/gpt-4o-mini", provider=provider)
```

List available models on a running server:

```bash
curl http://localhost:8321/v1/models
```

## How It Works

### Server Mode

OGX is OpenAI-compatible, so server mode creates a standard `AsyncOpenAI`
client pointed at the OGX server URL. All Pydantic AI features (tools,
structured outputs, streaming, etc.) work out of the box.

### Library Mode

Library mode uses a custom `httpx.AsyncBaseTransport` that intercepts the
OpenAI SDK's HTTP requests and dispatches them through the
`AsyncOGXAsLibraryClient`'s in-process route handlers. This means:

- No network overhead -- requests never leave your process
- The same `AsyncOpenAI` client type is used, so `OpenAIChatModel` and
  `OpenAIResponsesModel` work unchanged
- The caller manages the `AsyncOGXAsLibraryClient` lifecycle (typically
  via `async with`)

## Troubleshooting

### 404 Error: Model Not Found

Check that the model name matches what OGX knows about:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai_ogx import OgxProvider

provider = OgxProvider(base_url="http://localhost:8321/v1")

# For Ollama models, use the Ollama model name
model = OpenAIChatModel("ollama/llama3.2", provider=provider)

# For cloud providers, use the OGX provider prefix
model = OpenAIChatModel("openai/gpt-4o-mini", provider=provider)
```

For Ollama, ensure you have pulled the model: `ollama pull llama3.2`

### Connection Errors in Server Mode

1. Make sure the OGX server is running: `uv run ogx stack run starter`
2. Verify the base URL matches your server configuration.

### Library Mode Not Working

Install the required extras:

```bash
pip install 'ogx[starter]'   # starter distribution
pip install 'ogx[client]'    # for custom configs
```

## Resources

- [OGX GitHub](https://github.com/ogx-ai/ogx)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Pydantic AI GitHub](https://github.com/pydantic/pydantic-ai)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License -- see [LICENSE](LICENSE) file for details.
