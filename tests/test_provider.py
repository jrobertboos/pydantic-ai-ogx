"""Tests for pydantic_ai_ogx provider — server mode, library mode, and factory helpers."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from pydantic_ai_ogx import OgxProvider


# ---------------------------------------------------------------------------
# Server-mode provider construction
# ---------------------------------------------------------------------------


class TestOgxProviderServerMode:
    def test_default_construction(self) -> None:
        provider = OgxProvider()
        assert provider.name == "ogx"
        assert "localhost:8321" in provider.base_url
        assert isinstance(provider.client, AsyncOpenAI)

    def test_explicit_base_url(self) -> None:
        provider = OgxProvider(base_url="http://my-ogx:9999/v1")
        assert "my-ogx:9999" in provider.base_url

    def test_explicit_api_key(self) -> None:
        provider = OgxProvider(api_key="secret-key")
        assert provider.client.api_key == "secret-key"

    def test_env_var_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OGX_BASE_URL", "http://env-host:1234/v1")
        provider = OgxProvider()
        assert "env-host:1234" in provider.base_url

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OGX_API_KEY", "env-key")
        provider = OgxProvider()
        assert provider.client.api_key == "env-key"

    def test_custom_http_client(self) -> None:
        custom = httpx.AsyncClient()
        provider = OgxProvider(http_client=custom)
        assert isinstance(provider.client, AsyncOpenAI)

    def test_repr(self) -> None:
        provider = OgxProvider()
        r = repr(provider)
        assert "OgxProvider" in r
        assert "ogx" in r


# ---------------------------------------------------------------------------
# Library-mode provider construction
# ---------------------------------------------------------------------------


class TestOgxProviderLibraryMode:
    def test_rejects_base_url_with_ogx_client(self) -> None:
        from ogx.core.library_client import AsyncOGXAsLibraryClient

        client = AsyncOGXAsLibraryClient.__new__(AsyncOGXAsLibraryClient)
        with pytest.raises(AssertionError, match="base_url"):
            OgxProvider(ogx_client=client, base_url="http://nope")

    def test_rejects_api_key_with_ogx_client(self) -> None:
        from ogx.core.library_client import AsyncOGXAsLibraryClient

        client = AsyncOGXAsLibraryClient.__new__(AsyncOGXAsLibraryClient)
        with pytest.raises(AssertionError, match="api_key"):
            OgxProvider(ogx_client=client, api_key="nope")

    def test_rejects_http_client_with_ogx_client(self) -> None:
        from ogx.core.library_client import AsyncOGXAsLibraryClient

        client = AsyncOGXAsLibraryClient.__new__(AsyncOGXAsLibraryClient)
        with pytest.raises(AssertionError, match="http_client"):
            OgxProvider(ogx_client=client, http_client=httpx.AsyncClient())

    def test_library_mode_creates_openai_client(self) -> None:
        from ogx.core.library_client import AsyncOGXAsLibraryClient

        client = AsyncOGXAsLibraryClient.__new__(AsyncOGXAsLibraryClient)
        client.provider_data = None
        client.route_impls = None
        provider = OgxProvider(ogx_client=client)
        assert provider.name == "ogx"
        assert isinstance(provider.client, AsyncOpenAI)
        assert "ogx-library" in provider.base_url

    def test_plain_ogx_client_server_mode(self) -> None:
        from ogx_client import AsyncOgxClient

        client = AsyncOgxClient(base_url="http://my-ogx:9999/v1", api_key="my-key")
        provider = OgxProvider(ogx_client=client)
        assert provider.name == "ogx"
        assert isinstance(provider.client, AsyncOpenAI)
        assert "my-ogx:9999" in provider.base_url
        assert provider.client.api_key == "my-key"

    def test_plain_ogx_client_rejects_base_url(self) -> None:
        from ogx_client import AsyncOgxClient

        client = AsyncOgxClient(base_url="http://my-ogx:9999/v1")
        with pytest.raises(AssertionError, match="base_url"):
            OgxProvider(ogx_client=client, base_url="http://nope")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


class TestFactoryHelpers:
    def test_create_ogx_model_chat(self) -> None:
        model = create_ogx_model("test-model")
        assert isinstance(model, OpenAIChatModel)
        assert model.model_name == "test-model"

    def test_create_ogx_model_responses(self) -> None:
        model = create_ogx_model("test-model", model_type="responses")
        assert isinstance(model, OpenAIResponsesModel)
        assert model.model_name == "test-model"

    def test_create_ogx_model_custom_url(self) -> None:
        model = create_ogx_model("m", base_url="http://custom:5000/v1")
        assert isinstance(model, OpenAIChatModel)

    def test_create_ogx_library_model_chat(self) -> None:
        from ogx.core.library_client import AsyncOGXAsLibraryClient

        client = AsyncOGXAsLibraryClient.__new__(AsyncOGXAsLibraryClient)
        client.provider_data = None
        client.route_impls = None
        model = create_ogx_library_model("test-model", client)
        assert isinstance(model, OpenAIChatModel)
        assert model.model_name == "test-model"

    def test_create_ogx_library_model_responses(self) -> None:
        from ogx.core.library_client import AsyncOGXAsLibraryClient

        client = AsyncOGXAsLibraryClient.__new__(AsyncOGXAsLibraryClient)
        client.provider_data = None
        client.route_impls = None
        model = create_ogx_library_model("test-model", client, model_type="responses")
        assert isinstance(model, OpenAIResponsesModel)
        assert model.model_name == "test-model"


# ---------------------------------------------------------------------------
# Provider used with both model types
# ---------------------------------------------------------------------------


class TestProviderWithModels:
    def test_chat_model_with_provider(self) -> None:
        provider = OgxProvider(base_url="http://localhost:8321/v1")
        model = OpenAIChatModel("my-model", provider=provider)
        assert model.model_name == "my-model"
        assert model.system == "ogx"

    def test_responses_model_with_provider(self) -> None:
        provider = OgxProvider(base_url="http://localhost:8321/v1")
        model = OpenAIResponsesModel("my-model", provider=provider)
        assert model.model_name == "my-model"
        assert model.system == "ogx"


# ---------------------------------------------------------------------------
# Library transport — unit-level
# ---------------------------------------------------------------------------


class TestOgxLibraryTransport:
    def test_uninitialized_client_raises(self) -> None:
        from ogx.core.library_client import AsyncOGXAsLibraryClient

        from pydantic_ai_ogx._transport import OgxLibraryTransport

        client = AsyncOGXAsLibraryClient.__new__(AsyncOGXAsLibraryClient)
        client.route_impls = None
        client.provider_data = None
        transport = OgxLibraryTransport(client)

        request = httpx.Request("POST", "http://ogx-library/v1/chat/completions")

        with pytest.raises(RuntimeError, match="not initialized"):
            asyncio.get_event_loop().run_until_complete(
                transport.handle_async_request(request)
            )


# ---------------------------------------------------------------------------
# End-to-end with a real library client (requires ogx[starter] + model)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    True,  # flip to False when you have ollama running with a model pulled
    reason="Requires a running Ollama instance with a pulled model",
)
class TestEndToEnd:
    """Smoke tests that actually run inference. Skipped by default."""

    MODEL = "llama3.2"

    @pytest.mark.asyncio
    async def test_library_chat(self) -> None:
        from pydantic_ai import Agent
        from ogx.core.library_client import AsyncOGXAsLibraryClient

        async with AsyncOGXAsLibraryClient("starter") as ogx_client:
            model = create_ogx_library_model(self.MODEL, ogx_client)
            agent = Agent(model)
            result = await agent.run("Say hello in one word.")
            assert isinstance(result.output, str)
            assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_library_responses(self) -> None:
        from pydantic_ai import Agent
        from ogx.core.library_client import AsyncOGXAsLibraryClient

        async with AsyncOGXAsLibraryClient("starter") as ogx_client:
            model = create_ogx_library_model(self.MODEL, ogx_client, model_type="responses")
            agent = Agent(model)
            result = await agent.run("Say hello in one word.")
            assert isinstance(result.output, str)
            assert len(result.output) > 0
