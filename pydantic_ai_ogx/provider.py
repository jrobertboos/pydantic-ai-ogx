from __future__ import annotations as _annotations

import httpx
from ogx.core.library_client import AsyncOGXAsLibraryClient
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers import Provider


class OgxProvider(Provider[AsyncOpenAI]):
    """Provider for OGX API."""

    @property
    def name(self) -> str:
        return 'ogx'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return openai_model_profile(model_name)

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        ogx_client: AsyncOGXAsLibraryClient | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new OGX provider.

        Args:
            base_url: The base url for the OGX server. Required for server mode.
            api_key: The API key to use for authentication. If not provided, a placeholder
                ``'not-needed'`` is used since OGX servers typically don't require one.
            ogx_client: An existing
                [`AsyncOGXAsLibraryClient`](https://github.com/ogx-ai/ogx) for in-process (library)
                mode. If provided, ``base_url``, ``api_key``, and ``http_client`` must be ``None``.
            http_client: An existing ``httpx.AsyncClient`` to use for making HTTP requests.
        """
        if ogx_client is not None:
            assert base_url is None, 'Cannot provide both `ogx_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `ogx_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `ogx_client` and `api_key`'

            from ._transport import OgxLibraryTransport

            self._ogx_client = ogx_client
            transport = OgxLibraryTransport(ogx_client)
            lib_http_client = httpx.AsyncClient(transport=transport, base_url='http://ogx-library')
            self._client = AsyncOpenAI(
                http_client=lib_http_client,
                base_url='http://ogx-library/v1',
                api_key='not-needed',
            )
        else:
            if not base_url:
                raise UserError(
                    'Pass `base_url` to `OgxProvider(base_url=...)` or use library mode via'
                    ' `OgxProvider(ogx_client=...)` to use the OGX provider.'
                )

            # OGX servers typically don't require an API key, but the OpenAI client
            # requires a non-empty key — use a placeholder when none is provided.
            api_key = api_key or 'not-needed'

            if http_client is not None:
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = create_async_http_client()
                self._own_http_client = http_client
                self._http_client_factory = create_async_http_client
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
