from typing import TYPE_CHECKING

import httpx
import pytest

from typestats._typeshed import stub_dirs, stub_packages

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture
async def httpx_client(
    request: pytest.FixtureRequest,
) -> AsyncGenerator[httpx.AsyncClient]:
    payload = request.param

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        yield client


@pytest.mark.anyio
@pytest.mark.parametrize(
    "httpx_client",
    [
        [
            {"name": "b", "type": "dir"},
            {"name": "a", "type": "dir"},
            {"name": "x", "type": "file"},
        ],
    ],
    indirect=True,
)
async def test_typeshed_stubs(httpx_client: httpx.AsyncClient) -> None:
    assert await stub_dirs(httpx_client) == ["a", "b"]
    assert await stub_packages(httpx_client) == ["types-a", "types-b"]
