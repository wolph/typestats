import httpx
from httpx_retries import RetryTransport

from typestats._http import create_client


def test_create_client_returns_async_client() -> None:
    client = create_client()
    assert isinstance(client, httpx.AsyncClient)


def test_create_client_uses_retry_transport() -> None:
    client = create_client()
    assert isinstance(client._transport, RetryTransport)  # noqa: SLF001


def test_retry_configuration() -> None:
    client = create_client()
    transport = client._transport  # noqa: SLF001
    assert isinstance(transport, RetryTransport)
    retry = transport.retry
    assert retry.total == 3
    assert retry.backoff_factor == 0.5  # noqa: RUF069
    assert frozenset(retry.status_forcelist) == frozenset({429, 500, 502, 503, 504})
