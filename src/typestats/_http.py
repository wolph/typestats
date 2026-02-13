from typing import Any, Final

import httpx
from httpx_retries import Retry, RetryTransport

__all__ = ("create_client",)

_RETRY_STATUS_CODES: Final = frozenset({429, 500, 502, 503, 504})


def create_client(**kwargs: Any) -> httpx.AsyncClient:
    """Create an httpx AsyncClient with automatic retry transport.

    Configures ``httpx-retries`` for transient HTTP errors with exponential
    backoff. Retries up to 3 times on status codes 429, 500, 502, 503, and
    504, using a backoff factor of 0.5 seconds. HTTP/2 is enabled on the
    underlying transport.

    Args:
        **kwargs: Additional keyword arguments forwarded to
            :class:`httpx.AsyncClient`.
    """
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=_RETRY_STATUS_CODES,
    )
    transport = RetryTransport(
        transport=httpx.AsyncHTTPTransport(http2=True),
        retry=retry,
    )
    return httpx.AsyncClient(transport=transport, **kwargs)
