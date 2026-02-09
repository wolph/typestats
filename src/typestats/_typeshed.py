from typing import TYPE_CHECKING, Final, Literal

import mainpy

if TYPE_CHECKING:
    from typing import TypedDict, type_check_only

    import httpx

    @type_check_only
    class _ContentItemLinks(TypedDict):
        self: str
        git: str
        html: str

    @type_check_only
    class _ContentItem(TypedDict):
        name: str
        path: str
        sha: str
        size: int
        url: str
        html_url: str
        git_url: str
        download_url: str | None
        type: Literal["file", "dir", "symlink", "submodule"]
        _links: _ContentItemLinks


__all__ = "stub_dirs", "stub_packages"

_GH_STUBS_URL: Final = "https://api.github.com/repos/python/typeshed/contents/stubs"


async def stub_dirs(client: httpx.AsyncClient, /) -> list[str]:
    """List stub directory names in python/typeshed using GitHub's contents API."""
    response = await client.get(_GH_STUBS_URL)
    response.raise_for_status()
    data: list[_ContentItem] = response.json()

    return sorted(item["name"] for item in data if item.get("type") == "dir")


async def stub_packages(client: httpx.AsyncClient, /) -> list[str]:
    """List typeshed's PyPI stub package names."""
    return [f"types-{stub_dir}" for stub_dir in await stub_dirs(client)]


@mainpy.main
async def example() -> None:
    import httpx  # noqa: PLC0415

    async with httpx.AsyncClient(http2=True) as client:
        packages = await stub_packages(client)
        print(*packages, sep="\n")  # noqa: T201
