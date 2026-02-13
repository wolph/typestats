import io
import logging
import tarfile
from typing import TYPE_CHECKING, Any, Final, Literal, NotRequired, TypedDict

import anyio
import anyio.to_thread
import mainpy
from packaging.utils import parse_sdist_filename
from yarl import URL

if TYPE_CHECKING:
    import httpx
    from _typeshed import StrPath


__all__ = "download_sdist", "download_sdist_latest", "fetch_project_detail"


HOST: Final = URL("https://files.pythonhosted.org")
HEADERS: Final = {
    "Host": "pypi.org",
    "Accept": "application/vnd.pypi.simple.v1+json",
}


class _ProjectHashes(TypedDict):
    sha256: str
    blake2b: NotRequired[str]
    md5: NotRequired[str]


FileDetail = TypedDict(
    "FileDetail",
    {
        "core-metadata": NotRequired[dict[str, str] | bool],
        "data-dist-info-metadata": NotRequired[dict[str, str] | bool],
        "filename": str,
        "hashes": _ProjectHashes,
        "provenance": NotRequired[str | None],
        "requires-python": NotRequired[str | None],  # PEP 440 specifier
        "size": int,  # in bytes
        "upload-time": NotRequired[str],  # ISO 8601
        "url": str,
        "yanked": NotRequired[bool],
        "gpg-sig": NotRequired[bool],
    },
)


_ProjectMeta = TypedDict(
    "_ProjectMeta",
    {
        "_last-serial": NotRequired[int],
        "api-version": str,
        "project-status": NotRequired[str],
        "project-status-reason": NotRequired[str],
    },
)


class _ProjectStatus(TypedDict):
    # https://packaging.python.org/en/latest/specifications/project-status-markers/
    status: Literal["active", "archived", "quarantined", "deprecated"]


# https://packaging.python.org/en/latest/specifications/simple-repository-api/#simple-repository-json-project-detail
ProjectDetail = TypedDict(
    "ProjectDetail",
    {
        "name": str,
        "files": list[FileDetail],
        "meta": _ProjectMeta,
        "project-status": NotRequired[_ProjectStatus],
        "project-status-reason": NotRequired[str],
        "versions": list[str],
    },
)

_logger = logging.getLogger(__name__)


async def _get_json(client: httpx.AsyncClient, url: URL, /, **kwargs: Any) -> Any:
    response = await client.get(str(url), **kwargs)
    response.raise_for_status()
    return response.json()


async def fetch_project_detail(
    client: httpx.AsyncClient,
    project_name: str,
    /,
) -> ProjectDetail:
    """
    Get the project detail from PyPI's Simple API.

    For details, see:
    - https://peps.python.org/pep-0691/
    - https://docs.pypi.org/api/index-api/#json_1
    """
    url = HOST / "simple" / project_name / ""

    data = await _get_json(client, url, headers=HEADERS)
    return ProjectDetail(data)


def _latest_sdist(details: ProjectDetail, /) -> FileDetail:
    """Finds the latest sdist from the given project detail."""

    # we only return the URL, which contains the version
    sdists = [
        sdist
        for sdist in details["files"]
        if (sdist["filename"].endswith((".tar.gz", ".zip")))
        and not sdist.get("yanked", False)
    ]

    return max(sdists, key=lambda sdist: parse_sdist_filename(sdist["filename"])[1])


async def _sdist_extract(content: bytes, out_dir: anyio.Path, /) -> None:
    """
    Async version of `with tarfile.open(...) as tar: tar.extractall(out_dir)` for
    raw bytes.
    """

    def _extract() -> None:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            tar.extractall(path=out_dir, filter="data")

    await anyio.to_thread.run_sync(_extract)


async def download_sdist(
    client: httpx.AsyncClient,
    sdist: FileDetail,
    out_dir: StrPath,
    /,
) -> anyio.Path:
    """
    Download and extract the given sdist file (if not already present) and return its
    path.
    """
    out_dir = await anyio.Path(out_dir).resolve()
    await out_dir.mkdir(parents=True, exist_ok=True)

    target_path = out_dir / sdist["filename"].removesuffix(".tar.gz")
    if not await target_path.is_dir():
        response = await client.get(sdist["url"])
        response.raise_for_status()

        await _sdist_extract(response.content, out_dir)
        _logger.info("Extracted %s into %s", sdist["filename"], target_path)

    return target_path


async def download_sdist_latest(
    client: httpx.AsyncClient,
    project_name: str,
    out_dir: StrPath,
    /,
) -> tuple[anyio.Path, FileDetail]:
    """
    Download and extract the latest sdist for the given project name and return its
    path.
    """
    detail = await fetch_project_detail(client, project_name)
    sdist = _latest_sdist(detail)
    path = await download_sdist(client, sdist, out_dir)
    return path, sdist


@mainpy.main
async def example() -> None:
    from typestats._http import retry_client  # noqa: PLC0415

    async with retry_client() as client:
        await download_sdist_latest(client, "optype", "./projects")
