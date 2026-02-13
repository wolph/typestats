import io
import logging
import tarfile
import zipfile
from typing import TYPE_CHECKING, Any, Final, Literal, NotRequired, TypedDict

import anyio
import anyio.to_thread
import httpx
import mainpy
from packaging.utils import parse_sdist_filename, parse_wheel_filename
from yarl import URL

if TYPE_CHECKING:
    from _typeshed import StrPath


__all__ = "download_package", "download_package_latest", "fetch_project_detail"


class _NoDistributionFoundError(Exception):
    """No suitable distribution (sdist or wheel) found."""


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
    """Finds the latest sdist from the given project detail.

    Raises:
        _NoDistributionFoundError: If no sdist is found.
    """
    sdists = [
        sdist
        for sdist in details["files"]
        if sdist["filename"].endswith((".tar.gz", ".zip"))
        and not sdist.get("yanked", False)
    ]
    if not sdists:
        msg = "No sdist found"
        raise _NoDistributionFoundError(msg)
    return max(sdists, key=lambda sdist: parse_sdist_filename(sdist["filename"])[1])


def _latest_wheel(details: ProjectDetail, /) -> FileDetail:
    """Finds the latest wheel from the given project detail.

    Raises:
        _NoDistributionFoundError: If no wheel is found.
    """
    wheels = [
        w
        for w in details["files"]
        if w["filename"].endswith(".whl") and not w.get("yanked", False)
    ]
    if not wheels:
        msg = "No wheel found"
        raise _NoDistributionFoundError(msg)
    # Prefer pure-Python wheels (none-any)
    pure = [
        w
        for w in wheels
        if all(
            tag.abi == "none" and tag.platform == "any"
            for tag in parse_wheel_filename(w["filename"])[3]
        )
    ]
    candidates = pure or wheels
    return max(
        candidates,
        key=lambda w: parse_wheel_filename(w["filename"])[1],
    )


def _latest_distribution(
    details: ProjectDetail,
    /,
) -> tuple[FileDetail, Literal["sdist", "wheel"]]:
    """Try sdist first, fall back to wheel."""
    try:
        return _latest_sdist(details), "sdist"
    except _NoDistributionFoundError:
        return _latest_wheel(details), "wheel"


async def _tar_extract(content: bytes, out_dir: anyio.Path, /) -> None:
    """Extract a `.tar.gz` archive asynchronously."""

    def _extract() -> None:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            tar.extractall(path=out_dir, filter="data")

    await anyio.to_thread.run_sync(_extract)


async def _zip_extract(content: bytes, out_dir: anyio.Path, /) -> None:
    """Extract a `.zip` archive asynchronously."""

    def _extract() -> None:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(path=out_dir)  # noqa: S202

    await anyio.to_thread.run_sync(_extract)


async def download_package(
    client: httpx.AsyncClient,
    file_detail: FileDetail,
    out_dir: StrPath,
    /,
) -> anyio.Path:
    """Download and extract a package file (sdist or wheel) and return its path.

    Raises:
        ValueError: If the file type is unsupported.
    """
    out_dir = await anyio.Path(out_dir).resolve()
    await out_dir.mkdir(parents=True, exist_ok=True)

    filename = file_detail["filename"]
    if filename.endswith(".tar.gz"):
        target_name = filename.removesuffix(".tar.gz")
        extract = _tar_extract
    elif filename.endswith(".zip"):
        target_name = filename.removesuffix(".zip")
        extract = _zip_extract
    elif filename.endswith(".whl"):
        target_name = filename.removesuffix(".whl")
        extract = _zip_extract
    else:
        msg = f"Unsupported file type: {filename}"
        raise ValueError(msg)

    target_path = out_dir / target_name
    if not await target_path.is_dir():
        response = await client.get(file_detail["url"])
        response.raise_for_status()

        await extract(response.content, out_dir)
        _logger.info("Extracted %s into %s", filename, target_path)

    return target_path


async def download_package_latest(
    client: httpx.AsyncClient,
    project_name: str,
    out_dir: StrPath,
    /,
) -> tuple[anyio.Path, FileDetail]:
    """
    Download and extract the latest package (sdist or wheel) for the given
    project name and return its path.

    Returns:
        A ``(path, file_detail)`` tuple where *path* is the extracted
        directory and *file_detail* is the PyPI file metadata.
    """
    detail = await fetch_project_detail(client, project_name)
    file_detail, _ = _latest_distribution(detail)
    path = await download_package(client, file_detail, out_dir)
    return path, file_detail


@mainpy.main
async def example() -> None:
    async with httpx.AsyncClient(http2=True) as client:
        await download_package_latest(client, "optype", "./projects")
