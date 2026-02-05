import enum
import graphlib
import logging
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import mainpy

from typestats import _ruff

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from _typeshed import StrPath


__all__ = ("list_sources",)


_logger = logging.getLogger(__name__)

_INIT_PATTERN = re.compile(r"^__init__\.(py|pyi|pyx)$")
_STUBS_DIR_PATTERN = re.compile(r"^(.+)-stubs$")


async def list_sources(project_dir: StrPath, /) -> map[anyio.Path]:
    """
    List all source files in the given project directory in topological order.
    This ensures that dependencies are listed before the files that depend on them.
    For example, if module `A` imports module `B`, then `B` will appear before `A` in
    the list.

    Raises:
        graphlib.CycleError: if there is a cycle in the import graph.
    """
    graph = await _ruff.analyze_graph(project_dir, "--type-checking-imports")
    sorter = graphlib.TopologicalSorter(graph)

    try:
        sorter.prepare()
    except graphlib.CycleError:
        project_name = Path(project_dir).name
        _logger.warning(
            "[%s] Import cycle detected; retrying without TYPE_CHECKING imports",
            project_name,
        )
        graph = await _ruff.analyze_graph(project_dir, "--no-type-checking-imports")
        sorter = graphlib.TopologicalSorter(graph)

        try:
            sorter.prepare()
        except graphlib.CycleError as e:
            # TODO(@jorenham): automatically break the cycle,
            _logger.exception("[%s] Import cycle detected: %r", project_name, e.args[1])
            raise

    return map(anyio.Path, sorter.static_order())


def sources_root(sources: Iterable[StrPath], /) -> anyio.Path:
    """
    Return the root directory containing source files in the given project directory.
    This is determined by finding the common ancestor of all source files.
    """
    return anyio.Path(os.path.commonpath(sources))


class PyTyped(enum.Enum):
    NO = enum.auto()
    YES = enum.auto()
    PARTIAL = enum.auto()
    STUBS = enum.auto()


async def get_py_typed(project_dir: StrPath, /) -> PyTyped:
    """
    Determine the `py.typed` status of the given project directory.
    """
    sources = await list_sources(project_dir)
    root = sources_root(sources)

    if root.parent.name.endswith("-stubs"):
        return PyTyped.STUBS

    py_typed = root / "py.typed"
    if not await py_typed.exists():
        return PyTyped.NO

    # https://typing.python.org/en/latest/spec/distributing.html#partial-stub-packages
    contents = await py_typed.read_text()
    if contents and "partial\n" in contents:
        return PyTyped.PARTIAL

    return PyTyped.YES


def sources_to_module_paths(
    sources: Iterable[StrPath],
    /,
) -> Mapping[str, frozenset[anyio.Path]]:
    """
    Group the source files of a project by their fully qualified module paths.

    This will take into account any stubs files (i.e., `.pyi` files) that may exist
    alongside the source files, or in a separate `{project}-stubs` package.

    This also supports single-file modules (i.e., without `__init__`).

    Namespace packages are not currently supported.
    """
    source_paths = list(map(anyio.Path, sources))

    init_files = frozenset(p for p in source_paths if _INIT_PATTERN.match(p.name))
    init_dirs = frozenset(p.parent for p in init_files)

    def _module_path(source: anyio.Path) -> str:
        parts = deque[str]()
        current_dir = source.parent
        while current_dir in init_dirs:
            dir_name = current_dir.name
            if match := _STUBS_DIR_PATTERN.match(dir_name):
                parts.appendleft(match.group(1))
                break
            parts.appendleft(dir_name)
            current_dir = current_dir.parent

        if not _INIT_PATTERN.match(source.name):
            parts.append(source.stem)

        return ".".join(parts) if parts else source.stem

    module_paths: defaultdict[str, set[anyio.Path]] = defaultdict(set)
    for path in source_paths:
        module_paths[_module_path(path)].add(path)

    return {k: frozenset(ps) for k, ps in module_paths.items()}


@mainpy.main
async def example() -> None:
    import httpx  # noqa: PLC0415

    from typestats import _pypi  # noqa: PLC0415

    async with anyio.TemporaryDirectory() as temp_dir:
        async with httpx.AsyncClient(http2=True) as client:
            path, detail = await _pypi.download_sdist_latest(client, "optype", temp_dir)

        ruff_analyze_opts: list[str] = []
        if "requires-python" in detail and (req := detail["requires-python"]):
            py_min = req.split(">=", 1)[1].split(",", 1)[0].strip()
            assert py_min, req
            ruff_analyze_opts.extend(["--python-version", req])

        sources = list(await list_sources(path))
        root = sources_root(sources)
        print(*[s.relative_to(root.parent) for s in sources], sep="\n")  # noqa: T201
