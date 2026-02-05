import graphlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import mainpy

from typestats import _ruff

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import StrPath


__all__ = ("list_sources",)


_logger = logging.getLogger(__name__)


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
