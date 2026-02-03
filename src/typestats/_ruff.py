import json
import logging
from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    from _typeshed import StrPath


__all__ = ("analyze_graph",)


_logger = logging.getLogger(__name__)


async def analyze_graph(project_dir: StrPath, *opts: str) -> dict[str, list[str]]:
    """
    Run `ruff analyze graph` on the given project directory.

    Raises:
        NotADirectoryError:
            if `project_dir` is not a directory (i.e., does not exist or is not a
            directory).

    Returns:
        A mapping from each analyzed file to the list of files it depends on (or
        vice-versa if `--direction=dependents` is passed).
    """
    path = anyio.Path(project_dir)
    if not await path.is_dir():
        msg = f"{path} is not a directory"
        raise NotADirectoryError(msg)

    subprocess_args = ["ruff", "analyze", "graph", "--quiet", *opts, str(path)]
    _logger.info("Running subprocess: %s", " ".join(subprocess_args))
    result = await anyio.run_process(subprocess_args)
    result.check_returncode()
    return json.loads(result.stdout)
