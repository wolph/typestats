"""typestats CLI -- type annotation coverage for Python packages."""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

import anyio
import typer

from typestats import analyze

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class OutputFormat(StrEnum):
    """Supported output formats."""

    TEXT = "text"
    JSON = "json"


app = typer.Typer(
    name="typestats",
    help="Type annotation coverage statistics for Python packages on PyPI.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version  # noqa: PLC0415

        typer.echo(f"typestats {version('typestats')}")
        raise typer.Exit


@app.callback()
def main(
    *,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
    version: Annotated[  # noqa: ARG001
        bool | None,
        typer.Option(
            "--version",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Type annotation coverage statistics for Python packages on PyPI."""
    level = logging.DEBUG if verbose else logging.INFO
    if not logging.root.handlers:
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    else:
        logging.root.setLevel(level)


@app.command()
def check(
    package: Annotated[str, typer.Argument(help="PyPI package name to analyze.")],
    *,
    output: Annotated[
        OutputFormat,
        typer.Option("--output", "-o", help="Output format."),
    ] = OutputFormat.TEXT,
    temp_dir: Annotated[
        str | None,
        typer.Option("--temp-dir", help="Directory for downloaded packages."),
    ] = None,
) -> None:
    """Analyze type annotation coverage for a PyPI package."""  # noqa: DOC501
    import httpx  # noqa: PLC0415

    try:
        anyio.run(_check_async, package, output, temp_dir)
    except httpx.HTTPError as exc:
        typer.echo(f"HTTP error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


async def _check_async(
    package: str,
    output: OutputFormat,
    temp_dir: str | None,
) -> None:
    import tempfile  # noqa: PLC0415

    import httpx  # noqa: PLC0415

    from typestats._pypi import download_sdist_latest  # noqa: PLC0415
    from typestats.index import collect_public_symbols  # noqa: PLC0415

    work_dir = temp_dir or tempfile.mkdtemp(prefix="typestats-")

    async with httpx.AsyncClient(http2=True) as client:
        path, _file_detail = await download_sdist_latest(client, package, work_dir)

    symbols = await collect_public_symbols(path)
    _report(symbols, output, package)


def _report(
    symbols: Mapping[anyio.Path, Sequence[analyze.Symbol]],
    output: OutputFormat,
    package: str,
) -> None:
    total = sum(len(syms) for syms in symbols.values())
    annotated = sum(
        sum(1 for s in syms if analyze.is_annotated(s.type_))
        for syms in symbols.values()
    )

    if output == OutputFormat.JSON:
        typer.echo(
            json.dumps(
                {
                    "package": package,
                    "modules": len(symbols),
                    "symbols_total": total,
                    "symbols_annotated": annotated,
                },
                indent=2,
            ),
        )
    else:
        typer.echo(f"Package: {package}")
        typer.echo(f"Modules: {len(symbols)}")
        typer.echo(f"Symbols: {annotated}/{total} annotated")
