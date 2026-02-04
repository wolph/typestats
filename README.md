# typestats

A tool to analyze the type annotation coverage of Python projects on PyPI.

> [!IMPORTANT]
> This project is a work-in-progress and is not yet functional.

## Implementation details

### Pipeline

For a given project:

1. Query PyPI for the latest version
2. Download the latest (non-yanked) sdist, and extract it
3. TODO: If there exists a `types-{project}` or `{project}-stubs` package on PyPI, repeat steps
   1-2 for that package, and merge the stubs into the main package source tree.
4. Compute the import graph using `ruff analyze graph`
5. Topologically sort the import graph to get a linear order of `.py` and `.pyi` files
6. For each file in sorted order, parse the file using `libcst`, and extract:
   - all annotatable global symbols and their type annotations
   - the `__all__` exports (if defined)
   - imports and implicit re-exports (i.e. `from a import b as b`)
   - type aliases (`_: TypeAlias = ...` and `type _ = ...`)
   - TODO: type-ignore comments (`# (type|pyright|pyrefly|ty): ignore`)
   - TODO: overloads
7. TODO: Inline type aliases where used in annotations (so we can determine the `Any`-ness)
8. TODO: Unify `.py` and `.pyi` annotations for each symbol
9. TODO: Filter out any of the non-public symbols (requires following imports)
10. TODO: Collect the type-checker configs (to see which strictness flags are used and which
    type-checkers it supports)
11. TODO: Compute various statistics:
    - coverage (% of public symbols annotated)
    - strict coverage (% of public symbols annotated without `Any`)
    - average overload ratio (function without overloads counts as 1 overload)
    - supported type-checkers + strictness flags
      annotation kind (inline, bundled stubs, typeshed stubs, third-party stubs, etc)
12. TODO: Export the statistics for use in a website/dashboard (e.g. json, csv, or sqlite)

### Async IO

All IO (HTTP requests, subprocesses, file IO, etc) is performed asynchronously using `anyio` and
`httpx` (over HTTP/2). This way we effectively get pipeline parallelism for free (i.e. by doing
other things while waiting on IO, instead of blocking).
Use free-threading for best performance (e.g. use `--python 3.14t` with `uv`).

## Development

To set up a development environment (using [uv](https://github.com/astral-sh/uv)), run:

```bash
uv sync
```

In CI we currently run [ruff](https://github.com/astral-sh/ruff),
[dprint](https://github.com/dprint/dprint), [pyrefly](https://github.com/facebook/pyrefly), and
[pytest](https://github.com/pytest-dev/pytest). It's easy to run them locally as well, just

```bash
uv run ruff check
uv run ruff format

uv run dprint check
uv run dprint fmt

uv run pyrefly check

uv run pytest
```

(`uv run` can be omitted if you manually activated the virtual environment created by `uv`)

You can optionally install and enable lefthook by running:

```bash
uv tool install lefthook --upgrade
uvx lefthook install
uvx lefthook validate
```

For alternative ways of installing lefthook, see <https://github.com/evilmartians/lefthook#install>
