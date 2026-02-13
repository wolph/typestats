# typestats

A tool to analyze the type annotation coverage of Python projects on PyPI.

> [!IMPORTANT]
> This project is a work-in-progress and is not yet functional.

## Implementation details

### High-level Pipeline

For a given project:

1. Query PyPI for the latest version
2. Download the latest (non-yanked) sdist, and extract it
3. TODO: If there exists a `types-{project}` or `{project}-stubs` package on PyPI, repeat steps
   1-2 for that package, and merge the stubs into the main package source tree.
4. Compute the import graph using `ruff analyze graph`
5. Best-effort topological sort of the import graph (cycles are broken gracefully)
6. Filter to files transitively reachable from public modules (skip tests, tools, etc.)
7. For each reachable file, parse it using `libcst`, and extract:
   - all annotatable global symbols and their type annotations
   - the `__all__` exports (if defined)
   - imports and implicit re-exports (i.e. `from a import b as b`)
   - type aliases (`_: TypeAlias = ...` and `type _ = ...`)
   - type-ignore comments (`# (type|pyright|pyrefly|ty): ignore`)
   - overloaded functions/methods
8. TODO: Inline type aliases where used in annotations (so we can determine the `Any`-ness)
9. TODO: Unify `.py` and `.pyi` annotations for each symbol
10. Resolve public symbols via iterative fixed-point (handles import cycles)
11. Collect the type-checker configs to see which strictness flags are used and which
    type-checkers it supports (mypy, (based)pyright, pyrefly, ty, zuban)
12. TODO: Compute various statistics:
    - coverage (% of public symbols annotated)
    - strict coverage (% of public symbols annotated without `Any`)
    - average overload ratio (function without overloads counts as 1 overload)
    - supported type-checkers + strictness flags
      annotation kind (inline, bundled stubs, typeshed stubs, third-party stubs, etc)
13. TODO: Export the statistics for use in a website/dashboard (e.g. json, csv, or sqlite)

### Symbol collection

Per-module (via `libcst`):

- **Imports**: `import m`, `import m as a`, `from m import x`, `from m import x as a`
- **Wildcard imports**: `from m import *`
- **Explicit exports**: `__all__ = [...]` (list, tuple, or set literals)
- **Dynamic exports**: `__all__ += other.__all__`
  ([spec](https://typing.python.org/en/latest/spec/distributing.html#library-interface-public-and-private-symbols))
- **Implicit re-exports**: `from m import x as x`, `import m as m`
  ([spec](https://typing.python.org/en/latest/spec/distributing.html#import-conventions))
- **Type aliases**: `X: TypeAlias = ...`, `type X = ...`, `X = TypeAliasType("X", ...)`
- **Name aliases**: `X = Y` where `Y` is a local symbol (viz. type alias) or an imported name (viz.
  import alias)
- **Special typeforms** (excluded from symbols): `TypeVar`, `ParamSpec`, `TypeVarTuple`, `NewType`,
  `TypedDict`, `namedtuple`
- **Annotated variables**: `x: T` and `x: T = ...`
- **Functions/methods**: full parameter signatures with `self`/`cls` inference
- **Overloaded functions**: `@overload` signatures collected and merged
- **Properties**: `@property` / `@cached_property` (return type used as annotation)
- **Classes**: including nested attribute annotations
- **Enum members**: auto-detected as `KNOWN` (via `Enum`/`IntEnum`/`StrEnum`/`Flag`/... bases)
- **Dataclass / NamedTuple / TypedDict fields**: auto-detected as `KNOWN` (annotated by definition)
- **Type-ignore comments**: `# type: ignore[...]`, `# pyrefly:ignore[...]`, etc.
- **`Annotated` unwrapping**: `Annotated[T, ...]` → `T`
  ([spec](https://typing.python.org/en/latest/spec/qualifiers.html#annotated))
- **Aliased typing imports**: `import typing as t` resolved via `QualifiedNameProvider`

Cross-module (via import graph):

- **Import graph**: `ruff analyze graph` with/without `TYPE_CHECKING` branches; import cycles handled
  gracefully via best-effort topological sort and iterative fixed-point resolution
- **Reachability filtering**: only files transitively reachable from public modules are parsed,
  skipping tests, benchmarks, and internal tooling
- **Excluded directories and files**: the following directories are automatically excluded from
  analysis: `.spin`, `_examples`, `benchmarks`, `doc`, `docs`, `examples`, `tests`, `tools`.
  The file `conftest.py` is also excluded wherever it appears.
- **Namespace package exclusion**: directories without `__init__.py` nested inside a proper package
  are excluded (e.g. vendored third-party code like `numpy/linalg/lapack_lite/`)
- **Public symbol resolution**: follows imports across modules, iterating until convergence
- **Private module re-exports**: symbols re-exported from `_private` modules via `__all__`
- **Wildcard re-export expansion**: `from _internal import *` resolved to concrete symbols
- **External vs unknown**: imported symbols from external packages marked `EXTERNAL`, not `UNKNOWN`,
  and excluded from coverage denominator
- **Unresolved `__all__` names**: names listed in `__all__` that cannot be resolved to any local
  definition or import are treated as `UNKNOWN`—matching the behavior of type-checkers, which would
  infer these as `Any` or `Unknown` (e.g. modules using `__getattr__` for lazy loading)
- **Stub file priority**: When both `.py` and `.pyi` files exist for the same module, only the
  `.pyi` stub is used—matching the behavior of type-checkers
  ([spec](https://typing.python.org/en/latest/spec/distributing.html#import-resolution-ordering))
- **`py.typed` detection**: `YES`, `NO`, `PARTIAL`, or `STUBS` (for `-stubs` packages)
  ([spec](https://typing.python.org/en/latest/spec/distributing.html#packaging-type-information))

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
