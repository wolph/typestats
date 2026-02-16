# Copilot Instructions for `typestats`

## Project Goal

`typestats` quantifies the static typing quality of open-source Python packages published on PyPI.
By computing metrics such as **type-coverage** (the percentage of public symbols that carry
meaningful type annotations), it helps the Python community identify which projects would benefit
most from investment in improving their static typing quality.

The end-goal is a dataset (and eventually a dashboard) that ranks packages by typing completeness,
so maintainers, contributors, and sponsors can prioritize effort where it matters most.

## How It Works

For a given PyPI project the tool runs an end-to-end pipeline:

1. **Fetch** — download the latest sdist from PyPI (and any companion stub package).
2. **Graph** — compute the import graph via `ruff analyze graph`.
3. **Filter** — keep only modules reachable from public entry-points (skip tests, benchmarks,
   docs, vendored code, etc.).
4. **Parse** — use `libcst` to extract every annotatable symbol (variables, functions, methods,
   classes, properties, overloads, aliases, etc.) together with its type annotation (or lack
   thereof), building a flat symbol table of all local definitions.
5. **Resolve** — compute each public module's exports, tracing re-export chains back to their
   origin definition. Symbols are attributed to the source file where they are defined, not where
   they are re-exported.
6. **Measure** — compute coverage and other statistics.
7. **Export** — output the results for consumption by a website or dashboard.

## Architecture & Key Modules

| Module            | Responsibility                                                                    |
| ----------------- | --------------------------------------------------------------------------------- |
| `_pypi.py`        | PyPI HTTP queries and sdist downloading                                           |
| `_ruff.py`        | Subprocess wrapper around `ruff analyze graph`                                    |
| `_typeshed.py`    | Typeshed-related helpers                                                          |
| `analyze.py`      | `libcst`-based per-file symbol extraction (annotations, overloads, classes, etc.) |
| `index.py`        | Cross-module import resolution, public API construction, origin-based attribution |
| `report.py`       | Slot-level coverage reporting (`SymbolReport` protocol, module/package reports)   |
| `typecheckers.py` | Detection of type-checker configs and strictness flags                            |

## Conventions

- **Python ≥ 3.14** — the project targets the latest Python, and can optionally run on a
  free-threaded (e.g. `3.14t`) build for performance.
- **Async IO** — all IO (HTTP, subprocesses, file reads) is async via `anyio` + `httpx` (HTTP/2).
- **`libcst`** — used instead of `ast` because it preserves comments and formatting, which are
  needed for detecting type-ignore directives and other metadata.
- **Type-checking** — the project itself is strictly typed. Pyrefly and Pyright are both configured
  in strict mode in `pyproject.toml`.
- **Linting & formatting** — `ruff` for linting/formatting, `dprint` for non-Python formatting.
- **Testing** — `pytest` with `--doctest-modules`. Test fixtures live in `tests/fixtures/` as small
  installable packages.
- **Builds** — managed with `uv` (`uv sync`, `uv run pytest`, etc.).

## Key Domain Concepts

- **TypeForm** — the core data structure representing a symbol's type annotation. Variants include
  `UNKNOWN` (unannotated), `KNOWN` (annotated by construction, e.g. enum members or dataclass
  fields), and `EXTERNAL` (imported from an outside package).
- **`is_annotated()`** — the central helper that decides whether a `TypeForm` counts as
  "annotated". Classes are annotated only when *all* their members are annotated; functions are
  annotated when their full signature (including overloads) is annotated.
- **`__all__` resolution** — names in `__all__` that can't be resolved are treated as `UNKNOWN`,
  matching type-checker semantics.
- **Stub priority** — when both `.py` and `.pyi` exist, only the `.pyi` is used.
- **Private re-exports** — symbols re-exported from `_private` modules via `__all__` are followed
  correctly.
