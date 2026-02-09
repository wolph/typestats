import enum
import graphlib
import logging
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Final

import anyio
import mainpy

from typestats import _ruff, analyze

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from _typeshed import StrPath


__all__ = "collect_public_symbols", "list_sources"


_logger: Final = logging.getLogger(__name__)

_RE_INIT: Final = re.compile(r"^__init__\.pyi?$")
_RE_STUBS_DIR: Final = re.compile(r"^(.+)-stubs$")

type _SymbolMap = dict[str, analyze.Symbol]


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

    init_files = frozenset(p for p in source_paths if _RE_INIT.match(p.name))
    init_dirs = frozenset(p.parent for p in init_files)

    def _module_path(source: anyio.Path) -> str:
        parts = deque[str]()
        current_dir = source.parent
        while current_dir in init_dirs:
            dir_name = current_dir.name
            if match := _RE_STUBS_DIR.match(dir_name):
                parts.appendleft(match.group(1))
                break
            parts.appendleft(dir_name)
            current_dir = current_dir.parent

        if not _RE_INIT.match(source.name):
            parts.append(source.stem)

        return ".".join(parts) if parts else source.stem

    module_paths: defaultdict[str, set[anyio.Path]] = defaultdict(set)
    for path in source_paths:
        module_paths[_module_path(path)].add(path)

    return {k: frozenset(ps) for k, ps in module_paths.items()}


def _is_public_module(module_path: str, /) -> bool:
    """Check if all parts of a dotted module path are public."""
    return all(not part.startswith("_") for part in module_path.split("."))


def _local_symbol_candidates(symbols: analyze.ModuleSymbols, /) -> _SymbolMap:
    """Get locally-defined symbol candidates (type aliases, symbols)."""
    # Build with increasing priority: aliases < symbols
    candidates: _SymbolMap = {}
    for alias in symbols.type_aliases:
        if "." not in alias.name:
            candidates[alias.name] = analyze.Symbol(alias.name, alias.value)
    for sym in symbols.symbols:
        if "." not in sym.name:
            candidates[sym.name] = sym
    return candidates


def _package_name(module_path: str, source_path: anyio.Path, /) -> str:
    """Package name: the module itself for __init__, parent module otherwise."""
    if _RE_INIT.match(source_path.name):
        return module_path
    return module_path.rsplit(".", 1)[0] if "." in module_path else ""


async def _collect_module_symbols(
    module_paths: Mapping[str, frozenset[anyio.Path]],
    /,
) -> Mapping[str, Mapping[anyio.Path, analyze.ModuleSymbols]]:
    """Collect symbols per module."""
    result: dict[str, dict[anyio.Path, analyze.ModuleSymbols]] = {}
    for mod, paths in module_paths.items():
        mod_result: dict[anyio.Path, analyze.ModuleSymbols] = {}
        for path in paths:
            text = await path.read_text()
            package_name = _package_name(mod, path)
            mod_result[path] = analyze.collect_symbols(text, package_name=package_name)
        result[mod] = mod_result
    return result


class _SymbolResolver:
    """Resolves public symbols across modules in topological order."""

    def __init__(
        self,
        module_symbols: Mapping[str, Mapping[anyio.Path, analyze.ModuleSymbols]],
        sources_by_module: Mapping[str, Sequence[anyio.Path]],
        top_level_packages: frozenset[str],
    ) -> None:
        self._module_symbols = module_symbols
        self._sources_by_module = sources_by_module
        self._top_level_packages = top_level_packages
        self._resolved: defaultdict[str, dict[anyio.Path, _SymbolMap]] = defaultdict(
            dict,
        )
        self._resolved_union: defaultdict[str, _SymbolMap] = defaultdict(dict)
        self._candidates_cache: dict[str, _SymbolMap] = {}

    def _module_candidates(self, module_path: str, /) -> _SymbolMap:
        """Get all symbol candidates for a module (cached, merges source files)."""
        if module_path in self._candidates_cache:
            return self._candidates_cache[module_path]
        if module_path not in self._module_symbols:
            self._candidates_cache[module_path] = {}
            return {}

        candidates: _SymbolMap = {}
        for source_path in self._sources_by_module.get(module_path, ()):
            symbols = self._module_symbols[module_path][source_path]
            for name, symbol in _local_symbol_candidates(symbols).items():
                candidates.setdefault(name, symbol)

        self._candidates_cache[module_path] = candidates
        return candidates

    def _resolve_import(
        self,
        name: str,
        target: str,
        /,
        *,
        has_explicit_exports: bool,
    ) -> analyze.Symbol | None:
        """Resolve an imported symbol to its public type, if applicable."""
        if target in self._module_symbols:
            if not _is_public_module(target):
                return analyze.Symbol(name, analyze.UNKNOWN)
            return None

        if (
            "." not in target
            or target.rsplit(".", 1)[0].split(".", 1)[0] not in self._top_level_packages
        ):
            return None

        target_module, target_name = target.rsplit(".", 1)
        if not _is_public_module(target_module):
            if not has_explicit_exports:
                return None
            origin = (
                self._resolved_union.get(target_module, {}).get(target_name)
                or self._module_candidates(target_module).get(target_name)
            )  # fmt: skip
            type_ = origin.type_ if origin is not None else analyze.UNKNOWN
        else:
            origin = self._resolved_union.get(target_module, {}).get(target_name)
            if origin is None:
                return None
            type_ = origin.type_

        return analyze.Symbol(name, type_)

    def _expand_wildcard_imports(
        self,
        symbols: analyze.ModuleSymbols,
        import_map: dict[str, str],
        /,
    ) -> _SymbolMap:
        """Expand wildcard imports from internal modules using resolved exports."""
        expanded: _SymbolMap = {}
        for wc_module in symbols.imports_wildcard:
            if wc_module.split(".", 1)[0] not in self._top_level_packages:
                continue
            for sym_name, symbol in self._resolved_union.get(wc_module, {}).items():
                expanded.setdefault(sym_name, symbol)
                import_map.setdefault(sym_name, f"{wc_module}.{sym_name}")
        return expanded

    def _resolve_dynamic_all_sources(
        self,
        import_map: Mapping[str, str],
        symbols: analyze.ModuleSymbols,
        /,
    ) -> frozenset[str]:
        """Resolve dynamic __all__ += X.__all__ references to symbol names."""
        if not symbols.exports_explicit_dynamic:
            return frozenset()
        extra_names: set[str] = set()
        for source_name in symbols.exports_explicit_dynamic:
            # Resolve the local name (e.g. "_core") to a module path
            target = import_map.get(source_name)
            if target and target in self._module_symbols:
                # target is the fully qualified module path
                extra_names.update(self._resolved_union.get(target, {}))
            elif source_name in self._module_symbols:
                extra_names.update(self._resolved_union.get(source_name, {}))
        return frozenset(extra_names)

    def _resolve_source(
        self,
        symbols: analyze.ModuleSymbols,
        /,
        *,
        is_private: bool = False,
    ) -> _SymbolMap:
        """Resolve the exported symbols for a single source file."""
        import_map = dict(symbols.imports)
        candidates = _local_symbol_candidates(symbols)

        # Expand wildcard imports from internal modules
        wildcard_symbols = self._expand_wildcard_imports(symbols, import_map)

        if symbols.exports_explicit is not None:
            export_names = frozenset(symbols.exports_explicit)
            export_names |= self._resolve_dynamic_all_sources(import_map, symbols)
        elif is_private:
            # Without __all__, all non-private names available for re-export
            export_names = frozenset(
                n
                for n in {*candidates, *import_map}
                if analyze.is_public(n) and n != "*"
            )
        else:
            # Without __all__, only local symbols and re-exports (PEP 484)
            export_names = frozenset(
                n
                for n in {*candidates, *symbols.exports_implicit}
                if analyze.is_public(n)
            )

        has_explicit_exports = symbols.exports_explicit is not None or is_private
        exports: _SymbolMap = {}
        for name in sorted(export_names):
            sym = candidates.get(name) or wildcard_symbols.get(name)
            if sym is not None:
                exports[name] = sym
            elif target := import_map.get(name):
                resolved = self._resolve_import(
                    name,
                    target,
                    has_explicit_exports=has_explicit_exports,
                )
                if resolved is not None:
                    exports[name] = resolved
        return exports

    def resolve_all(self) -> None:
        """Resolve public exports for all modules."""
        for module_path, entries in self._module_symbols.items():
            is_public = _is_public_module(module_path)
            for symbol_path in self._sources_by_module.get(module_path, ()):
                exports = self._resolve_source(
                    entries[symbol_path],
                    is_private=not is_public,
                )
                self._resolved[module_path][symbol_path] = exports if is_public else {}
                # Populate _resolved_union for downstream module lookups
                union = self._resolved_union[module_path]
                for name, symbol in exports.items():
                    union.setdefault(name, symbol)

    def public_symbols(
        self,
        sources: Iterable[anyio.Path],
        path_to_mod: Mapping[anyio.Path, str],
        /,
    ) -> Mapping[anyio.Path, Sequence[analyze.Symbol]]:
        """Build fully qualified public symbols grouped by source path."""
        result: defaultdict[anyio.Path, list[analyze.Symbol]] = defaultdict(list)
        for symbol_path in sources:
            module_path = path_to_mod.get(symbol_path)
            if not module_path or not _is_public_module(module_path):
                continue
            exports = self._resolved.get(module_path, {}).get(symbol_path, {})
            result[symbol_path] = [
                analyze.Symbol(f"{module_path}.{name}", symbol.type_)
                for name, symbol in exports.items()
            ]
        return result


async def collect_public_symbols(
    project_dir: StrPath,
    /,
) -> Mapping[anyio.Path, Sequence[analyze.Symbol]]:
    """Collect public, fully qualified symbols from a package by source path."""
    sources = list(await list_sources(project_dir))
    module_paths = sources_to_module_paths(sources)

    module_symbols = await _collect_module_symbols(module_paths)

    path_to_mod = {path: mod for mod, paths in module_paths.items() for path in paths}

    # Build module → paths mapping in topological order (single pass)
    sources_by_module: defaultdict[str, list[anyio.Path]] = defaultdict(list)
    for path in sources:
        mod = path_to_mod.get(path)
        if mod is not None and mod in module_symbols:
            sources_by_module[mod].append(path)

    # Sort .pyi files before .py files so stub types take precedence
    # (setdefault in _resolved_union and _module_candidates keeps the first value)
    for mod_paths in sources_by_module.values():
        mod_paths.sort(key=lambda p: not p.name.endswith(".pyi"))

    top_level_packages = frozenset(module.split(".", 1)[0] for module in module_symbols)

    resolver = _SymbolResolver(
        module_symbols=module_symbols,
        sources_by_module=sources_by_module,
        top_level_packages=top_level_packages,
    )
    resolver.resolve_all()
    return resolver.public_symbols(sources, path_to_mod)


@mainpy.main
async def example() -> None:
    import httpx  # noqa: PLC0415

    from typestats import _pypi  # noqa: PLC0415

    async with anyio.TemporaryDirectory() as temp_dir:
        async with httpx.AsyncClient(http2=True) as client:
            path, _detail = await _pypi.download_sdist_latest(
                client,
                "optype",
                temp_dir,
            )

        public_symbols = await collect_public_symbols(path)
        for source_path, symbols in sorted(public_symbols.items()):
            rel_path = source_path.relative_to(path)
            n_total = sum(1 for s in symbols if s.type_ is not analyze.KNOWN)
            n_unknown = sum(1 for s in symbols if s.type_ is analyze.UNKNOWN)
            names_unknown = ", ".join(
                sorted(s.name for s in symbols if s.type_ is analyze.UNKNOWN),
            )
            print(  # noqa: T201
                f"{rel_path} -> {n_total} known, {n_unknown} unknown ({names_unknown})",
            )
