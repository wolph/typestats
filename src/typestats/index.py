import enum
import logging
import os
import re
from collections import defaultdict, deque
from itertools import chain
from typing import TYPE_CHECKING, Final

import anyio
import mainpy
from libcst.helpers import get_full_name_for_node

from typestats import _ruff, analyze

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from _typeshed import StrPath


__all__ = "collect_public_symbols", "list_sources"


_logger: Final = logging.getLogger(__name__)

_RE_INIT: Final = re.compile(r"^__init__\.pyi?$")
_RE_STUBS_DIR: Final = re.compile(r"^(.+)-stubs$")

# Directory names to exclude from analysis (tests, benchmarks, docs, etc.)
_EXCLUDED_DIR_NAMES: Final[frozenset[str]] = frozenset({
    ".spin",
    "_examples",
    "benchmarks",
    "doc",
    "docs",
    "examples",
    "tests",
    "tools",
})
# File names to exclude from analysis.
_EXCLUDED_FILE_NAMES: Final[frozenset[str]] = frozenset({"conftest.py"})
# Module-level dunder names that are not real symbols for typing purposes.
_MODULE_DUNDERS: Final[frozenset[str]] = frozenset({"__all__", "__doc__"})
# Fully qualified names that are considered equivalent to having no annotation.
_ANY_FQNS: Final[frozenset[str]] = frozenset({
    "typing.Any",
    "typing_extensions.Any",
    "_typeshed.Incomplete",
    "_typeshed.MaybeNone",
    "_typeshed.sentinel",
    "_typeshed.AnnotationForm",
})


def _is_public(name: str) -> bool:
    return not name.startswith("_") or name.endswith("__")


def _is_excluded_path(path: str, /, *, prefix: str = "") -> bool:
    """Check if a source path contains an excluded directory or file name.

    When *prefix* is given, it is stripped before inspecting the remaining
    path components so that the project directory itself does not trigger
    false positives (e.g. a project stored under a ``tests/`` directory).
    """
    rel = path.removeprefix(prefix).lstrip("/")
    parts = rel.split("/")
    if parts[-1] in _EXCLUDED_FILE_NAMES:
        return True
    return bool(_EXCLUDED_DIR_NAMES.intersection(parts))


async def _analyze_graph(project_dir: StrPath, /, *opts: str) -> dict[str, list[str]]:
    """Run `ruff analyze graph` and clean self/parent-package dependencies."""
    graph = await _ruff.analyze_graph(project_dir, *opts)

    # Build both absolute and CWD-relative prefixes so we can strip the
    # project directory from graph keys regardless of how ruff reports them.
    abs_path = await anyio.Path(project_dir).resolve()
    abs_prefix = str(abs_path).replace(os.sep, "/").rstrip("/") + "/"
    try:
        cwd = await anyio.Path.cwd()
        rel = str(abs_path.relative_to(cwd)).replace(os.sep, "/")
        rel_prefix = rel.rstrip("/") + "/"
    except ValueError:
        # project_dir is not under cwd; ruff will use absolute paths
        rel_prefix = abs_prefix

    def _excluded(path: str) -> bool:
        return _is_excluded_path(
            path,
            prefix=abs_prefix if path.startswith("/") else rel_prefix,
        )

    def _skip_dep(node: str, dep: str, deps: list[str]) -> bool:
        return (
            # break self-dependencies
            dep == node
            # break self/super package dependencies
            or (node.count("/") >= dep.count("/") and "/__init__.py" in dep)
            # remove .py deps that also have a .pyi counterpart
            or (dep.endswith(".py") and dep + "i" in deps)
            # exclude test/benchmark/doc directories
            or _excluded(dep)
        )

    return {
        node: list(dict.fromkeys(dep for dep in deps if not _skip_dep(node, dep, deps)))
        for node, deps in graph.items()
        if not _excluded(node)
    }


async def list_sources(project_dir: StrPath, /) -> list[anyio.Path]:
    """List all source files in the given project directory."""
    graph = await _analyze_graph(project_dir, "--type-checking-imports")
    return [anyio.Path(p) for p in graph]


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

    def _in_namespace_package(source: anyio.Path) -> bool:
        """True when *source* sits in a directory without ``__init__`` that is
        nested inside a proper package directory.  Such directories are not
        importable (e.g. vendored third-party code) and should be excluded."""

        if (parent := source.parent) in init_dirs:
            return False

        # Walk up and check whether any ancestor is a package directory.
        current = parent.parent
        while current != parent:
            if current in init_dirs:
                return True
            parent = current
            current = current.parent
        return False

    # Exclude files that live in non-package subdirectories of a package
    source_paths = [p for p in source_paths if not _in_namespace_package(p)]

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


def _resolve_expr_name(name: str, import_map: Mapping[str, str], mod: str) -> str:
    """Resolve a dotted name to its FQN using the module's import map.

    Looks up the whole *name* first, then tries to resolve just the first
    component (for ``import typing; typing.Any`` style access), and finally
    falls back to treating it as a module-local name.
    """
    if name in import_map:
        return import_map[name]

    first, _, rest = name.partition(".")
    if rest and first in import_map:
        return f"{import_map[first]}.{rest}"
    return f"{mod}.{name}"


def _resolves_to_any(fqn: str, alias_targets: Mapping[str, str]) -> bool:
    """Check if *fqn* ultimately resolves to ``typing.Any`` through alias chains."""
    seen: set[str] = set()
    current = fqn
    while current not in seen:
        if current in _ANY_FQNS:
            return True
        if current not in alias_targets:
            return False
        seen.add(current)
        current = alias_targets[current]
    return False


def _unfold_any(
    type_: analyze.TypeForm,
    import_map: Mapping[str, str],
    mod: str,
    alias_targets: Mapping[str, str],
) -> analyze.TypeForm:
    """Replace ``Expr`` annotations that resolve to ``Any`` with ``ANY``.

    Walks *type_* recursively so that function parameters, return types, and
    class members are all checked.
    """
    match type_:
        case analyze.Expr(expr=expr):
            name = get_full_name_for_node(expr)
            if name is not None:
                fqn = _resolve_expr_name(name, import_map, mod)
                if _resolves_to_any(fqn, alias_targets):
                    return analyze.ANY
            return type_
        case analyze.Function(name=fn_name, overloads=overloads):
            new_overloads = tuple(
                analyze.Overload(
                    tuple(
                        analyze.Param(
                            p.name,
                            p.kind,
                            _unfold_any(p.annotation, import_map, mod, alias_targets),
                        )
                        for p in o.params
                    ),
                    _unfold_any(o.returns, import_map, mod, alias_targets),
                )
                for o in overloads
            )
            return analyze.Function(fn_name, (new_overloads[0], *new_overloads[1:]))
        case analyze.Class(name=cls_name, members=members):
            return analyze.Class(
                cls_name,
                tuple(_unfold_any(m, import_map, mod, alias_targets) for m in members),
            )
        case _:
            return type_


def _is_public_module(module_path: str, /) -> bool:
    """Check if all parts of a dotted module path are public."""
    return all(not part.startswith("_") for part in module_path.split("."))


async def collect_public_symbols(  # noqa: C901, PLR0912, PLR0914, PLR0915
    project_dir: StrPath,
    /,
) -> Mapping[anyio.Path, Sequence[analyze.Symbol]]:
    """Collect public, fully qualified symbols from a package by source path.

    Symbols are attributed to their *origin* source file.  When a public module
    re-exports a symbol from a private module, the origin's fully qualified name
    (and source path) is used rather than the re-exporting module's.
    """
    sources = list(await list_sources(project_dir))

    # Drop .py when .pyi exists for the same file
    pyi_set = frozenset(str(s) for s in sources if s.suffix == ".pyi")
    sources = [s for s in sources if not (s.suffix == ".py" and f"{s}i" in pyi_set)]

    module_paths = sources_to_module_paths(sources)
    top_level = frozenset(m.split(".", 1)[0] for m in module_paths)

    # Step 1: Parse all modules, build flat symbol table (fqn → (path, type))
    all_local: dict[str, tuple[anyio.Path, analyze.TypeForm]] = {}
    module_data: dict[str, dict[anyio.Path, analyze.ModuleSymbols]] = {}
    module_locals: dict[str, dict[str, str]] = {}  # mod → {name: fqn}
    for mod, paths in module_paths.items():
        entries: dict[anyio.Path, analyze.ModuleSymbols] = {}
        local: dict[str, str] = {}
        for path in sorted(paths, key=lambda p: not p.name.endswith(".pyi")):
            pkg = (
                mod
                if _RE_INIT.match(path.name)
                else (mod.rsplit(".", 1)[0] if "." in mod else "")
            )
            syms = analyze.collect_symbols(await path.read_text(), package_name=pkg)
            entries[path] = syms

            for name, type_ in chain(
                ((a.name, a.value) for a in syms.type_aliases),
                ((s.name, s.type_) for s in syms.symbols),
            ):
                if "." not in name:
                    fqn = f"{mod}.{name}"
                    all_local.setdefault(fqn, (path, type_))
                    local.setdefault(name, fqn)

        module_data[mod] = entries
        module_locals[mod] = local

    # Step 1.5: Resolve type aliases that point to `Any`
    #
    # Build a table mapping type-alias FQNs to the FQN of their RHS
    # value, then walk every entry in ``all_local`` and replace any
    # ``Expr`` annotation that resolves to ``typing.Any`` (directly or
    # through alias chains) with ``ANY``.
    alias_targets: dict[str, str] = {}
    path_to_mod: dict[anyio.Path, str] = {}
    module_import_maps: dict[str, dict[str, str]] = {}
    for mod, entries in module_data.items():
        imap: dict[str, str] = {}
        for syms in entries.values():
            imap.update(syms.imports)
        module_import_maps[mod] = imap

        for path in entries:
            path_to_mod[path] = mod

        for syms in entries.values():
            for alias in syms.type_aliases:
                if (
                    "." not in alias.name  # skip class-level aliases
                    and isinstance(alias.value, analyze.Expr)
                    and (value_name := get_full_name_for_node(alias.value.expr))
                ):
                    alias_fqn = f"{mod}.{alias.name}"
                    alias_targets[alias_fqn] = _resolve_expr_name(value_name, imap, mod)

    if alias_targets:
        for fqn, (path, type_) in list(all_local.items()):
            if (p2m := path_to_mod.get(path)) is None:
                continue

            new_type = _unfold_any(
                type_,
                module_import_maps.get(p2m, {}),
                p2m,
                alias_targets,
            )
            if new_type is not type_:
                all_local[fqn] = (path, new_type)

    # Step 2: Compute module exports with origin tracing
    exports_cache: dict[str, dict[str, str]] = {}

    def resolve_origin(fqn: str, _seen: frozenset[str] = frozenset()) -> str:
        """Follow import chains to the original local definition."""
        if fqn in all_local or fqn in _seen or "." not in fqn:
            return fqn
        mod, name = fqn.rsplit(".", 1)
        seen = _seen | {fqn}
        for syms in module_data.get(mod, {}).values():
            if target := dict(syms.imports).get(name):
                return resolve_origin(target, seen)
            for wc_mod in syms.imports_wildcard:
                if name in (wc := module_exports(wc_mod)):
                    return resolve_origin(wc[name], seen)
        return fqn

    def module_exports(mod: str) -> dict[str, str]:  # noqa: C901, PLR0912
        """Return ``{export_name: origin_fqn}`` for all names exported by *mod*."""
        if mod in exports_cache:
            return exports_cache[mod]
        exports_cache[mod] = {}  # break cycles

        import_map: dict[str, str] = {}
        wc_mods: list[str] = []
        explicit: frozenset[str] | None = None
        dynamic: list[str] = []
        implicit: set[str] = set()

        for syms in module_data.get(mod, {}).values():
            import_map.update(syms.imports)
            wc_mods.extend(syms.imports_wildcard)
            if syms.exports_explicit is not None:
                explicit = syms.exports_explicit
            dynamic.extend(syms.exports_explicit_dynamic)
            implicit.update(syms.exports_implicit)

        local = module_locals.get(mod, {})

        wc: dict[str, str] = {}
        for wc_mod in wc_mods:
            if wc_mod.split(".", 1)[0] in top_level:
                for name, origin in module_exports(wc_mod).items():
                    wc.setdefault(name, origin)

        if explicit is not None:
            names = set(explicit)
            for src in dynamic:
                if (t := import_map.get(src, src)) in module_data:
                    names |= module_exports(t).keys()
        elif _is_public_module(mod):
            names = {n for n in {*local, *implicit} if _is_public(n)}
        else:
            names = {n for n in {*local, *import_map} if _is_public(n) and n != "*"}

        names -= _MODULE_DUNDERS

        result: dict[str, str] = {}
        for name in names:
            if name in local:
                result[name] = local[name]
            elif name in wc:
                result[name] = wc[name]
            elif target := import_map.get(name):
                result[name] = resolve_origin(target)
            else:
                result[name] = f"{mod}.{name}"

        exports_cache[mod] = result
        return result

    # Step 3: Mark public symbols and build result
    public: dict[str, tuple[anyio.Path, analyze.TypeForm]] = {}
    for mod, entries in module_data.items():
        if not _is_public_module(mod):
            continue
        first_path = next(iter(entries))
        for name, origin in module_exports(mod).items():
            if origin in module_data:
                continue  # skip submodule references
            if origin in all_local:
                public.setdefault(origin, all_local[origin])
            else:
                type_ = (
                    analyze.EXTERNAL
                    if origin.split(".", 1)[0] not in top_level
                    else analyze.UNKNOWN
                )
                public.setdefault(f"{mod}.{name}", (first_path, type_))

    result: defaultdict[anyio.Path, list[analyze.Symbol]] = defaultdict(list)
    for fqn, (path, type_) in sorted(public.items()):
        result[path].append(analyze.Symbol(fqn, type_))
    return result


@mainpy.main
async def example() -> None:
    import sys  # noqa: PLC0415
    import time  # noqa: PLC0415

    from typestats import _pypi  # noqa: PLC0415
    from typestats._http import retry_client  # noqa: PLC0415

    package = sys.argv[1] if len(sys.argv) > 1 else "optype"

    t0 = time.monotonic()
    async with anyio.TemporaryDirectory() as temp_dir:
        async with retry_client() as client:
            path, _ = await _pypi.download_sdist_latest(client, package, temp_dir)

        total_annotated = 0
        total_annotatable = 0
        public_symbols = await collect_public_symbols(path)
        for source_path, symbols in sorted(public_symbols.items()):
            rel_path = source_path.relative_to(path)
            file_annotated = 0
            file_annotatable = 0
            names_unannotated: list[str] = []
            for s in symbols:
                a, t = analyze.annotation_counts(s.type_)
                file_annotated += a
                file_annotatable += t
                if a < t:
                    names_unannotated.append(s.name)
            total_annotated += file_annotated
            total_annotatable += file_annotatable
            print(  # noqa: T201
                f"{rel_path} -> {file_annotated}/{file_annotatable} annotated"
                f" ({', '.join(sorted(names_unannotated))})",
            )

        pct = total_annotated / total_annotatable * 100 if total_annotatable else 0
        print(  # noqa: T201
            f"\nTotal: {total_annotated}/{total_annotatable} annotated ({pct:.1f}%)",
        )

    elapsed = time.monotonic() - t0
    _logger.info("Total runtime: %.2fs", elapsed)
