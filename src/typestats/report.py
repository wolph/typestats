# ruff: noqa: PLC0415

import asyncio
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, Self, cast

if TYPE_CHECKING:
    from _typeshed import StrPath

import anyio
import mainpy

from typestats import analyze
from typestats.typecheckers import TypeCheckerConfigDict, TypeCheckerName

__all__ = "ClassReport", "FunctionReport", "ModuleReport", "NameReport", "PackageReport"

type _Symbols = Sequence[analyze.Symbol]
type _Max1 = Literal[0, 1]


class _SlotState(NamedTuple):
    annotated: _Max1
    any: _Max1
    unannotated: _Max1

    @classmethod
    def of(cls, ty: analyze.TypeForm) -> Self:
        """Classify a single annotation slot."""
        match ty:
            case analyze.Expr():
                return cls(1, 0, 0)
            case analyze.ANY:
                return cls(0, 1, 0)
            case analyze.UNKNOWN:
                return cls(0, 0, 1)
            case _:  # KNOWN / EXTERNAL
                return cls(0, 0, 0)


class _SymbolReport[SymbolT: analyze.TypeForm = Any](Protocol):
    """Common interface for per-symbol reports."""

    @property
    def name(self) -> str: ...
    @property
    def n_annotatable(self) -> int: ...
    @property
    def n_annotated(self) -> int: ...
    @property
    def n_any(self) -> int: ...
    @property
    def n_unannotated(self) -> int: ...

    @classmethod
    def from_symbol(cls, name: str, ty: SymbolT, /) -> Self: ...


@dataclass(frozen=True, slots=True)
class NameReport:
    """Report for a module-level variable or constant (single slot)."""

    name: str
    n_annotated: _Max1
    n_any: _Max1
    n_unannotated: _Max1

    @property
    def n_annotatable(self) -> _Max1:
        return cast("_Max1", self.n_annotated + self.n_any + self.n_unannotated)

    @classmethod
    def from_symbol(cls, name: str, ty: analyze.TypeForm, /) -> Self:
        s = _SlotState.of(ty)
        return cls(name, s.annotated, s.any, s.unannotated)


@dataclass(frozen=True, slots=True)
class FunctionReport:
    """Report for a function/method; counts individual param + return slots."""

    name: str
    n_annotated: int
    n_any: int
    n_unannotated: int

    @property
    def n_annotatable(self) -> int:
        return self.n_annotated + self.n_any + self.n_unannotated

    @classmethod
    def from_symbol(cls, name: str, ty: analyze.Function, /) -> Self:
        annotated = any_ = unannotated = 0
        for overload in ty.overloads:
            for ann in [*(p.annotation for p in overload.params), overload.returns]:
                s = _SlotState.of(ann)
                annotated += s.annotated
                any_ += s.any
                unannotated += s.unannotated

        return cls(name, annotated, any_, unannotated)


@dataclass(frozen=True, slots=True)
class ClassReport:
    """Report for a class; aggregates its method reports.

    Class-level attributes are ignored (for now?).
    """

    name: str
    methods: tuple[FunctionReport, ...]

    @property
    def n_annotatable(self) -> int:
        return sum(m.n_annotatable for m in self.methods)

    @property
    def n_annotated(self) -> int:
        return sum(m.n_annotated for m in self.methods)

    @property
    def n_any(self) -> int:
        return sum(m.n_any for m in self.methods)

    @property
    def n_unannotated(self) -> int:
        return sum(m.n_unannotated for m in self.methods)

    @classmethod
    def from_class(cls, name: str, class_: analyze.Class) -> Self:
        methods = [
            FunctionReport.from_symbol(member.name, member)
            for member in class_.members
            if isinstance(member, analyze.Function)
        ]
        return cls(name, tuple(methods))

    @classmethod
    def from_symbol(cls, name: str, ty: analyze.Class, /) -> Self:
        return cls.from_class(name, ty)


def _symbol_report(symbol: analyze.Symbol) -> _SymbolReport[Any]:
    """Create the appropriate report for a symbol."""
    match symbol.type_:
        case analyze.Function():
            return FunctionReport.from_symbol(symbol.name, symbol.type_)
        case analyze.Class():
            return ClassReport.from_symbol(symbol.name, symbol.type_)
        case _:
            return NameReport.from_symbol(symbol.name, symbol.type_)


@dataclass(frozen=True, slots=True)
class ModuleReport:
    path: anyio.Path
    symbol_reports: tuple[_SymbolReport, ...]

    @property
    def name(self) -> str:
        """Fully qualified module name."""
        parts = self.path.with_suffix("").parts
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(s.name for s in self.symbol_reports)

    @property
    def n_annotatable(self) -> int:
        return sum(s.n_annotatable for s in self.symbol_reports)

    @property
    def n_annotated(self) -> int:
        return sum(s.n_annotated for s in self.symbol_reports)

    @property
    def n_any(self) -> int:
        return sum(s.n_any for s in self.symbol_reports)

    @property
    def n_unannotated(self) -> int:
        return sum(s.n_unannotated for s in self.symbol_reports)

    def coverage(self, strict: bool = False, /) -> float:
        """
        Coverage ratio.

        Args:
            strict (bool): If `True`, `Any` types won't be counted as annotated.
        """
        total = self.n_annotatable
        annotated = self.n_annotated if strict else self.n_annotated + self.n_any
        return annotated / total if total else 0.0

    @classmethod
    def from_symbols(cls, path: StrPath, symbols: _Symbols) -> Self:
        return cls(anyio.Path(path), tuple(_symbol_report(s) for s in symbols))


@dataclass(frozen=True, slots=True)
class PackageReport:
    package: str
    module_reports: tuple[ModuleReport, ...]
    typecheckers: Mapping[TypeCheckerName, TypeCheckerConfigDict] = field(
        default_factory=dict[TypeCheckerName, TypeCheckerConfigDict],
    )

    @property
    def n_annotatable(self) -> int:
        return sum(m.n_annotatable for m in self.module_reports)

    @property
    def n_annotated(self) -> int:
        return sum(m.n_annotated for m in self.module_reports)

    @property
    def n_any(self) -> int:
        return sum(m.n_any for m in self.module_reports)

    @property
    def n_unannotated(self) -> int:
        return sum(m.n_unannotated for m in self.module_reports)

    def coverage(self, strict: bool = False, /) -> float:
        """Coverage ratio. If *strict*, `Any` slots don't count."""
        total = self.n_annotatable
        annotated = self.n_annotated if strict else self.n_annotated + self.n_any
        return annotated / total if total else 0.0

    def print(self) -> None:
        """Print a human-readable summary to stdout."""
        for f in sorted(self.module_reports, key=lambda r: r.path):
            typed = f.n_annotated + f.n_any
            print(  # noqa: T201
                f"{f.path} -> {f.coverage():.1%} "
                f"({typed}/{f.n_annotatable} annotated, "
                f"{f.n_any} Any, {f.n_unannotated} missing)",
            )

        typed = self.n_annotated + self.n_any
        print(  # noqa: T201
            f"=> Total: {self.coverage():.1%} "
            f"({typed}/{self.n_annotatable} annotated, "
            f"{self.n_any} Any, {self.n_unannotated} missing)",
        )
        if self.typecheckers:
            checkers = ", ".join(sorted(self.typecheckers))
            print(f"   Type-checkers: {checkers}")  # noqa: T201

    @classmethod
    async def from_path(
        cls,
        pkg: str,
        path: StrPath,
        /,
        *,
        stubs_path: StrPath | None = None,
    ) -> Self:
        """Build a `PackageReport` by analysing the package at *path*.

        When *stubs_path* is given (a companion ``{pkg}-stubs`` sdist),
        symbols from the stubs overlay take priority and any original symbol
        whose module is covered by stubs but absent from those stubs is
        marked ``UNKNOWN``.

        Runs ``collect_public_symbols`` (and optionally the stubs collection)
        and ``discover_configs`` concurrently.
        """

        from typestats.index import collect_public_symbols, merge_stubs_overlay
        from typestats.typecheckers import discover_configs

        coros: list[Any] = [
            collect_public_symbols(
                path,
                trace_origins=stubs_path is None,
                package_name=pkg,
            ),
            discover_configs(path),
        ]
        if stubs_path is not None:
            coros.append(
                collect_public_symbols(
                    stubs_path,
                    trace_origins=False,
                    package_name=pkg,
                ),
            )

        results: list[Any] = await asyncio.gather(*coros)
        symbols: Mapping[anyio.Path, _Symbols] = results[0]
        configs: Mapping[TypeCheckerName, TypeCheckerConfigDict] = results[1]

        if stubs_path is not None:
            symbols = merge_stubs_overlay(symbols, results[2])

        stubs_p = anyio.Path(stubs_path) if stubs_path is not None else None

        def _rel(src: anyio.Path) -> anyio.Path:
            try:
                return src.relative_to(stubs_p or path)
            except ValueError:
                return src.relative_to(path)

        files = tuple(
            ModuleReport.from_symbols(_rel(src_path), syms)
            for src_path, syms in symbols.items()
        )
        return cls(pkg, files, configs)


@mainpy.main
async def main() -> None:
    import re

    from typestats import _pypi
    from typestats._http import retry_client

    package = sys.argv[1] if len(sys.argv) > 1 else "optype"

    async with anyio.TemporaryDirectory() as temp_dir:
        async with retry_client() as client:
            if m := re.match(r"^(.+)-stubs$", package):
                # Stubs package: download both base and stubs concurrently
                base_name = m.group(1)
                (base_path, _), (stubs_path, _) = await asyncio.gather(
                    _pypi.download_sdist_latest(client, base_name, temp_dir),
                    _pypi.download_sdist_latest(client, package, temp_dir),
                )
                report = await PackageReport.from_path(
                    base_name,
                    base_path,
                    stubs_path=stubs_path,
                )
            else:
                # Base package: analyze standalone
                path, _ = await _pypi.download_sdist_latest(client, package, temp_dir)
                report = await PackageReport.from_path(package, path)

        report.print()
