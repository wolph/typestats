import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, Self

import anyio
import mainpy

from typestats import _pypi, analyze
from typestats._http import retry_client
from typestats.index import collect_public_symbols

__all__ = (
    "ClassReport",
    "FunctionReport",
    "ModuleReport",
    "NameReport",
    "PackageReport",
    "SymbolReport",
)

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


class SymbolReport[SymbolT: analyze.TypeForm](Protocol):
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
    def n_annotatable(self) -> int:
        return self.n_annotated + self.n_any + self.n_unannotated

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


def _symbol_report(symbol: analyze.Symbol) -> SymbolReport[Any]:
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
    symbols: tuple[SymbolReport[Any], ...]

    @property
    def name_module(self) -> str:
        """Fully qualified module name derived from the path."""
        parts = self.path.with_suffix("").parts
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(s.name for s in self.symbols)

    @property
    def n_annotatable(self) -> int:
        return sum(s.n_annotatable for s in self.symbols)

    @property
    def n_annotated(self) -> int:
        return sum(s.n_annotated for s in self.symbols)

    @property
    def n_any(self) -> int:
        return sum(s.n_any for s in self.symbols)

    @property
    def n_unannotated(self) -> int:
        return sum(s.n_unannotated for s in self.symbols)

    def coverage(self, strict: bool = False, /) -> float:
        """Coverage ratio. If *strict*, `Any` slots don't count."""
        total = self.n_annotatable
        annotated = self.n_annotated if strict else self.n_annotated + self.n_any
        return annotated / total if total else 0.0

    @classmethod
    def from_symbols(cls, path: str | anyio.Path, symbols: _Symbols) -> Self:
        """Compute stats for a single source file."""
        reports = tuple(_symbol_report(s) for s in symbols)
        return cls(anyio.Path(path), reports)


@dataclass(frozen=True, slots=True)
class PackageReport:
    package: str
    module_reports: tuple[ModuleReport, ...]

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

    # ruff: noqa: T201
    def print(self) -> None:
        """Print a human-readable summary to stdout."""
        for f in self.module_reports:
            typed = f.n_annotated + f.n_any
            print(
                f"{f.path} -> {f.coverage():.1%} "
                f"({typed}/{f.n_annotatable} annotated, "
                f"{f.n_any} Any, {f.n_unannotated} missing)",
            )

        typed = self.n_annotated + self.n_any
        print(
            f"=> Total: {self.coverage():.1%} "
            f"({typed}/{self.n_annotatable} annotated, "
            f"{self.n_any} Any, {self.n_unannotated} missing)",
        )

    @classmethod
    def from_symbols(
        cls,
        package: str,
        base_path: str | anyio.Path,
        public_symbols: Mapping[anyio.Path, _Symbols],
    ) -> Self:
        """Build a `PackageReport` from collected public symbols."""
        files = tuple(
            ModuleReport.from_symbols(source_path.relative_to(base_path), symbols)
            for source_path, symbols in sorted(
                public_symbols.items(),
                key=lambda kv: str(kv[0]),
            )
        )
        return cls(package, files)


@mainpy.main
async def main() -> None:
    package = sys.argv[1] if len(sys.argv) > 1 else "optype"

    async with anyio.TemporaryDirectory() as temp_dir:
        async with retry_client() as client:
            path, _ = await _pypi.download_sdist_latest(client, package, temp_dir)

        public_symbols = await collect_public_symbols(path)
        report = PackageReport.from_symbols(package, path, public_symbols)
        report.print()
