import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, Self

if TYPE_CHECKING:
    from _typeshed import StrPath

    from typestats.typecheckers import TypeCheckerConfig, TypeCheckerName

import anyio
import mainpy

from typestats import analyze

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
        # pyrefly: ignore[bad-return]
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
    typecheckers: Mapping[TypeCheckerName, TypeCheckerConfig] = field(
        default_factory=dict,
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
        for f in self.module_reports:
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
    async def from_path(cls, pkg: str, path: StrPath, /) -> Self:
        """Build a `PackageReport` by analysing the package at *path*.

        Runs `collect_public_symbols` and `discover_configs` concurrently.
        """
        from typestats.index import collect_public_symbols  # noqa: PLC0415
        from typestats.typecheckers import discover_configs  # noqa: PLC0415

        symbols: Mapping[anyio.Path, _Symbols] = {}
        configs: Mapping[TypeCheckerName, TypeCheckerConfig] = {}

        async def _collect() -> None:
            nonlocal symbols
            symbols = await collect_public_symbols(path)

        async def _discover() -> None:
            nonlocal configs
            configs = await discover_configs(path)

        async with anyio.create_task_group() as tg:
            tg.start_soon(_collect)
            tg.start_soon(_discover)

        files = tuple(
            ModuleReport.from_symbols(src_path.relative_to(path), symbols)
            for src_path, symbols in symbols.items()
        )
        return cls(pkg, files, configs)


@mainpy.main
async def main() -> None:
    from typestats import _pypi  # noqa: PLC0415
    from typestats._http import retry_client  # noqa: PLC0415

    package = sys.argv[1] if len(sys.argv) > 1 else "optype"

    async with anyio.TemporaryDirectory() as temp_dir:
        async with retry_client() as client:
            path, _ = await _pypi.download_sdist_latest(client, package, temp_dir)

        report = await PackageReport.from_path(package, path)
        report.print()
