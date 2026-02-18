# ruff: noqa: PLC0415

import asyncio
import sys
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple, Self, cast

if TYPE_CHECKING:
    from _typeshed import StrPath

import anyio
import mainpy
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    NonNegativeInt,
    computed_field,
)

from typestats import analyze
from typestats.typecheckers import TypeCheckerConfigDict, TypeCheckerName

__all__ = "ClassReport", "FunctionReport", "ModuleReport", "NameReport", "PackageReport"

type _Symbols = Sequence[analyze.Symbol]
type _Max1 = Literal[0, 1]

type _AnySymbolReport = Annotated[
    NameReport | FunctionReport | ClassReport,
    Discriminator("kind"),
]


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


class NameReport(BaseModel):
    """Report for a module-level variable or constant (single slot)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["name"] = "name"
    name: str
    n_annotated: _Max1
    n_any: _Max1
    n_unannotated: _Max1

    @computed_field
    @property
    def n_annotatable(self) -> _Max1:
        return cast("_Max1", self.n_annotated + self.n_any + self.n_unannotated)

    n_functions: Literal[0] = Field(0, exclude=True)
    n_methods: Literal[0] = Field(0, exclude=True)
    n_function_overloads: Literal[0] = Field(0, exclude=True)
    n_method_overloads: Literal[0] = Field(0, exclude=True)
    n_classes: Literal[0] = Field(0, exclude=True)
    n_names: Literal[1] = Field(1, exclude=True)

    @classmethod
    def from_symbol(cls, name: str, ty: analyze.TypeForm, /) -> Self:
        s = _SlotState.of(ty)
        return cls(
            name=name,
            n_annotated=s.annotated,
            n_any=s.any,
            n_unannotated=s.unannotated,
        )


class FunctionReport(BaseModel):
    """Report for a function/method; counts individual param + return slots."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["function"] = "function"
    name: str
    n_annotated: NonNegativeInt
    n_any: NonNegativeInt
    n_unannotated: NonNegativeInt
    n_overloads: NonNegativeInt

    @computed_field
    @property
    def n_annotatable(self) -> NonNegativeInt:
        return self.n_annotated + self.n_any + self.n_unannotated

    n_functions: Literal[1] = Field(1, exclude=True)
    n_methods: Literal[0] = Field(0, exclude=True)
    n_method_overloads: Literal[0] = Field(0, exclude=True)
    n_classes: Literal[0] = Field(0, exclude=True)
    n_names: Literal[0] = Field(0, exclude=True)

    @computed_field
    @property
    def n_function_overloads(self) -> NonNegativeInt:
        return self.n_overloads

    @classmethod
    def from_symbol(cls, name: str, ty: analyze.Function, /) -> Self:
        annotated = any_ = unannotated = 0
        for overload in ty.overloads:
            for ann in [*(p.annotation for p in overload.params), overload.returns]:
                s = _SlotState.of(ann)
                annotated += s.annotated
                any_ += s.any
                unannotated += s.unannotated

        return cls(
            name=name,
            n_annotated=annotated,
            n_any=any_,
            n_unannotated=unannotated,
            n_overloads=len(ty.overloads),
        )


class ClassReport(BaseModel):
    """Report for a class; aggregates its method reports.

    Class-level attributes are ignored (for now?).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["class"] = "class"
    name: str
    methods: tuple[FunctionReport, ...]

    @computed_field
    @property
    def n_annotatable(self) -> NonNegativeInt:
        return sum(m.n_annotatable for m in self.methods)

    @computed_field
    @property
    def n_annotated(self) -> NonNegativeInt:
        return sum(m.n_annotated for m in self.methods)

    @computed_field
    @property
    def n_any(self) -> NonNegativeInt:
        return sum(m.n_any for m in self.methods)

    @computed_field
    @property
    def n_unannotated(self) -> NonNegativeInt:
        return sum(m.n_unannotated for m in self.methods)

    @computed_field
    @property
    def n_functions(self) -> Literal[0]:
        return 0

    @computed_field
    @property
    def n_methods(self) -> NonNegativeInt:
        return len(self.methods)

    @computed_field
    @property
    def n_function_overloads(self) -> Literal[0]:
        return 0

    @computed_field
    @property
    def n_method_overloads(self) -> NonNegativeInt:
        return sum(m.n_overloads for m in self.methods)

    n_classes: Literal[1] = Field(1, exclude=True)
    n_names: Literal[0] = Field(0, exclude=True)

    @classmethod
    def from_class(cls, name: str, class_: analyze.Class) -> Self:
        methods = [
            FunctionReport.from_symbol(member.name, member)
            for member in class_.members
            if isinstance(member, analyze.Function)
        ]
        return cls(name=name, methods=tuple(methods))

    @classmethod
    def from_symbol(cls, name: str, ty: analyze.Class, /) -> Self:
        return cls.from_class(name, ty)


def _symbol_report(symbol: analyze.Symbol) -> _AnySymbolReport:
    """Create the appropriate report for a symbol."""
    match symbol.type_:
        case analyze.Function():
            return FunctionReport.from_symbol(symbol.name, symbol.type_)
        case analyze.Class():
            return ClassReport.from_symbol(symbol.name, symbol.type_)
        case _:
            return NameReport.from_symbol(symbol.name, symbol.type_)


class ModuleReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    symbol_reports: tuple[_AnySymbolReport, ...]
    type_ignores: tuple[analyze.IgnoreComment, ...] = ()

    @computed_field
    @property
    def name(self) -> str:
        """Fully qualified module name."""
        parts = PurePosixPath(self.path).with_suffix("").parts
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    @computed_field
    @property
    def names(self) -> frozenset[str]:
        return frozenset(s.name for s in self.symbol_reports)

    @computed_field
    @property
    def n_annotatable(self) -> NonNegativeInt:
        return sum(s.n_annotatable for s in self.symbol_reports)

    @computed_field
    @property
    def n_annotated(self) -> NonNegativeInt:
        return sum(s.n_annotated for s in self.symbol_reports)

    @computed_field
    @property
    def n_any(self) -> NonNegativeInt:
        return sum(s.n_any for s in self.symbol_reports)

    @computed_field
    @property
    def n_unannotated(self) -> NonNegativeInt:
        return sum(s.n_unannotated for s in self.symbol_reports)

    @computed_field
    @property
    def n_functions(self) -> NonNegativeInt:
        return sum(s.n_functions for s in self.symbol_reports)

    @computed_field
    @property
    def n_methods(self) -> NonNegativeInt:
        return sum(s.n_methods for s in self.symbol_reports)

    @computed_field
    @property
    def n_function_overloads(self) -> NonNegativeInt:
        return sum(s.n_function_overloads for s in self.symbol_reports)

    @computed_field
    @property
    def n_method_overloads(self) -> NonNegativeInt:
        return sum(s.n_method_overloads for s in self.symbol_reports)

    @computed_field
    @property
    def n_classes(self) -> NonNegativeInt:
        return sum(s.n_classes for s in self.symbol_reports)

    @computed_field
    @property
    def n_names(self) -> NonNegativeInt:
        return sum(s.n_names for s in self.symbol_reports)

    @computed_field
    @property
    def n_type_ignores(self) -> NonNegativeInt:
        return len(self.type_ignores)

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
    def from_symbols(
        cls,
        path: StrPath,
        symbols: _Symbols,
        /,
        *,
        type_ignores: Sequence[analyze.IgnoreComment] = (),
    ) -> Self:
        return cls(
            path=anyio.Path(path).as_posix(),
            symbol_reports=tuple(_symbol_report(s) for s in symbols),
            type_ignores=tuple(type_ignores),
        )


class PackageReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    package: str
    module_reports: tuple[ModuleReport, ...]
    version: str
    typecheckers: dict[TypeCheckerName, TypeCheckerConfigDict] = Field(
        default_factory=dict,
    )

    @computed_field
    @property
    def n_modules(self) -> NonNegativeInt:
        return len(self.module_reports)

    @computed_field
    @property
    def n_annotatable(self) -> NonNegativeInt:
        return sum(m.n_annotatable for m in self.module_reports)

    @computed_field
    @property
    def n_annotated(self) -> NonNegativeInt:
        return sum(m.n_annotated for m in self.module_reports)

    @computed_field
    @property
    def n_any(self) -> NonNegativeInt:
        return sum(m.n_any for m in self.module_reports)

    @computed_field
    @property
    def n_unannotated(self) -> NonNegativeInt:
        return sum(m.n_unannotated for m in self.module_reports)

    @computed_field
    @property
    def n_functions(self) -> NonNegativeInt:
        return sum(m.n_functions for m in self.module_reports)

    @computed_field
    @property
    def n_methods(self) -> NonNegativeInt:
        return sum(m.n_methods for m in self.module_reports)

    @computed_field
    @property
    def n_function_overloads(self) -> NonNegativeInt:
        return sum(m.n_function_overloads for m in self.module_reports)

    @computed_field
    @property
    def n_method_overloads(self) -> NonNegativeInt:
        return sum(m.n_method_overloads for m in self.module_reports)

    @computed_field
    @property
    def n_classes(self) -> NonNegativeInt:
        return sum(m.n_classes for m in self.module_reports)

    @computed_field
    @property
    def n_names(self) -> NonNegativeInt:
        return sum(m.n_names for m in self.module_reports)

    @computed_field
    @property
    def type_ignores(self) -> tuple[analyze.IgnoreComment, ...]:
        result: tuple[analyze.IgnoreComment, ...] = ()
        for m in self.module_reports:
            result += m.type_ignores
        return result

    @computed_field
    @property
    def n_type_ignores(self) -> NonNegativeInt:
        return sum(m.n_type_ignores for m in self.module_reports)

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
            f"=> {self.package} {self.version}: {self.coverage():.1%} "
            f"({typed}/{self.n_annotatable} annotated, "
            f"{self.n_any} Any, {self.n_unannotated} missing)",
        )
        print(  # noqa: T201
            f"   {self.n_modules} modules, "
            f"{self.n_functions} functions ({self.n_function_overloads} overloads), "
            f"{self.n_methods} methods ({self.n_method_overloads} overloads), "
            f"{self.n_classes} classes, {self.n_names} names, "
            f"{self.n_type_ignores} type-ignore comments",
        )
        if self.typecheckers:
            checkers = ", ".join(sorted(self.typecheckers))
            print(f"   Type-checkers: {checkers}")  # noqa: T201

    @classmethod
    async def from_path(
        cls,
        pkg: str,
        path: StrPath,
        version: str,
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
        from typestats.index import PublicSymbols

        pub: PublicSymbols = results[0]
        configs: Mapping[TypeCheckerName, TypeCheckerConfigDict] = results[1]

        if stubs_path is not None:
            stubs_pub: PublicSymbols = results[2]
            symbols = merge_stubs_overlay(pub.symbols, stubs_pub.symbols)
            # Merge type-ignore comments from both
            type_ignores = dict(pub.type_ignores)
            for p, comments in stubs_pub.type_ignores.items():
                type_ignores[p] = type_ignores.get(p, ()) + comments
        else:
            symbols = pub.symbols
            type_ignores = pub.type_ignores

        stubs_p = anyio.Path(stubs_path) if stubs_path is not None else None

        def _rel(src: anyio.Path) -> anyio.Path:
            try:
                return src.relative_to(stubs_p or path)
            except ValueError:
                return src.relative_to(path)

        files = tuple(
            ModuleReport.from_symbols(
                _rel(src_path),
                syms,
                type_ignores=type_ignores.get(src_path, ()),
            )
            for src_path, syms in symbols.items()
        )
        return cls(
            package=pkg,
            module_reports=files,
            version=version,
            typecheckers=dict(configs),
        )


@mainpy.main
async def main() -> None:
    import re

    from packaging.utils import parse_sdist_filename

    from typestats import _pypi
    from typestats._http import retry_client

    package = sys.argv[1] if len(sys.argv) > 1 else "optype"

    async with anyio.TemporaryDirectory() as temp_dir:
        async with retry_client() as client:
            if m := re.match(r"^(.+)-stubs$", package):
                # Stubs package: download both base and stubs concurrently
                base_name = m.group(1)
                (base_path, base_sdist), (stubs_path, _) = await asyncio.gather(
                    _pypi.download_sdist_latest(client, base_name, temp_dir),
                    _pypi.download_sdist_latest(client, package, temp_dir),
                )
                _, base_ver = parse_sdist_filename(base_sdist["filename"])
                report = await PackageReport.from_path(
                    base_name,
                    base_path,
                    str(base_ver),
                    stubs_path=stubs_path,
                )
            else:
                # Base package: analyze standalone
                path, sdist = await _pypi.download_sdist_latest(
                    client,
                    package,
                    temp_dir,
                )
                _, ver = parse_sdist_filename(sdist["filename"])
                report = await PackageReport.from_path(package, path, str(ver))

        report.print()
