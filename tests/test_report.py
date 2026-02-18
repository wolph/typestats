import libcst as cst
import pytest

from typestats.analyze import (
    ANY,
    EXTERNAL,
    KNOWN,
    UNKNOWN,
    Class,
    Expr,
    Function,
    Overload,
    Param,
    ParamKind,
    Symbol,
    TypeForm,
)
from typestats.report import (
    ClassReport,
    FunctionReport,
    ModuleReport,
    NameReport,
    PackageReport,
    _SlotState,
    _symbol_report,
)

_INT = Expr(cst.parse_expression("int"))
_PARAM = ParamKind.POSITIONAL_OR_KEYWORD

# necessary because `pytest.approx` is not (fully) annotated
# pyright: reportUnknownMemberType=false


class TestSlotState:
    def test_expr(self) -> None:
        assert _SlotState.of(_INT) == (1, 0, 0)

    def test_any(self) -> None:
        assert _SlotState.of(ANY) == (0, 1, 0)

    def test_unknown(self) -> None:
        assert _SlotState.of(UNKNOWN) == (0, 0, 1)

    def test_known(self) -> None:
        assert _SlotState.of(KNOWN) == (0, 0, 0)

    def test_external(self) -> None:
        assert _SlotState.of(EXTERNAL) == (0, 0, 0)


class TestNameReport:
    def test_annotated(self) -> None:
        r = NameReport.from_symbol("x", _INT)
        assert r.n_annotatable == 1
        assert r.n_annotated == 1
        assert r.n_any == 0
        assert r.n_unannotated == 0

    def test_any(self) -> None:
        r = NameReport.from_symbol("x", ANY)
        assert r.n_annotatable == 1
        assert r.n_annotated == 0
        assert r.n_any == 1

    def test_unknown(self) -> None:
        r = NameReport.from_symbol("x", UNKNOWN)
        assert r.n_annotatable == 1
        assert r.n_unannotated == 1

    def test_known_not_annotatable(self) -> None:
        r = NameReport.from_symbol("x", KNOWN)
        assert r.n_annotatable == 0

    def test_external_not_annotatable(self) -> None:
        r = NameReport.from_symbol("x", EXTERNAL)
        assert r.n_annotatable == 0


def _func(overload0: Overload, /, *overloads: Overload) -> Function:
    return Function("f", (overload0, *overloads))


def _overload(params: list[tuple[str, TypeForm]], returns: TypeForm = _INT) -> Overload:
    return Overload(tuple(Param(n, _PARAM, t) for n, t in params), returns)


class TestFunctionReport:
    def test_fully_annotated(self) -> None:
        func = _func(_overload([("a", _INT), ("b", _INT)]))
        r = FunctionReport.from_symbol("f", func)
        assert r.n_annotatable == 3  # 2 params + return
        assert r.n_annotated == 3
        assert r.n_any == 0
        assert r.n_unannotated == 0

    def test_mixed(self) -> None:
        func = _func(_overload([("a", _INT), ("b", UNKNOWN)], returns=ANY))
        r = FunctionReport.from_symbol("f", func)
        assert r.n_annotatable == 3
        assert r.n_annotated == 1
        assert r.n_any == 1
        assert r.n_unannotated == 1

    def test_all_unknown(self) -> None:
        func = _func(_overload([("a", UNKNOWN)], returns=UNKNOWN))
        r = FunctionReport.from_symbol("f", func)
        assert r.n_annotatable == 2
        assert r.n_annotated == 0
        assert r.n_unannotated == 2

    def test_known_params_excluded(self) -> None:
        """KNOWN params (self/cls) don't count as annotatable."""
        func = _func(_overload([("self", KNOWN), ("x", _INT)]))
        r = FunctionReport.from_symbol("f", func)
        assert r.n_annotatable == 2  # x + return, not self
        assert r.n_annotated == 2

    def test_multiple_overloads(self) -> None:
        func = _func(
            _overload([("a", _INT)]),
            _overload([("a", UNKNOWN)], returns=UNKNOWN),
        )
        r = FunctionReport.from_symbol("f", func)
        assert r.n_annotatable == 4  # 2 params + 2 returns
        assert r.n_annotated == 2
        assert r.n_unannotated == 2


class TestClassReport:
    def test_methods_only(self) -> None:
        method = Function("m", (_overload([("x", _INT)]),))
        cls_ = Class("C", (method,))
        r = ClassReport.from_symbol("C", cls_)
        assert len(r.methods) == 1
        assert r.n_annotatable == 2  # x + return
        assert r.n_annotated == 2

    def test_non_function_members_ignored(self) -> None:
        cls_ = Class("C", (KNOWN, _INT, UNKNOWN))
        r = ClassReport.from_symbol("C", cls_)
        assert len(r.methods) == 0
        assert r.n_annotatable == 0

    def test_aggregation(self) -> None:
        m1 = Function("a", (_overload([("x", _INT)]),))
        m2 = Function("b", (_overload([("y", UNKNOWN)], returns=UNKNOWN),))
        cls_ = Class("C", (m1, m2))
        r = ClassReport.from_symbol("C", cls_)
        assert r.n_annotatable == 4
        assert r.n_annotated == 2
        assert r.n_unannotated == 2


class TestSymbolReport:
    def test_function(self) -> None:
        func = _func(_overload([("a", _INT)]))
        r = _symbol_report(Symbol("f", func))
        assert isinstance(r, FunctionReport)

    def test_class(self) -> None:
        cls_ = Class("C", ())
        r = _symbol_report(Symbol("C", cls_))
        assert isinstance(r, ClassReport)

    def test_name(self) -> None:
        r = _symbol_report(Symbol("x", _INT))
        assert isinstance(r, NameReport)

    def test_unknown(self) -> None:
        r = _symbol_report(Symbol("x", UNKNOWN))
        assert isinstance(r, NameReport)
        assert r.n_unannotated == 1


class TestModuleReport:
    def test_name_module(self) -> None:
        m = ModuleReport(path="pkg/sub/mod.py", symbol_reports=())
        assert m.name == "pkg.sub.mod"

    def test_name_module_init(self) -> None:
        m = ModuleReport(path="pkg/__init__.py", symbol_reports=())
        assert m.name == "pkg"

    def test_names(self) -> None:
        m = ModuleReport.from_symbols(
            "mod.py",
            [Symbol("a", _INT), Symbol("b", UNKNOWN)],
        )
        assert m.names == frozenset({"a", "b"})

    def test_counts(self) -> None:
        m = ModuleReport.from_symbols(
            "mod.py",
            [Symbol("a", _INT), Symbol("b", ANY), Symbol("c", UNKNOWN)],
        )
        assert m.n_annotatable == 3
        assert m.n_annotated == 1
        assert m.n_any == 1
        assert m.n_unannotated == 1

    def test_coverage_default(self) -> None:
        """Non-strict: Any counts as annotated."""
        m = ModuleReport.from_symbols("m.py", [Symbol("a", _INT), Symbol("b", ANY)])
        assert m.coverage() == pytest.approx(1)

    def test_coverage_strict(self) -> None:
        """Strict: Any doesn't count as annotated."""
        m = ModuleReport.from_symbols("m.py", [Symbol("a", _INT), Symbol("b", ANY)])
        assert m.coverage(True) == pytest.approx(1 / 2)

    def test_coverage_empty(self) -> None:
        m = ModuleReport(path="m.py", symbol_reports=())
        assert m.coverage() == pytest.approx(0)


class TestPackageReport:
    def _pkg(self, *symbols: Symbol) -> PackageReport:
        mod = ModuleReport.from_symbols("mod.py", list(symbols))
        return PackageReport(package="pkg", module_reports=(mod,), version="1.0.0")

    def test_coverage(self) -> None:
        r = self._pkg(Symbol("a", _INT), Symbol("b", ANY))
        assert r.coverage() == pytest.approx(1)

    def test_coverage_strict(self) -> None:
        r = self._pkg(Symbol("a", _INT), Symbol("b", ANY))
        assert r.coverage(True) == pytest.approx(1 / 2)

    def test_aggregation(self) -> None:
        r = self._pkg(Symbol("a", _INT), Symbol("b", ANY), Symbol("c", UNKNOWN))
        assert r.n_annotatable == 3
        assert r.n_annotated == 1
        assert r.n_any == 1
        assert r.n_unannotated == 1

    def test_typechecker_configs_default_empty(self) -> None:
        r = self._pkg(Symbol("a", _INT))
        assert r.typecheckers == {}

    def test_typechecker_configs_stored(self) -> None:
        mod = ModuleReport.from_symbols("mod.py", [Symbol("a", _INT)])
        r = PackageReport(
            package="pkg",
            module_reports=(mod,),
            version="1.0.0",
            typecheckers={
                "mypy": {"strict": True},
                "ty": {"python-version": "3.14"},
            },
        )
        assert len(r.typecheckers) == 2
        assert "mypy" in r.typecheckers
        assert "ty" in r.typecheckers
