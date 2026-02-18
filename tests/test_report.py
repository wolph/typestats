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
    @pytest.mark.parametrize(
        ("typeform", "expected"),
        [
            (_INT, (1, 0, 0)),
            (ANY, (0, 1, 0)),
            (UNKNOWN, (0, 0, 1)),
            (KNOWN, (0, 0, 0)),
            (EXTERNAL, (0, 0, 0)),
        ],
        ids=["expr", "any", "unknown", "known", "external"],
    )
    def test_slot_state(
        self,
        typeform: TypeForm,
        expected: tuple[int, int, int],
    ) -> None:
        assert _SlotState.of(typeform) == expected


class TestNameReport:
    @pytest.mark.parametrize(
        ("typeform", "n_annotatable", "n_annotated", "n_any", "n_unannotated"),
        [
            (_INT, 1, 1, 0, 0),
            (ANY, 1, 0, 1, 0),
            (UNKNOWN, 1, 0, 0, 1),
            (KNOWN, 0, 0, 0, 0),
            (EXTERNAL, 0, 0, 0, 0),
        ],
        ids=["annotated", "any", "unknown", "known", "external"],
    )
    def test_from_symbol(
        self,
        typeform: TypeForm,
        n_annotatable: int,
        n_annotated: int,
        n_any: int,
        n_unannotated: int,
    ) -> None:
        r = NameReport.from_symbol("x", typeform)
        assert r.n_annotatable == n_annotatable
        assert r.n_annotated == n_annotated
        assert r.n_any == n_any
        assert r.n_unannotated == n_unannotated


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
        assert r.n_overloads == 1

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
        assert r.n_overloads == 2


class TestClassReport:
    def test_methods_only(self) -> None:
        method = Function("m", (_overload([("x", _INT)]),))
        cls_ = Class("C", (method,))
        r = ClassReport.from_symbol("C", cls_)
        assert len(r.methods) == 1
        assert r.n_annotatable == 2  # x + return
        assert r.n_annotated == 2
        assert r.n_functions == 0
        assert r.n_methods == 1
        assert r.n_method_overloads == 1

    def test_non_function_members_ignored(self) -> None:
        cls_ = Class("C", (KNOWN, _INT, UNKNOWN))
        r = ClassReport.from_symbol("C", cls_)
        assert len(r.methods) == 0
        assert r.n_annotatable == 0
        assert r.n_functions == 0
        assert r.n_methods == 0

    def test_aggregation(self) -> None:
        m1 = Function("a", (_overload([("x", _INT)]),))
        m2 = Function("b", (_overload([("y", UNKNOWN)], returns=UNKNOWN),))
        cls_ = Class("C", (m1, m2))
        r = ClassReport.from_symbol("C", cls_)
        assert r.n_annotatable == 4
        assert r.n_annotated == 2
        assert r.n_unannotated == 2

    def test_overloaded_methods(self) -> None:
        m1 = Function(
            "a",
            (_overload([("x", _INT)]), _overload([("x", UNKNOWN)])),
        )
        m2 = Function("b", (_overload([("y", _INT)]),))
        cls_ = Class("C", (m1, m2))
        r = ClassReport.from_symbol("C", cls_)
        assert r.n_functions == 0
        assert r.n_methods == 2
        assert r.n_method_overloads == 3  # m1 has 2 overloads + m2 has 1


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

    def test_entity_counts(self) -> None:
        func = _func(_overload([("a", _INT)]))
        overloaded = _func(
            _overload([("a", _INT)]),
            _overload([("a", UNKNOWN)]),
        )
        cls_ = Class("C", ())
        m = ModuleReport.from_symbols(
            "mod.py",
            [
                Symbol("f", func),
                Symbol("g", overloaded),
                Symbol("C", cls_),
                Symbol("x", _INT),
                Symbol("y", UNKNOWN),
            ],
        )
        assert m.n_functions == 2  # f + g (empty class has no methods)
        assert m.n_methods == 0
        assert m.n_function_overloads == 3  # f has 1 + g has 2
        assert m.n_method_overloads == 0
        assert m.n_classes == 1
        assert m.n_names == 2

    def test_entity_counts_empty(self) -> None:
        m = ModuleReport(path="m.py", symbol_reports=())
        assert m.n_functions == 0
        assert m.n_methods == 0
        assert m.n_function_overloads == 0
        assert m.n_method_overloads == 0
        assert m.n_classes == 0
        assert m.n_names == 0

    def test_overloads_from_class_methods(self) -> None:
        overloaded_method = Function(
            "m",
            (
                _overload([("x", _INT)]),
                _overload([("x", UNKNOWN)]),
                _overload([("x", ANY)]),
            ),
        )
        cls_ = Class("C", (overloaded_method,))
        m = ModuleReport.from_symbols("mod.py", [Symbol("C", cls_)])
        assert m.n_functions == 0
        assert m.n_methods == 1
        assert m.n_function_overloads == 0
        assert m.n_method_overloads == 3  # 3 overloads from the class method

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

    def test_entity_counts(self) -> None:
        func = _func(_overload([("a", _INT)]))
        method = Function("m", (_overload([("x", _INT)]),))
        cls_ = Class("C", (method,))
        r = self._pkg(Symbol("f", func), Symbol("C", cls_), Symbol("x", _INT))
        assert r.n_functions == 1  # f
        assert r.n_methods == 1  # C.m
        assert r.n_function_overloads == 1
        assert r.n_method_overloads == 1
        assert r.n_classes == 1
        assert r.n_names == 1

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
