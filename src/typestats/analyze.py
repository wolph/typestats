from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, override

import libcst as cst
import mainpy
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import GatherExportsVisitor
from libcst.helpers import get_full_name_for_node
from libcst.metadata import MetadataWrapper, QualifiedNameProvider, QualifiedNameSource

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = "Annotation", "ModuleSymbols", "Symbol", "collect_global_symbols"

_EMPTY_MODULE: Final[cst.Module] = cst.Module([])

type Annotation = Unknown | ExprAnnotation | FunctionAnnotation | ClassAnnotation


class ParamKind(StrEnum):
    # matches inspect.Parameter.kind
    POSITIONAL_ONLY = "positional-only"
    POSITIONAL_OR_KEYWORD = "positional or keyword"
    VAR_POSITIONAL = "variadic positional"
    KEYWORD_ONLY = "keyword-only"
    VAR_KEYWORD = "variadic keyword"


@dataclass(frozen=True, slots=True)
class Unknown:
    @override
    def __str__(self) -> str:
        return "?"


UNKNOWN: Final[Unknown] = Unknown()


@dataclass(frozen=True, slots=True)
class ExprAnnotation:
    expr: cst.BaseExpression

    @override
    def __str__(self) -> str:
        return _EMPTY_MODULE.code_for_node(self.expr).strip()


@dataclass(frozen=True, slots=True)
class ParamAnnotation:
    name: str
    kind: ParamKind
    annotation: Annotation

    @override
    def __str__(self) -> str:
        prefix = {
            ParamKind.VAR_POSITIONAL: "*",
            ParamKind.VAR_KEYWORD: "**",
        }.get(self.kind, "")
        return f"{prefix}{self.name}: {self.annotation}"


@dataclass(frozen=True, slots=True)
class FunctionAnnotation:
    params: list[ParamAnnotation]
    returns: Annotation

    @override
    def __str__(self) -> str:
        params = ", ".join(str(param) for param in self.params)
        return f"({params}) -> {self.returns}"


@dataclass(frozen=True, slots=True)
class ClassAnnotation:
    name: str

    @override
    def __str__(self) -> str:
        return f"type[{self.name}]"


@dataclass(slots=True)
class Symbol:
    name: str
    annotation: Annotation

    @override
    def __str__(self) -> str:
        return f"{self.name}: {self.annotation}"


@dataclass(slots=True)
class ModuleSymbols:
    symbols: list[Symbol]
    all_: set[str] | None


class _SymbolVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (QualifiedNameProvider,)
    module: Final[cst.Module]
    symbols: Final[list[Symbol]]
    _class_stack: Final[deque[str]]
    _function_depth: int

    def __init__(self, module: cst.Module, /) -> None:
        self.module = module
        self.symbols = []
        self._class_stack = deque()
        self._function_depth = 0

    def _qualified_name(self, node: cst.CSTNode) -> str | None:
        names = self.get_metadata(QualifiedNameProvider, node, default=set())
        if not names:
            return None
        preferred = next(
            (qn for qn in names if qn.source is not QualifiedNameSource.LOCAL),
            None,
        )
        return (preferred or next(iter(names))).name

    @staticmethod
    def _display_from_qualified(qualified: str) -> str:
        parts = qualified.split(".")
        return ".".join(parts[1:]) if len(parts) > 1 else qualified

    def _display_name_for(self, node: cst.CSTNode, fallback: str) -> str:
        qualified = self._qualified_name(node)
        if qualified is None:
            return fallback
        return self._display_from_qualified(qualified)

    def _display_name(self, name_node: cst.Name) -> str:
        return self._display_name_for(name_node, name_node.value)

    def _class_display_name(self, node: cst.ClassDef) -> str:
        return self._display_name_for(node, node.name.value)

    @staticmethod
    def _leaf_name(name: str) -> str:
        return name.rsplit(".", 1)[-1]

    @staticmethod
    def _param_annotation(param: cst.Param, kind: ParamKind) -> ParamAnnotation:
        annotation = (
            ExprAnnotation(param.annotation.annotation) if param.annotation else UNKNOWN
        )
        return ParamAnnotation(name=param.name.value, kind=kind, annotation=annotation)

    @classmethod
    def _extend_params(
        cls,
        params: list[ParamAnnotation],
        items: Sequence[cst.Param],
        kind: ParamKind,
    ) -> None:
        params.extend(cls._param_annotation(param, kind) for param in items)

    def add(self, name_node: cst.Name, annotation: Annotation) -> None:
        name = self._display_name(name_node)
        if self._class_stack and self._function_depth == 0 and "." not in name:
            name = f"{self._class_stack[-1]}.{name}"
        if _is_public(self._leaf_name(name)):
            self.symbols.append(Symbol(name, annotation))

    @staticmethod
    def signature_annotation(node: cst.FunctionDef) -> FunctionAnnotation:
        params: list[ParamAnnotation] = []
        for node_params, kind in [
            (node.params.posonly_params, ParamKind.POSITIONAL_ONLY),
            (node.params.params, ParamKind.POSITIONAL_OR_KEYWORD),
            (node.params.kwonly_params, ParamKind.KEYWORD_ONLY),
        ]:
            _SymbolVisitor._extend_params(params, node_params, kind)

        star_arg = node.params.star_arg
        if isinstance(star_arg, cst.Param):
            params.append(
                _SymbolVisitor._param_annotation(star_arg, ParamKind.VAR_POSITIONAL),
            )

        star_kwarg = node.params.star_kwarg
        if isinstance(star_kwarg, cst.Param):
            params.append(
                _SymbolVisitor._param_annotation(star_kwarg, ParamKind.VAR_KEYWORD),
            )

        returns = ExprAnnotation(node.returns.annotation) if node.returns else UNKNOWN
        return FunctionAnnotation(params=params, returns=returns)

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if self._function_depth != 0:
            return False
        class_name = self._class_display_name(node)
        if _is_public(self._leaf_name(class_name)):
            self.symbols.append(Symbol(class_name, ClassAnnotation(name=class_name)))
        self._class_stack.append(class_name)
        return True

    @override
    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        del original_node
        if self._class_stack:
            self._class_stack.pop()

    @override
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if self._function_depth == 0:
            if self._class_stack:
                name = node.name.value
                if _is_public(name):
                    decorators = {
                        self._leaf_name(full)
                        for dec in node.decorators
                        if (full := get_full_name_for_node(dec.decorator))
                    }
                    if "property" in decorators or "cached_property" in decorators:
                        annotation = (
                            ExprAnnotation(node.returns.annotation)
                            if node.returns
                            else UNKNOWN
                        )
                    else:
                        annotation = self.signature_annotation(node)
                    self.add(node.name, annotation)
            else:
                self.add(node.name, self.signature_annotation(node))
        self._function_depth += 1
        return True

    @override
    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        del original_node
        self._function_depth -= 1

    @override
    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if self._function_depth != 0:
            return
        for name_node in _extract_names(node.target):
            self.add(name_node, ExprAnnotation(node.annotation.annotation))

    @override
    def visit_Assign(self, node: cst.Assign) -> None:
        if self._function_depth != 0:
            return
        for target in node.targets:
            for name_node in _extract_names(target.target):
                self.add(name_node, UNKNOWN)

    @override
    def visit_AugAssign(self, node: cst.AugAssign) -> None:
        if self._function_depth != 0:
            return


def collect_global_symbols(source: str, /) -> ModuleSymbols:
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    visitor = _SymbolVisitor(wrapper.module)
    exports_visitor = GatherExportsVisitor(CodemodContext())
    wrapper.visit(visitor)
    wrapper.visit(exports_visitor)
    exports = set(exports_visitor.explicit_exported_objects)
    return ModuleSymbols(symbols=visitor.symbols, all_=exports or None)


def _is_public(name: str) -> bool:
    return not name.startswith("__") and not name.endswith("__") and name != "_"


def _extract_names(expr: cst.BaseExpression) -> list[cst.Name]:
    match expr:
        case cst.Name():
            return [expr]
        case cst.Tuple(elements=elements) | cst.List(elements=elements):
            names: list[cst.Name] = []
            for element in elements:
                if element.value is not None:
                    names.extend(_extract_names(element.value))
            return names
        case _:
            return []


@mainpy.main
def example() -> None:
    src = """
SPAM1 = 123
SPAM2: int = 123

def func(a: int, *args, **kwds) -> str:
    def nested(c: int) -> float:
        return c * 2
    return str(nested(a) + nested(b))

class MyClass[T]:
    attr: T

    def method(self, x: int, y) -> str:
        def nested(a: int) -> float:
            return a * 2
        return str(nested(x) + nested(y))
"""
    module_symbols = collect_global_symbols(src)
    print("Exports:", module_symbols.all_)  # noqa: T201
    print()  # noqa: T201

    for symbol in module_symbols.symbols:
        print(f"{symbol.name}: {symbol.annotation}")  # noqa: T201
