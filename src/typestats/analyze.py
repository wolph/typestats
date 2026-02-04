from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, override

import libcst as cst
import mainpy
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import GatherExportsVisitor, GatherImportsVisitor
from libcst.helpers import get_full_name_for_node
from libcst.metadata import MetadataWrapper, QualifiedNameProvider, QualifiedNameSource

if TYPE_CHECKING:
    import types
    from collections.abc import Mapping, Sequence

__all__ = (
    "Annotation",
    "ModuleImports",
    "ModuleSymbols",
    "Symbol",
    "collect_global_symbols",
)

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
    type_: Annotation

    @override
    def __str__(self) -> str:
        return f"{self.name}: {self.type_}"


@dataclass(slots=True)
class ModuleImports:
    imports: dict[str, str]

    @override
    def __str__(self) -> str:
        return str(self.imports)


@dataclass(slots=True)
class ModuleSymbols:
    symbols: list[Symbol]
    type_aliases: list[Symbol]
    exports_explicit: frozenset[str] | None  # __all__
    exports_implicit: frozenset[str]  # [from _ ]import $name as $name
    imports: ModuleImports


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


class _SymbolVisitor(cst.BatchableCSTVisitor):
    METADATA_DEPENDENCIES = (QualifiedNameProvider,)
    _TYPEFORMS: Final[frozenset[str]] = frozenset({
        "namedtuple",
        "NewType",
        "ParamSpec",
        "TypeAliasType",
        "TypedDict",
        "TypeVar",
        "TypeVarTuple",
    })
    _TYPEALIAS_VALUE_INDEX: Final[int] = 1

    module: Final[cst.Module]
    symbols: Final[list[Symbol]]
    type_aliases: Final[list[Symbol]]
    _class_stack: Final[deque[str]]
    _function_depth: int

    def __init__(self, module: cst.Module, /) -> None:
        self.module = module
        self.symbols = []
        self.type_aliases = []
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

    @classmethod
    def _is_typealias_annotation(cls, annotation: cst.BaseExpression) -> bool:
        full_name = get_full_name_for_node(annotation)
        if full_name is None:
            return False
        return cls._leaf_name(full_name) == "TypeAlias"

    @classmethod
    def _is_special_typeform(cls, expr: cst.BaseExpression) -> bool:
        if not isinstance(expr, cst.Call):
            return False
        full_name = get_full_name_for_node(expr.func)
        if full_name is None:
            return False
        return cls._leaf_name(full_name) in cls._TYPEFORMS

    @classmethod
    def _typealias_value_from_call(
        cls,
        expr: cst.BaseExpression,
    ) -> cst.BaseExpression | None:
        if not isinstance(expr, cst.Call):
            return None
        full_name = get_full_name_for_node(expr.func)
        if full_name is None or cls._leaf_name(full_name) != "TypeAliasType":
            return None
        positional_args = [arg for arg in expr.args if arg.keyword is None]
        if len(positional_args) > cls._TYPEALIAS_VALUE_INDEX:
            return positional_args[cls._TYPEALIAS_VALUE_INDEX].value
        for arg in expr.args:
            if arg.keyword and arg.keyword.value == "value":
                return arg.value
        return None

    def _add_type_alias(self, name_node: cst.Name, value: cst.BaseExpression) -> None:
        name = self._display_name(name_node)
        if self._class_stack and self._function_depth == 0 and "." not in name:
            name = f"{self._class_stack[-1]}.{name}"
        if _is_public(self._leaf_name(name)):
            self.type_aliases.append(Symbol(name, ExprAnnotation(value)))

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
        if node.value is not None and self._is_typealias_annotation(
            node.annotation.annotation,
        ):
            for name_node in _extract_names(node.target):
                self._add_type_alias(name_node, node.value)
            return
        if node.value is not None and self._is_special_typeform(node.value):
            return
        for name_node in _extract_names(node.target):
            self.add(name_node, ExprAnnotation(node.annotation.annotation))

    @override
    def visit_Assign(self, node: cst.Assign) -> None:
        if self._function_depth != 0:
            return
        if typealias_value := self._typealias_value_from_call(node.value):
            for target in node.targets:
                for name_node in _extract_names(target.target):
                    self._add_type_alias(name_node, typealias_value)
            return
        if self._is_special_typeform(node.value):
            return
        for target in node.targets:
            for name_node in _extract_names(target.target):
                self.add(name_node, UNKNOWN)

    @override
    def visit_AugAssign(self, node: cst.AugAssign) -> None:
        if self._function_depth != 0:
            return

    @override
    def visit_TypeAlias(self, node: cst.TypeAlias) -> None:
        if self._function_depth != 0:
            return
        self._add_type_alias(node.name, node.value)


class _ExportsVisitor(GatherExportsVisitor, cst.BatchableCSTVisitor):
    has_explicit_all: bool

    def __init__(self, context: CodemodContext) -> None:
        super().__init__(context)
        self.has_explicit_all = False

    @override
    def get_visitors(self) -> Mapping[str, types.MethodType]:
        # workaround for https://github.com/Instagram/LibCST/pull/1439
        return {
            "visit_AnnAssign": self.visit_AnnAssign,
            "visit_AugAssign": self.visit_AugAssign,
            "visit_Assign": self.visit_Assign,
            "visit_List": self.visit_List,
            "leave_List": self.leave_List,
            "visit_Tuple": self.visit_Tuple,
            "leave_Tuple": self.leave_Tuple,
            "visit_Set": self.visit_Set,
            "leave_Set": self.leave_Set,
            "visit_SimpleString": self.visit_SimpleString,
            "visit_ConcatenatedString": self.visit_ConcatenatedString,
        }

    @staticmethod
    def _is_all_target(target: cst.BaseExpression) -> bool:
        return get_full_name_for_node(target) == "__all__"

    @property
    def exports_explicit(self) -> frozenset[str] | None:
        return (
            frozenset(self.explicit_exported_objects) if self.has_explicit_all else None
        )

    @override
    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        if self._is_all_target(node.target):
            self.has_explicit_all = True
        return super().visit_AnnAssign(node)

    @override
    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        if self._is_all_target(node.target):
            self.has_explicit_all = True
        return super().visit_AugAssign(node)

    @override
    def visit_Assign(self, node: cst.Assign) -> bool:
        if any(self._is_all_target(target.target) for target in node.targets):
            self.has_explicit_all = True
        return super().visit_Assign(node)


class _ImportsVisitor(GatherImportsVisitor, cst.BatchableCSTVisitor):
    @override
    def get_visitors(self) -> Mapping[str, types.MethodType]:
        return {
            "visit_Import": self.visit_Import,
            "visit_ImportFrom": self.visit_ImportFrom,
        }


def collect_global_symbols(source: str, /) -> ModuleSymbols:
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    symbol_visitor = _SymbolVisitor(wrapper.module)
    exports_visitor = _ExportsVisitor(CodemodContext())
    imports_visitor = _ImportsVisitor(CodemodContext())
    wrapper.visit_batched([symbol_visitor, exports_visitor, imports_visitor])

    import_mappings: dict[str, str] = {
        module: module for module in imports_visitor.module_imports
    }
    import_mappings.update(imports_visitor.module_aliases)
    import_mappings.update({
        f"{module}.{obj}": obj
        for module, objects in imports_visitor.object_mapping.items()
        for obj in objects
    })
    import_mappings.update({
        f"{module}.{obj}": alias
        for module, aliases in imports_visitor.alias_mapping.items()
        for obj, alias in aliases
    })
    reexports = frozenset(
        name
        for aliases in imports_visitor.alias_mapping.values()
        for name, alias in aliases
        if name == alias
    )
    imports = ModuleImports(imports=import_mappings)

    return ModuleSymbols(
        symbols=symbol_visitor.symbols,
        type_aliases=symbol_visitor.type_aliases,
        exports_explicit=exports_visitor.exports_explicit,
        exports_implicit=reexports,
        imports=imports,
    )


@mainpy.main
def example() -> None:
    src = """
from typing import TypeAlias, TypeVar
from a import spam as spam, ham as bacon

__all__ = ["SPAM1", "func", "MyClass"]

SPAM1 = 123
SPAM2: int = 123

_T = TypeVar("_T")

Ham: TypeAlias = bytes | int
type Bacon = str | float

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
    print("Imports:", module_symbols.imports)  # noqa: T201
    print("Exports (explicit):", module_symbols.exports_explicit)  # noqa: T201
    print("Exports (implicit):", module_symbols.exports_implicit)  # noqa: T201

    print()  # noqa: T201
    for symbol in module_symbols.type_aliases:
        print(f"{symbol.name} = {symbol.type_}")  # noqa: T201

    print()  # noqa: T201
    for symbol in module_symbols.symbols:
        print(f"{symbol.name}: {symbol.type_}")  # noqa: T201
