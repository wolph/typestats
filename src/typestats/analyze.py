import re
from collections import defaultdict, deque
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
    from collections.abc import Mapping

__all__ = (
    "IgnoreComment",
    "Imports",
    "ModuleSymbols",
    "Symbol",
    "TypeForm",
    "collect_global_symbols",
)

_EMPTY_MODULE: Final[cst.Module] = cst.Module([])

type TypeForm = Unknown | Known | Expr | Function | Class


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
class Known:
    @override
    def __str__(self) -> str:
        return "_"


KNOWN: Final[Known] = Known()


@dataclass(frozen=True, slots=True)
class Expr:
    expr: cst.BaseExpression

    @override
    def __str__(self) -> str:
        return _EMPTY_MODULE.code_for_node(self.expr).strip()


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    kind: ParamKind
    annotation: TypeForm

    @override
    def __str__(self) -> str:
        prefix = {
            ParamKind.VAR_POSITIONAL: "*",
            ParamKind.VAR_KEYWORD: "**",
        }.get(self.kind, "")
        if isinstance(self.annotation, Known):
            return f"{prefix}{self.name}"
        return f"{prefix}{self.name}: {self.annotation}"


@dataclass(frozen=True, slots=True)
class Overload:
    params: list[Param]
    returns: TypeForm

    @override
    def __str__(self) -> str:
        params = ", ".join(str(param) for param in self.params)
        return f"({params}) -> {self.returns}"


@dataclass(frozen=True, slots=True)
class Function:
    name: str
    overloads: tuple[Overload, *tuple[Overload, ...]]

    def __post_init__(self) -> None:
        if not self.overloads:
            msg = "FunctionOverloads must have at least one signature"
            raise ValueError(msg)

    @override
    def __str__(self) -> str:
        if len(self.overloads) == 1:
            return str(self.overloads[0])
        # an overloaded function type is the intersection of its overloads
        return " & ".join(f"({sig})" for sig in self.overloads)


@dataclass(frozen=True, slots=True)
class Class:
    name: str

    @override
    def __str__(self) -> str:
        return f"type[{self.name}]"


@dataclass(slots=True)
class Symbol:
    name: str
    type_: TypeForm

    @override
    def __str__(self) -> str:
        return f"{self.name}: {self.type_}"


@dataclass(slots=True)
class Imports:
    imports: dict[str, str]

    @override
    def __str__(self) -> str:
        return str(self.imports)


@dataclass(frozen=True, slots=True)
class IgnoreComment:
    kind: str  # e.g., "type", "pyright", "pyrefly", "ty", etc
    rules: frozenset[str] | None

    @override
    def __str__(self) -> str:
        if self.rules is None:
            return f"{self.kind}: ignore"
        return f"{self.kind}: ignore[{', '.join(sorted(self.rules))}]"


@dataclass(slots=True)
class ModuleSymbols:
    imports: Imports
    exports_explicit: frozenset[str] | None  # __all__
    exports_implicit: frozenset[str]  # [from _ ]import $name as $name
    symbols: list[Symbol]
    type_aliases: list[Symbol]
    ignore_comments: list[IgnoreComment]


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
    _overload_map: defaultdict[str, list[Overload]]
    _added_functions: set[str]

    def __init__(self, module: cst.Module, /) -> None:
        self.module = module
        self.symbols = []
        self.type_aliases = []
        self._class_stack = deque()
        self._function_depth = 0
        self._overload_map = defaultdict(list)
        self._added_functions = set()

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

    def _symbol_name(self, name_node: cst.Name) -> str:
        name = self._display_name(name_node)
        if self._class_stack and self._function_depth == 0 and "." not in name:
            name = f"{self._class_stack[-1]}.{name}"
        return name

    @staticmethod
    def _leaf_name(name: str) -> str:
        return name.rsplit(".", 1)[-1]

    @classmethod
    def _is_typealias_annotation(cls, annotation: cst.BaseExpression) -> bool:
        full_name = get_full_name_for_node(annotation)
        return full_name is not None and cls._leaf_name(full_name) == "TypeAlias"

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
            self.type_aliases.append(Symbol(name, Expr(value)))

    @staticmethod
    def _param(
        param: cst.Param,
        kind: ParamKind,
        known_name: str | None = None,
    ) -> Param:
        if (
            known_name is not None
            and param.name.value == known_name
            and not param.annotation
        ):
            annotation = KNOWN
        else:
            annotation = (
                Expr(param.annotation.annotation) if param.annotation else UNKNOWN
            )
        return Param(param.name.value, kind, annotation)

    def _add(self, name_node: cst.Name, annotation: TypeForm) -> None:
        name = self._symbol_name(name_node)
        if _is_public(self._leaf_name(name)):
            self.symbols.append(Symbol(name, annotation))

    @classmethod
    def _callable_signature(
        cls,
        node: cst.FunctionDef,
        known_name: str | None = None,
    ) -> Overload:
        params: list[Param] = []
        for node_params, kind in [
            (node.params.posonly_params, ParamKind.POSITIONAL_ONLY),
            (node.params.params, ParamKind.POSITIONAL_OR_KEYWORD),
            (node.params.kwonly_params, ParamKind.KEYWORD_ONLY),
        ]:
            params.extend(cls._param(param, kind, known_name) for param in node_params)

        star_arg = node.params.star_arg
        if isinstance(star_arg, cst.Param):
            params.append(
                cls._param(star_arg, ParamKind.VAR_POSITIONAL, known_name),
            )

        star_kwarg = node.params.star_kwarg
        if isinstance(star_kwarg, cst.Param):
            params.append(cls._param(star_kwarg, ParamKind.VAR_KEYWORD, known_name))

        return Overload(
            params,
            Expr(node.returns.annotation) if node.returns else UNKNOWN,
        )

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if self._function_depth != 0:
            return False

        class_name = self._class_display_name(node)
        if _is_public(self._leaf_name(class_name)):
            self.symbols.append(Symbol(class_name, Class(class_name)))

        self._class_stack.append(class_name)
        return True

    @override
    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        if self._class_stack:
            self._class_stack.pop()

    @override
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if self._function_depth == 0:
            decorators = {
                self._leaf_name(full)
                for dec in node.decorators
                if (full := get_full_name_for_node(dec.decorator))
            }
            name = self._symbol_name(node.name)

            if self._class_stack and "staticmethod" not in decorators:
                known_name = "cls" if "classmethod" in decorators else "self"
            else:
                known_name = None

            if _is_public(self._leaf_name(name)):
                if "overload" in decorators:
                    self._overload_map[name].append(
                        self._callable_signature(node, known_name),
                    )
                elif "property" in decorators or "cached_property" in decorators:
                    self._add(
                        node.name,
                        Expr(node.returns.annotation) if node.returns else UNKNOWN,
                    )
                else:
                    if overload_list := self._overload_map.pop(name, None):
                        overloads = overload_list[0], *overload_list[1:]
                    else:
                        overloads = (self._callable_signature(node, known_name),)

                    self.symbols.append(Symbol(name, Function(name, overloads)))
                    self._added_functions.add(name)

        self._function_depth += 1
        return True

    @override
    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._function_depth -= 1

    @override
    def leave_Module(self, original_node: cst.Module) -> None:
        for name, overloads in self._overload_map.items():
            if name in self._added_functions or not _is_public(self._leaf_name(name)):
                continue

            self.symbols.append(
                Symbol(name, Function(name, (overloads[0], *overloads[1:]))),
            )

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
            self._add(name_node, Expr(node.annotation.annotation))

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
                self._add(name_node, UNKNOWN)

    @override
    def visit_TypeAlias(self, node: cst.TypeAlias) -> None:
        if self._function_depth == 0:
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


class _TypeIgnoreVisitor(cst.BatchableCSTVisitor):
    _TYPE_IGNORE_RE: Final[re.Pattern[str]] = re.compile(
        r"""
        \s*\#\s*
        ([a-z]+)\s*:\s*
        ignore\b
        (?:\s*\[\s*([^\]]+)\s*\])?
        """,
        re.VERBOSE,
    )

    comments: Final[list[IgnoreComment]]

    def __init__(self) -> None:
        self.comments = []

    @override
    def visit_TrailingWhitespace(self, node: cst.TrailingWhitespace) -> None:
        if node.comment is None:
            return

        comment = node.comment.value
        matches = list(self._TYPE_IGNORE_RE.finditer(comment))
        if not matches:
            return
        for match in matches:
            rules = match.group(2)
            if rules is not None:
                rules = frozenset(
                    rule.strip() for rule in rules.split(",") if rule.strip()
                )
            self.comments.append(
                IgnoreComment(kind=match.group(1), rules=rules),
            )


def collect_global_symbols(source: str, /) -> ModuleSymbols:
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    symbol_visitor = _SymbolVisitor(wrapper.module)
    exports_visitor = _ExportsVisitor(CodemodContext())
    imports_visitor = _ImportsVisitor(CodemodContext())
    type_ignore_visitor = _TypeIgnoreVisitor()
    wrapper.visit_batched(
        [symbol_visitor, exports_visitor, imports_visitor, type_ignore_visitor],
    )

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
    imports = Imports(imports=import_mappings)

    return ModuleSymbols(
        symbols=symbol_visitor.symbols,
        type_aliases=symbol_visitor.type_aliases,
        exports_explicit=exports_visitor.exports_explicit,
        exports_implicit=reexports,
        imports=imports,
        ignore_comments=type_ignore_visitor.comments,
    )


@mainpy.main
def example() -> None:
    src = """
from typing import TypeAlias, TypeVar, overload
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

@overload
def overloaded(x: int) -> int: ...
@overload
def overloaded(x: str) -> str: ...
def overloaded(x):
    return x

class MyClass[T]:  # type: ignore[misc,deprecated]  # ty:ignore[deprecated]
    attr: T

    def method(self, x: int, y) -> str:
        def nested(a: int) -> float:
            return a * 2
        return str(nested(x) + nested(y))

    @classmethod
    def class_method(cls) -> None:
        pass
"""
    module_symbols = collect_global_symbols(src)
    print("Imports:", module_symbols.imports)  # noqa: T201
    print("Exports (explicit):", module_symbols.exports_explicit)  # noqa: T201
    print("Exports (implicit):", module_symbols.exports_implicit)  # noqa: T201

    print()  # noqa: T201
    for alias in module_symbols.type_aliases:
        print(f"{alias.name} = {alias.type_}")  # noqa: T201

    print()  # noqa: T201
    for symbol in module_symbols.symbols:
        print(symbol)  # noqa: T201

    print()  # noqa: T201
    for comment in module_symbols.ignore_comments:
        print(comment)  # noqa: T201
