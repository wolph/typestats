import re
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal, Self, override
from typing import TypeAlias as _TypeAlias

import libcst as cst
import mainpy
from libcst.helpers import (
    get_absolute_module_from_package_for_import,
    get_full_name_for_node,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

__all__ = (
    "ANY",
    "IgnoreComment",
    "ModuleSymbols",
    "Property",
    "Symbol",
    "TypeAlias",
    "TypeForm",
    "annotation_counts",
    "collect_symbols",
    "is_annotated",
)

_EMPTY_MODULE: Final = cst.Module([])
_ENUM_BASES: Final = frozenset({
    "Enum",
    "IntEnum",
    "StrEnum",
    "ReprEnum",
    "Flag",
    "IntFlag",
})
_SCHEMA_BASES: Final = frozenset({"NamedTuple", "TypedDict"})
_DATACLASS_DECORATORS: Final = frozenset({"dataclass"})
_TYPE_CHECK_ONLY: Final = frozenset({"type_check_only"})
_SPECIAL_TYPEFORMS: Final = frozenset({
    "namedtuple",
    "NewType",
    "ParamSpec",
    "TypeAliasType",
    "TypedDict",
    "TypeVar",
    "TypeVarTuple",
})
_ALL: Final = "__all__"

type TypeForm = _TypeMarker | Expr | Function | Property | Class


class ParamKind(StrEnum):
    # matches inspect.Parameter.kind
    POSITIONAL_ONLY = "positional-only"
    POSITIONAL_OR_KEYWORD = "positional or keyword"
    VAR_POSITIONAL = "variadic positional"
    KEYWORD_ONLY = "keyword-only"
    VAR_KEYWORD = "variadic keyword"

    def prefix(self) -> str:
        return {
            ParamKind.VAR_POSITIONAL: "*",
            ParamKind.VAR_KEYWORD: "**",
        }.get(self, "")


class _TypeMarker(StrEnum):
    KNOWN = ""  # for `self` and `cls` parameters
    UNKNOWN = "?"  # for other missing annotations
    ANY = "any"  # for annotations that resolve to `typing.Any`
    EXTERNAL = "~"  # for re-exports from external (non-local) packages

    @override
    def __str__(self) -> str:
        return self.value


type _UnknownType = Literal[_TypeMarker.UNKNOWN]
type _KnownType = Literal[_TypeMarker.KNOWN]
type _AnyType = Literal[_TypeMarker.ANY]
type _ExternalType = Literal[_TypeMarker.EXTERNAL]


UNKNOWN: Final[_UnknownType] = _TypeMarker.UNKNOWN
KNOWN: Final[_KnownType] = _TypeMarker.KNOWN
ANY: Final[_AnyType] = _TypeMarker.ANY
EXTERNAL: Final[_ExternalType] = _TypeMarker.EXTERNAL

type _NameResolver = Callable[[cst.CSTNode], str | None]
type _PropertyAccessor = Literal["setter", "deleter"]
# used in isinstance, so we can't use `type _` syntax
_Sequence: _TypeAlias = cst.List | cst.Tuple  # noqa: UP040
_Container: _TypeAlias = _Sequence | cst.Set  # noqa: UP040


def is_annotated(type_: TypeForm, /) -> bool:
    """Check if a type form represents a meaningfully annotated symbol.

    Returns `True` for `Expr`, `ANY`, and for `Function`/`Property`/`Class`
    types that are annotated (see below).  Returns `False` for `UNKNOWN`,
    `KNOWN`, `EXTERNAL`, and unannotated `Function`/`Property`/`Class` types.
    Note: *self*/*cls* parameters are excluded during parsing.

    For `Function` types, the function is annotated when at least one
    overload has an annotated return type or parameter.

    For `Property` types, the property is annotated when at least one
    accessor (fget, fset, or fdel) has an annotated return type or parameter.

    For `Class` types, the class is only considered annotated when **all**
    of its members (stored in `Class.members`) are also annotated.
    Members marked `KNOWN` (e.g. dataclass fields, enum values) are
    considered annotated.  A class with no members is considered annotated.
    """
    match type_:
        case Expr() | _TypeMarker.ANY:
            return True
        case Function() | Property() | Class():
            return type_.is_annotated
        case _:
            return False


@dataclass(frozen=True, slots=True)
class Expr:
    expr: cst.BaseExpression

    @override
    def __str__(self) -> str:
        return _EMPTY_MODULE.code_for_node(self.expr).strip()

    @classmethod
    def from_annotation(
        cls,
        annotation: cst.Annotation | None,
        name_resolver: _NameResolver | None = None,
    ) -> Self | _UnknownType:
        return (
            cls.from_expr(annotation.annotation, name_resolver)
            if annotation
            else UNKNOWN
        )

    @classmethod
    def from_expr(
        cls,
        expr: cst.BaseExpression,
        name_resolver: _NameResolver | None = None,
    ) -> Self:
        return cls(_unwrap_annotated(_parse_string_annotation(expr), name_resolver))


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    kind: ParamKind
    annotation: TypeForm

    @property
    def is_annotated(self) -> bool:
        return is_annotated(self.annotation)

    @override
    def __str__(self) -> str:
        return f"{self.kind.prefix()}{self.name}: {self.annotation}"


@dataclass(frozen=True, slots=True)
class Overload:
    params: tuple[Param, ...]
    returns: TypeForm

    @property
    def is_annotated(self) -> bool:
        return is_annotated(self.returns) or any(p.is_annotated for p in self.params)

    @property
    def annotation_counts(self) -> tuple[int, int]:
        """`(annotated, annotatable)` counts for params + return."""
        annotated = sum(1 for p in self.params if p.is_annotated)
        if is_annotated(self.returns):
            annotated += 1
        return annotated, len(self.params) + 1  # params + return

    @override
    def __str__(self) -> str:
        params = ", ".join(str(param) for param in self.params)
        return f"({params}) -> {self.returns}"


def _nonempty_tuple(items: list[Overload], /) -> tuple[Overload, *tuple[Overload, ...]]:
    """Convert a non-empty list of overloads to a non-empty tuple."""
    return items[0], *items[1:]


@dataclass(frozen=True, slots=True)
class Function:
    name: str
    overloads: tuple[Overload, *tuple[Overload, ...]]

    def __post_init__(self) -> None:
        if not self.overloads:
            msg = "FunctionOverloads must have at least one signature"
            raise ValueError(msg)

    @property
    def is_annotated(self) -> bool:
        return any(o.is_annotated for o in self.overloads)

    @override
    def __str__(self) -> str:
        if len(self.overloads) == 1:
            return str(self.overloads[0])
        # an overloaded function type is the intersection of its overloads
        return " & ".join(f"({sig})" for sig in self.overloads)

    @property
    def annotation_counts(self) -> tuple[int, int]:
        """`(annotated, annotatable)` counts across all overloads."""
        counts = [o.annotation_counts for o in self.overloads]
        return sum(a for a, _ in counts), sum(t for _, t in counts)


@dataclass(frozen=True, slots=True)
class Property:
    name: str
    fget: Overload | None = None
    fset: Overload | None = None
    fdel: Overload | None = None

    @property
    def is_annotated(self) -> bool:
        return any(
            accessor is not None and accessor.is_annotated
            for accessor in (self.fget, self.fset, self.fdel)
        )

    @property
    def annotation_counts(self) -> tuple[int, int]:
        """`(annotated, annotatable)` counts across all accessors."""
        counts = [
            accessor.annotation_counts
            for accessor in (self.fget, self.fset, self.fdel)
            if accessor is not None
        ]
        return sum(a for a, _ in counts), sum(t for _, t in counts)

    @override
    def __str__(self) -> str:
        parts: list[str] = []
        if self.fget is not None:
            parts.append(f"fget={self.fget}")
        if self.fset is not None:
            parts.append(f"fset={self.fset}")
        if self.fdel is not None:
            parts.append(f"fdel={self.fdel}")
        return f"property({', '.join(parts)})"


@dataclass(frozen=True, slots=True)
class Class:
    name: str
    members: tuple[TypeForm, ...] = ()

    @property
    def is_annotated(self) -> bool:
        return all(m is KNOWN or is_annotated(m) for m in self.members)

    @property
    def annotation_counts(self) -> tuple[int, int]:
        """`(annotated, annotatable)` counts across all members."""
        counts = [annotation_counts(m) for m in self.members]
        return sum(a for a, _ in counts), sum(t for _, t in counts)

    @override
    def __str__(self) -> str:
        return f"type[{self.name}]"


def annotation_counts(type_: TypeForm, /) -> tuple[int, int]:
    """`(annotated, annotatable)` counts for an arbitrary type form."""
    match type_:
        case Function() | Property() | Class():
            return type_.annotation_counts
        case Expr():
            return 1, 1
        case _TypeMarker.ANY:
            return 1, 1
        case _TypeMarker.UNKNOWN:
            return 0, 1
        case _:
            return 0, 0


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str
    type_: TypeForm

    @override
    def __str__(self) -> str:
        return f"{self.name}: {self.type_}"


@dataclass(frozen=True, slots=True)
class TypeAlias:
    name: str
    value: TypeForm

    @override
    def __str__(self) -> str:
        return f"type {self.name} = {self.value}"


@dataclass(frozen=True, slots=True)
class IgnoreComment:
    kind: str  # e.g., "type", "pyright", "pyrefly", "ty", etc
    rules: frozenset[str] | None

    @override
    def __str__(self) -> str:
        if self.rules is None:
            return f"{self.kind}: ignore"
        return f"{self.kind}: ignore[{', '.join(self.rules)}]"


@dataclass(frozen=True, slots=True)
class ModuleSymbols:
    imports: tuple[tuple[str, str], ...]
    imports_wildcard: tuple[str, ...]  # modules from `from _ import *`
    exports_explicit: frozenset[str] | None  # __all__
    exports_explicit_dynamic: tuple[str, ...]  # __all__ += mod.__all__
    exports_implicit: frozenset[str]  # [from _ ]import $name as $name
    symbols: tuple[Symbol, ...]
    type_aliases: tuple[TypeAlias, ...]
    ignore_comments: tuple[IgnoreComment, ...]
    type_check_only: frozenset[str]  # @type_check_only decorated names


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


def _parse_string_annotation(expr: cst.BaseExpression) -> cst.BaseExpression:
    """Parse a stringified annotation like `"list[str]"` into a CST expression.

    If *expr* is a `SimpleString` or `ConcatenatedString` whose evaluated value
    is valid Python, the parsed expression is returned.  Otherwise the original
    *expr* is returned unchanged.
    """
    if not isinstance(expr, cst.SimpleString | cst.ConcatenatedString):
        return expr
    value = expr.evaluated_value
    if value is None or not isinstance(value, str):
        return expr
    try:
        return cst.parse_expression(value)
    except cst.ParserSyntaxError:
        return expr


def _is_dunder_slots(expr: cst.BaseExpression) -> bool:
    return isinstance(expr, cst.Name) and expr.value == "__slots__"


def _leaf_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _unwrap_annotated(
    expr: cst.BaseExpression,
    name_resolver: _NameResolver | None = None,
) -> cst.BaseExpression:
    current = expr
    while isinstance(current, cst.Subscript):
        value = current.value
        full_name = name_resolver(value) if name_resolver else None
        if full_name is None:
            full_name = get_full_name_for_node(value)
        if full_name is None or _leaf_name(full_name) != "Annotated":
            break
        if not current.slice:
            break

        first = current.slice[0].slice
        if not isinstance(first, cst.Index):
            break

        current = first.value
    return current


def _is_all_target(target: cst.BaseExpression) -> bool:
    return get_full_name_for_node(target) == _ALL


@dataclass(slots=True)
class _ClassStackItem:
    name: str
    is_enum: bool
    is_schema: bool
    symbol_index: int  # index into _SymbolVisitor.symbols where the Class symbol lives
    members: list[TypeForm]


class _SymbolVisitor(cst.CSTVisitor):  # noqa: PLR0904
    _TYPE_IGNORE_RE: Final[re.Pattern[str]] = re.compile(
        r"""
        \s*\#\s*
        ([a-z]+)\s*:\s*
        ignore\b
        (?:\s*\[\s*([^\]]+)\s*\])?
        """,
        re.VERBOSE,
    )

    # --- Results ---
    symbols: Final[list[Symbol]]
    type_aliases: Final[list[TypeAlias]]
    type_check_only_names: Final[set[str]]
    ignore_comments: Final[list[IgnoreComment]]

    # --- Imports state ---
    module_aliases: Final[dict[str, str]]
    from_imports: Final[defaultdict[str, set[str]]]
    alias_mapping: Final[dict[str, list[tuple[str, str]]]]

    # --- Exports state ---
    has_explicit_all: bool
    all_sources: Final[list[str]]
    _exported_objects: Final[set[str]]
    _is_assigned_export: Final[set[_Container]]
    _in_assigned_export: Final[set[_Container]]

    # --- Symbol state ---
    imports: Final[dict[str, str]]
    _defined_names: Final[set[str]]
    _class_stack: Final[deque[_ClassStackItem]]
    _function_depth: int
    _skipped_class_depth: int
    _overload_map: defaultdict[str, list[Overload]]
    _property_map: dict[str, int]
    _added_functions: set[str]

    _package_name: Final[str]

    def __init__(self, /, *, package_name: str = "") -> None:
        self.symbols = []
        self.type_aliases = []
        self.type_check_only_names = set()
        self.ignore_comments = []

        self.module_aliases = {}
        self.from_imports = defaultdict(set)
        self.alias_mapping = {}

        self.has_explicit_all = False
        self.all_sources = []
        self._exported_objects = set()
        self._is_assigned_export = set()
        self._in_assigned_export = set()

        self.imports = {}
        self._defined_names = set()
        self._class_stack = deque()
        self._function_depth = 0
        self._skipped_class_depth = 0
        self._overload_map = defaultdict(list)
        self._property_map = {}
        self._added_functions = set()

        self._package_name = package_name

    @property
    def exports_explicit(self) -> frozenset[str] | None:
        return frozenset(self._exported_objects) if self.has_explicit_all else None

    def _resolve_name(self, node: cst.CSTNode) -> str | None:
        """Resolve a CST node to its fully qualified name using the import map."""
        if (raw := get_full_name_for_node(node)) is None:
            return None

        first, _, rest = raw.partition(".")
        if fqn := self.imports.get(first):
            return f"{fqn}.{rest}" if rest else fqn
        return raw

    def _symbol_name(self, node: cst.Name) -> str:
        name = node.value
        if (stack := self._class_stack) and not self._function_depth:
            name = f"{stack[-1].name}.{name}"
        return name

    def _is_name_in(self, node: cst.CSTNode, haystack: Collection[str]) -> bool:
        full_name = self._resolve_name(node)
        return full_name is not None and _leaf_name(full_name) in haystack

    def _is_schema_class(self, node: cst.ClassDef) -> bool:
        for dec in node.decorators:
            expr = dec.decorator
            if isinstance(expr, cst.Call):
                expr = expr.func

            if self._is_name_in(expr, _DATACLASS_DECORATORS):
                return True

        return any(self._is_name_in(b.value, _SCHEMA_BASES) for b in node.bases)

    def _is_typealias_annotation(self, annotation: cst.Annotation) -> bool:
        full_name = self._resolve_name(annotation.annotation)
        return full_name is not None and _leaf_name(full_name) == "TypeAlias"

    def _is_special_typeform(self, expr: cst.BaseExpression) -> bool:
        return (
            isinstance(expr, cst.Call)
            and self._is_name_in(expr.func, _SPECIAL_TYPEFORMS)
        )  # fmt: skip

    def _typealias_value_from_call(
        self,
        expr: cst.BaseExpression,
    ) -> cst.BaseExpression | None:
        if not isinstance(expr, cst.Call):
            return None

        full_name = self._resolve_name(expr.func)
        if full_name is None or _leaf_name(full_name) != "TypeAliasType":
            return None

        positional_args = [arg for arg in expr.args if arg.keyword is None]
        if len(positional_args) > 1:
            return positional_args[1].value

        for arg in expr.args:
            if arg.keyword and arg.keyword.value == "value":
                return arg.value

        return None

    def _add_type_aliases(
        self,
        name_nodes: list[cst.Name],
        value: cst.BaseExpression,
    ) -> None:
        if not name_nodes:
            return

        type_aliases = self.type_aliases
        symbol_name = self._symbol_name
        expr = Expr.from_expr(value, self._resolve_name)

        if not self._class_stack:
            self._defined_names.update(n.value for n in name_nodes)

        for name_node in name_nodes:
            type_aliases.append(TypeAlias(symbol_name(name_node), expr))

    def _add_symbols(
        self,
        name_nodes: list[cst.Name],
        annotation: TypeForm,
    ) -> None:
        if not name_nodes:
            return
        symbols = self.symbols
        symbol_name = self._symbol_name
        if not self._class_stack:
            self._defined_names.update(n.value for n in name_nodes)
            for name_node in name_nodes:
                symbols.append(Symbol(symbol_name(name_node), annotation))
        elif not self._function_depth:
            members = self._class_stack[-1].members
            for name_node in name_nodes:
                symbols.append(Symbol(symbol_name(name_node), annotation))
                members.append(annotation)
        else:
            for name_node in name_nodes:
                symbols.append(Symbol(symbol_name(name_node), annotation))

    def _callable_signature(
        self,
        node: cst.FunctionDef,
        *,
        skip_first: bool = False,
    ) -> Overload:
        params: list[Param] = []
        skipped = False
        for node_params, kind in [
            (node.params.posonly_params, ParamKind.POSITIONAL_ONLY),
            (node.params.params, ParamKind.POSITIONAL_OR_KEYWORD),
            (node.params.kwonly_params, ParamKind.KEYWORD_ONLY),
            ((node.params.star_arg,), ParamKind.VAR_POSITIONAL),
            ((node.params.star_kwarg,), ParamKind.VAR_KEYWORD),
        ]:
            for param in node_params:
                if not isinstance(param, cst.Param):
                    continue
                if (
                    skip_first
                    and not skipped
                    and (
                        kind
                        in {ParamKind.POSITIONAL_ONLY, ParamKind.POSITIONAL_OR_KEYWORD}
                    )
                ):
                    skipped = True
                    continue
                params.append(
                    Param(
                        param.name.value,
                        kind,
                        Expr.from_annotation(param.annotation, self._resolve_name),
                    ),
                )

        return Overload(
            tuple(params),
            Expr.from_annotation(node.returns, self._resolve_name),
        )

    def _has_type_check_only(self, node: cst.ClassDef | cst.FunctionDef) -> bool:
        for dec in node.decorators:
            expr = dec.decorator
            if isinstance(expr, cst.Call):
                expr = expr.func
            if self._is_name_in(expr, _TYPE_CHECK_ONLY):
                return True
        return False

    def _property_accessor(
        self,
        node: cst.FunctionDef,
    ) -> tuple[_PropertyAccessor, str] | None:
        """Return the `@name.setter` or `@name.deleter` for a known property.

        Returns `(accessor_kind, property_full_name)` or `None`.
        """
        for dec in node.decorators:
            if not isinstance(expr := dec.decorator, cst.Attribute):
                continue

            if (attr := expr.attr.value) not in {"setter", "deleter"}:
                continue
            if not isinstance(expr.value, cst.Name):
                continue
            prop_base_name = expr.value.value
            full_name = (
                f"{self._class_stack[-1].name}.{prop_base_name}"
                if self._class_stack
                else prop_base_name
            )
            if full_name in self._property_map:
                return attr, full_name

        return None

    def _add_property(self, node: cst.FunctionDef, name: str, sig: Overload) -> None:
        """Create a new `Property` with *sig* as its `fget`."""
        prop = Property(name, fget=sig)
        self._property_map[name] = len(self.symbols)
        if not self._class_stack:
            self._defined_names.add(node.name.value)
        self.symbols.append(Symbol(name, prop))
        if self._class_stack:
            self._class_stack[-1].members.append(prop)

    def _update_property(
        self,
        kind: _PropertyAccessor,
        prop_name: str,
        sig: Overload,
    ) -> None:
        """Attach *sig* as the setter or deleter of an existing `Property`."""
        idx = self._property_map[prop_name]
        old_prop = self.symbols[idx].type_
        assert isinstance(old_prop, Property)

        if kind == "setter":
            fset, fdel = sig, old_prop.fdel
        else:
            fset, fdel = old_prop.fset, sig
        new_prop = Property(old_prop.name, old_prop.fget, fset, fdel)
        self.symbols[idx] = Symbol(old_prop.name, new_prop)

        if self._class_stack:
            for i, m in enumerate(members := self._class_stack[-1].members):
                if m is old_prop:
                    members[i] = new_prop
                    break

    # --- Import handling ---

    def _collect_import_from_names(
        self,
        module: str,
        nodenames: Collection[cst.ImportAlias],
    ) -> None:
        for ia in nodenames:
            alias = ia.evaluated_alias
            name = ia.evaluated_name
            if alias is not None:
                self.alias_mapping.setdefault(module, []).append((name, alias))
                self.imports[alias] = f"{module}.{name}"
            elif name != "*":
                objects = self.from_imports[module]
                if "*" not in objects:
                    objects.add(name)
                self.imports[name] = f"{module}.{name}"

    @override
    def visit_Import(self, node: cst.Import) -> bool:
        if not isinstance(node.names, cst.ImportStar):
            for name in node.names:
                evaluated_name = name.evaluated_name
                if alias := name.evaluated_alias:
                    self.module_aliases[evaluated_name] = alias
                    self.imports[alias] = evaluated_name
                else:
                    self.imports[evaluated_name] = evaluated_name

        return False

    @override
    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        module = get_absolute_module_from_package_for_import(self._package_name, node)
        if module:
            nodenames = node.names
            if isinstance(nodenames, cst.ImportStar):
                self.from_imports[module] = {"*"}
            else:
                self._collect_import_from_names(module, nodenames)

        return False

    # --- Export handling ---

    def _handle_assign_target_exports(
        self,
        target: cst.BaseExpression,
        value: cst.BaseExpression,
    ) -> bool:
        # Find the value assigned to __all__, whether direct or via tuple unpacking
        all_value: cst.BaseExpression | None = None
        if _is_all_target(target):
            all_value = value
        elif isinstance(target, cst.Tuple) and isinstance(value, cst.Tuple):
            for idx, element_node in enumerate(target.elements):
                if _is_all_target(element_node.value):
                    all_value = value.elements[idx].value
                    break

        if isinstance(all_value, _Container):
            self._is_assigned_export.add(all_value)
            return True

        return False

    def _visit_container(self, node: _Container) -> Literal[True]:
        if node in self._is_assigned_export:
            self._in_assigned_export.add(node)
        return True

    @override
    def visit_List(self, node: cst.List) -> bool:
        return self._visit_container(node)

    @override
    def visit_Tuple(self, node: cst.Tuple) -> bool:
        return self._visit_container(node)

    @override
    def visit_Set(self, node: cst.Set) -> bool:
        return self._visit_container(node)

    def _leave_container(self, node: _Container) -> None:
        self._is_assigned_export.discard(node)
        self._in_assigned_export.discard(node)

    @override
    def leave_List(self, original_node: cst.List) -> None:
        self._leave_container(original_node)

    @override
    def leave_Tuple(self, original_node: cst.Tuple) -> None:
        self._leave_container(original_node)

    @override
    def leave_Set(self, original_node: cst.Set) -> None:
        self._leave_container(original_node)

    def _visit_string(
        self,
        node: cst.SimpleString | cst.ConcatenatedString,
    ) -> Literal[False]:
        if self._in_assigned_export and isinstance(name := node.evaluated_value, str):
            self._exported_objects.add(name)
        return False

    @override
    def visit_SimpleString(self, node: cst.SimpleString) -> bool:
        return self._visit_string(node)

    @override
    def visit_ConcatenatedString(self, node: cst.ConcatenatedString) -> bool:
        return self._visit_string(node)

    # --- Type-ignore comment handling ---

    @override
    def visit_TrailingWhitespace(self, node: cst.TrailingWhitespace) -> bool:
        if node.comment is not None:
            for match in self._TYPE_IGNORE_RE.finditer(node.comment.value):
                rules = match.group(2)
                if rules is not None:
                    rules = frozenset(r.strip() for r in rules.split(",") if r.strip())
                self.ignore_comments.append(IgnoreComment(match.group(1), rules))
        return False

    # --- Symbol handling ---

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if self._function_depth:
            self._skipped_class_depth += 1
            return True  # still visit children for type-ignore comments

        name = node.name.value
        if not self._class_stack:
            self._defined_names.add(name)
            if self._has_type_check_only(node):
                self.type_check_only_names.add(name)
        symbol_index = len(self.symbols)
        self.symbols.append(Symbol(name, Class(name)))
        self._class_stack.append(
            _ClassStackItem(
                name=name,
                is_enum=any(self._is_name_in(b.value, _ENUM_BASES) for b in node.bases),
                is_schema=self._is_schema_class(node),
                symbol_index=symbol_index,
                members=[],
            ),
        )

        return True

    @override
    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        if self._skipped_class_depth:
            self._skipped_class_depth -= 1
            return
        if stack := self._class_stack:
            item = stack.pop()
            self.symbols[item.symbol_index] = Symbol(
                item.name,
                Class(item.name, tuple(item.members)),
            )

    @override
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if self._function_depth == 0:
            self._handle_function_def(node)

        self._function_depth += 1
        return True

    def _handle_function_def(self, node: cst.FunctionDef) -> None:
        if not self._class_stack:
            self._defined_names.add(node.name.value)
            if self._has_type_check_only(node):
                self.type_check_only_names.add(node.name.value)

        decorators = {
            _leaf_name(full)
            for dec in node.decorators
            if (full := get_full_name_for_node(dec.decorator))
        }
        name = self._symbol_name(node.name)
        skip_first = bool(self._class_stack) and "staticmethod" not in decorators

        if "overload" in decorators:
            self._overload_map[name].append(
                self._callable_signature(node, skip_first=skip_first),
            )
        elif "property" in decorators or "cached_property" in decorators:
            sig = self._callable_signature(node, skip_first=skip_first)
            self._add_property(node, name, sig)
        elif (accessor := self._property_accessor(node)) is not None:
            accessor_kind, prop_name = accessor
            sig = self._callable_signature(node, skip_first=skip_first)
            self._update_property(accessor_kind, prop_name, sig)
        else:
            if overload_list := self._overload_map.pop(name, None):
                overloads = _nonempty_tuple(overload_list)
            else:
                overloads = (self._callable_signature(node, skip_first=skip_first),)

            func = Function(name, overloads)
            self.symbols.append(Symbol(name, func))
            self._added_functions.add(name)
            if self._class_stack:
                self._class_stack[-1].members.append(func)

    @override
    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._function_depth -= 1

    @override
    def leave_Module(self, original_node: cst.Module) -> None:
        added_functions = self._added_functions
        for name, overloads in self._overload_map.items():
            if name not in added_functions:
                self.symbols.append(
                    Symbol(name, Function(name, _nonempty_tuple(overloads))),
                )

    @override
    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        # Exports: detect `__all__: ... = [...]`
        if _is_all_target(node.target):
            self.has_explicit_all = True
        if node.value is not None:
            self._handle_assign_target_exports(node.target, node.value)

        # Symbols
        if self._function_depth == 0:
            self._handle_ann_assign_symbols(node)

    def _handle_ann_assign_symbols(self, node: cst.AnnAssign) -> None:
        # __slots__ is a runtime implementation detail, not a type annotation
        if self._class_stack and _is_dunder_slots(node.target):
            return

        if node.value is not None:
            if self._is_typealias_annotation(node.annotation):
                self._add_type_aliases(_extract_names(node.target), node.value)
                return

            if self._is_special_typeform(node.value):
                return

        if self._class_stack and self._class_stack[-1].is_schema:
            ty = KNOWN
        else:
            ty = Expr.from_expr(node.annotation.annotation, self._resolve_name)

        self._add_symbols(_extract_names(node.target), ty)

    def _try_resolve_method_alias(self, node: cst.Assign) -> bool:
        if not self._class_stack or not isinstance(node.value, cst.Name):
            return False

        current_class = self._class_stack[-1]
        methods = current_class.members
        symbols = self.symbols

        ref = f"{current_class.name}.{node.value.value}"

        if overloads := self._overload_map.get(ref):
            ref_func = Function(ref, _nonempty_tuple(overloads))
        else:
            src_type = next((s.type_ for s in symbols if s.name == ref), None)
            if not isinstance(src_type, Function):
                return False
            ref_func = src_type

        for target in node.targets:
            for name_node in _extract_names(target.target):
                alias_name = self._symbol_name(name_node)
                func = Function(alias_name, ref_func.overloads)
                methods.append(func)
                symbols.append(Symbol(alias_name, func))

        return True

    def _try_add_name_alias(self, node: cst.Assign) -> bool:
        """Handle `X = {name}` or `X = {name}[...]` as an import alias or type alias."""
        if self._class_stack:
            return False

        # Unwrap subscript: `X = SomeType[args]` → resolve `SomeType`
        value = node.value
        is_subscript = isinstance(value, cst.Subscript)
        base = value.value if is_subscript else value

        if not isinstance(base, cst.Name | cst.Attribute):
            return False

        if not (raw := get_full_name_for_node(base)):
            return False

        first = raw.split(".", 1)[0]

        if first in self.imports:
            resolved = self.imports[first]
            _, _, rest = raw.partition(".")
            fqn = f"{resolved}.{rest}" if rest else resolved
            for target in node.targets:
                names = _extract_names(target.target)
                if is_subscript:
                    # `X = ImportedType[args]` is a type alias, not a re-export
                    self._add_type_aliases(names, value)
                else:
                    for n in names:
                        self.imports[n.value] = fqn
            return True

        if first in self._defined_names:
            for target in node.targets:
                self._add_type_aliases(_extract_names(target.target), value)
            return True

        return False

    @override
    def visit_AugAssign(self, node: cst.AugAssign) -> None:
        # Exports: detect `__all__ += [...]` and `__all__ += mod.__all__`
        if not _is_all_target(node.target):
            return

        self.has_explicit_all = True
        if (
            isinstance(value := node.value, cst.Attribute)
            and value.attr.value == _ALL
            and (source_name := get_full_name_for_node(value.value))
        ):
            self.all_sources.append(source_name)

        if isinstance(node.operator, cst.AddAssign) and isinstance(value, _Sequence):
            self._is_assigned_export.add(value)

    @override
    def visit_Assign(self, node: cst.Assign) -> None:
        value = node.value
        targets = [target.target for target in node.targets]

        # Exports: detect `__all__ = [...]`

        if not self.has_explicit_all and any(map(_is_all_target, targets)):
            self.has_explicit_all = True

        for target in targets:
            self._handle_assign_target_exports(target, value)

        if self._function_depth:
            return

        # __slots__ is a runtime implementation detail, not a type annotation
        if (stack := self._class_stack) and all(map(_is_dunder_slots, targets)):
            return

        if typealias_value := self._typealias_value_from_call(value):
            for target in targets:
                self._add_type_aliases(_extract_names(target), typealias_value)
            return

        if self._is_special_typeform(value):
            return

        if self._try_add_name_alias(node) or self._try_resolve_method_alias(node):
            return

        # enum attributes are considered KNOWN
        ty = KNOWN if stack and stack[-1].is_enum else UNKNOWN
        for target in targets:
            self._add_symbols(_extract_names(target), ty)

    @override
    def visit_TypeAlias(self, node: cst.TypeAlias) -> None:
        if not self._function_depth:
            self._add_type_aliases([node.name], node.value)


_EMPTY_SYMBOLS: Final = ModuleSymbols(
    imports=(),
    imports_wildcard=(),
    exports_explicit=None,
    exports_explicit_dynamic=(),
    exports_implicit=frozenset(),
    symbols=(),
    type_aliases=(),
    ignore_comments=(),
    type_check_only=frozenset(),
)


def collect_symbols(
    source: str,
    /,
    *,
    package_name: str | None = None,
) -> ModuleSymbols:
    if not source or source.isspace():
        return _EMPTY_SYMBOLS

    module = cst.parse_module(source)
    visitor = _SymbolVisitor(package_name=package_name or "")
    module.visit(visitor)  # type: ignore[arg-type]  # CSTVisitor is accepted at runtime

    imports = visitor.imports

    wildcard_modules = tuple(
        mod for mod, objects in visitor.from_imports.items() if "*" in objects
    )

    reexports = frozenset(
        name
        for aliases in visitor.alias_mapping.values()
        for name, alias in aliases
        if name == alias
    ) | frozenset(mod for mod, alias in visitor.module_aliases.items() if mod == alias)

    return ModuleSymbols(
        symbols=tuple(visitor.symbols),
        type_aliases=tuple(visitor.type_aliases),
        imports=tuple(imports.items()),
        imports_wildcard=wildcard_modules,
        exports_explicit=visitor.exports_explicit,
        exports_explicit_dynamic=tuple(visitor.all_sources),
        exports_implicit=reexports,
        ignore_comments=tuple(visitor.ignore_comments),
        type_check_only=frozenset(visitor.type_check_only_names),
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
    module_symbols = collect_symbols(src)
    print("Imports:", module_symbols.imports)  # noqa: T201
    print("Exports (explicit):", module_symbols.exports_explicit)  # noqa: T201
    print("Exports (implicit):", module_symbols.exports_implicit)  # noqa: T201

    print()  # noqa: T201
    for alias in module_symbols.type_aliases:
        print(alias)  # noqa: T201

    print()  # noqa: T201
    for symbol in module_symbols.symbols:
        print(symbol)  # noqa: T201

    print()  # noqa: T201
    for comment in module_symbols.ignore_comments:
        print(comment)  # noqa: T201
