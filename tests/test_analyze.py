import textwrap

import libcst as cst

from typestats.analyze import (
    EXTERNAL,
    KNOWN,
    UNKNOWN,
    Class,
    Expr,
    Function,
    Overload,
    Param,
    ParamKind,
    collect_symbols,
    is_annotated,
)


class TestImports:
    def test_basic(self) -> None:
        src = textwrap.dedent("""
        import a
        import a as _a
        import a.b
        import a.c as _c
        from b import d
        from b import e as _e
        """)
        imports = dict(collect_symbols(src).imports)
        assert imports["a"] == "a"
        assert imports["_a"] == "a"
        assert imports["a.b"] == "a.b"
        assert imports["_c"] == "a.c"
        assert imports["d"] == "b.d"
        assert imports["_e"] == "b.e"

    def test_assign_imported_name_is_import_alias(self) -> None:
        """X = Never (imported) should become an import alias, not an UNKNOWN symbol."""
        src = textwrap.dedent("""
        from typing import Never

        Complex64 = Never
        """)
        module = collect_symbols(src)
        imports = dict(module.imports)
        assert imports["Complex64"] == "typing.Never"
        assert len(module.symbols) == 0


class TestExports:
    def test_implicit_direct(self) -> None:
        src = textwrap.dedent("""
        import a
        import b as _b
        import c as c
        """)
        exports = collect_symbols(src).exports_implicit
        assert exports == {"c"}

    def test_implicit_from(self) -> None:
        src = textwrap.dedent("""
        from m import a
        from m import b as _b
        from m import c as c
        """)
        exports = collect_symbols(src).exports_implicit
        assert exports == {"c"}

    def test_explicit(self) -> None:
        src = """__all__ = ["a", "b", "c"]"""
        exports = collect_symbols(src).exports_explicit
        assert exports == {"a", "b", "c"}

    def test_explicit_missing(self) -> None:
        src = """a = 1"""
        exports = collect_symbols(src).exports_explicit
        assert exports is None


class TestTypeAliases:
    def test_basic(self) -> None:
        src = textwrap.dedent("""
        from typing import TypeAlias, TypeAliasType

        A: TypeAlias = str
        B = TypeAliasType("B", str)
        type C = str
        D = str

        class E: ...
        def f() -> None: ...
        """)
        type_aliases = collect_symbols(src).type_aliases
        assert type_aliases[0].name == "A"
        assert type_aliases[1].name == "B"
        assert type_aliases[2].name == "C"

    def test_indirect(self) -> None:
        src = textwrap.dedent("""
        import typing as t
        from typing import TypeAlias as Alias
        from typing_extensions import TypeAliasType as AliasType

        A1: t.TypeAlias = str
        A2: Alias = str

        B1 = t.TypeAliasType("B1", str)
        B2 = AliasType("B2", str)
        """)
        type_aliases = collect_symbols(src).type_aliases
        assert type_aliases[0].name == "A1"
        assert type_aliases[1].name == "A2"
        assert type_aliases[2].name == "B1"
        assert type_aliases[3].name == "B2"

    def test_assign_local_name(self) -> None:
        """X = Y (locally defined) should become a type alias, not an UNKNOWN symbol."""
        src = textwrap.dedent("""
        from typing import TypeAlias

        AnyInt8Array: TypeAlias = int
        AnyByteArray = AnyInt8Array
        """)
        module = collect_symbols(src)
        aliases = {a.name: str(a.value) for a in module.type_aliases}
        assert "AnyByteArray" in aliases
        assert aliases["AnyByteArray"] == "AnyInt8Array"
        assert all(s.name != "AnyByteArray" for s in module.symbols)

    def test_assign_subscript_imported(self) -> None:
        """X = ImportedType[args] should become a type alias, not UNKNOWN."""
        src = textwrap.dedent("""
        from numpy import signedinteger
        from numpy._typing import _8Bit, _16Bit

        int8 = signedinteger[_8Bit]
        int16 = signedinteger[_16Bit]
        """)
        module = collect_symbols(src)
        aliases = {a.name: str(a.value) for a in module.type_aliases}
        assert "int8" in aliases
        assert aliases["int8"] == "signedinteger[_8Bit]"
        assert "int16" in aliases
        assert aliases["int16"] == "signedinteger[_16Bit]"
        assert all(s.name not in {"int8", "int16"} for s in module.symbols)

    def test_assign_subscript_local(self) -> None:
        """X = LocalType[args] should become a type alias, not UNKNOWN."""
        src = textwrap.dedent("""
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class MyGeneric(Generic[T]): ...

        Alias = MyGeneric[int]
        """)
        module = collect_symbols(src)
        aliases = {a.name: str(a.value) for a in module.type_aliases}
        assert "Alias" in aliases
        assert aliases["Alias"] == "MyGeneric[int]"
        assert all(s.name != "Alias" for s in module.symbols)


class TestSymbols:
    def test_basic(self) -> None:
        src = textwrap.dedent("""
        import a

        x: int = 1

        class A:
            pass

        def f() -> None:
            pass
        """)
        symbols = collect_symbols(src).symbols
        assert symbols[0].name == "x"
        assert symbols[1].name == "A"
        assert symbols[2].name == "f"
        assert len(symbols) == 3

    def test_no_type_alias(self) -> None:
        src = textwrap.dedent("""
        from typing import TypeAlias, TypeAliasType

        A: TypeAlias = str
        B = TypeAliasType("B", str)
        type C = str
        D = str
        """)
        symbols = collect_symbols(src).symbols
        assert symbols[0].name == "D"
        assert len(symbols) == 1

    def test_special_typeforms_ignored_aliases(self) -> None:
        src = textwrap.dedent("""
        import typing as t
        from typing import NewType as NT

        UserId = t.NewType("UserId", int)
        Token = NT("Token", str)
        D = 1
        """)
        symbols = collect_symbols(src).symbols
        assert symbols[0].name == "D"
        assert len(symbols) == 1

    def test_special_typeforms_ignored_annassign(self) -> None:
        src = textwrap.dedent("""
        import typing as t

        T: object = t.TypeVar("T")
        D: int = 1
        """)
        symbols = collect_symbols(src).symbols
        assert symbols[0].name == "D"
        assert len(symbols) == 1


class TestIgnoreComments:
    def test_basic(self) -> None:
        src = textwrap.dedent("""
        x: int = 1  # type: ignore[misc,deprecated]  # ty:ignore[deprecated]
        y: str = "hello"  # pyrefly : ignore
        """)
        ignore_comments = collect_symbols(src).ignore_comments

        assert ignore_comments[0].kind == "type"
        assert ignore_comments[0].rules == {"misc", "deprecated"}

        assert ignore_comments[1].kind == "ty"
        assert ignore_comments[1].rules == {"deprecated"}

        assert ignore_comments[2].kind == "pyrefly"
        assert ignore_comments[2].rules is None


class TestAnnotatedUnwrap:
    def test_basic(self) -> None:
        src = textwrap.dedent("""
        from typing import Annotated, TypeAlias

        X: Annotated[int, "meta"] = 1
        A: TypeAlias = Annotated[str, "alias-meta"]
        """)
        module = collect_symbols(src)

        assert module.symbols[0].name == "X"
        assert str(module.symbols[0].type_) == "int"

        assert module.type_aliases[0].name == "A"
        assert str(module.type_aliases[0].value) == "str"

    def test_indirect_import(self) -> None:
        src = textwrap.dedent("""
        import typing as t

        X: t.Annotated[int, "meta"] = 1
        A: t.TypeAlias = t.Annotated[str, "alias-meta"]
        """)
        module = collect_symbols(src)

        assert module.symbols[0].name == "X"
        assert str(module.symbols[0].type_) == "int"

        assert module.type_aliases[0].name == "A"
        assert str(module.type_aliases[0].value) == "str"


class TestKnownAttrs:
    def test_enum_members(self) -> None:
        src = textwrap.dedent("""
        from enum import Enum

        class Color(Enum):
            RED = 1
            BLUE = 2
        """)
        module = collect_symbols(src)
        symbols = {symbol.name: symbol.type_ for symbol in module.symbols}

        assert symbols["Color.RED"] is KNOWN
        assert symbols["Color.BLUE"] is KNOWN

    def test_enum_members_with_alias(self) -> None:
        src = textwrap.dedent("""
        from enum import Enum as MyEnum

        class Status(MyEnum):
            READY = 1
        """)
        module = collect_symbols(src)
        symbols = {symbol.name: symbol.type_ for symbol in module.symbols}

        assert symbols["Status.READY"] is KNOWN

    def test_dataclass_attrs(self) -> None:
        src = textwrap.dedent("""
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: float
        """)
        module = collect_symbols(src)
        symbols = {s.name: s.type_ for s in module.symbols}

        assert symbols["Point.x"] is KNOWN
        assert symbols["Point.y"] is KNOWN

    def test_dataclass_call_attrs(self) -> None:
        src = textwrap.dedent("""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Point:
            x: int
            y: float
        """)
        module = collect_symbols(src)
        symbols = {s.name: s.type_ for s in module.symbols}

        assert symbols["Point.x"] is KNOWN
        assert symbols["Point.y"] is KNOWN

    def test_dataclass_alias_attrs(self) -> None:
        src = textwrap.dedent("""
        import dataclasses

        @dataclasses.dataclass
        class Point:
            x: int
        """)
        module = collect_symbols(src)
        symbols = {s.name: s.type_ for s in module.symbols}

        assert symbols["Point.x"] is KNOWN

    def test_namedtuple_attrs(self) -> None:
        src = textwrap.dedent("""
        from typing import NamedTuple

        class Coord(NamedTuple):
            x: int
            y: int
        """)
        module = collect_symbols(src)
        symbols = {s.name: s.type_ for s in module.symbols}

        assert symbols["Coord.x"] is KNOWN
        assert symbols["Coord.y"] is KNOWN

    def test_typeddict_attrs(self) -> None:
        src = textwrap.dedent("""
        from typing import TypedDict

        class Config(TypedDict):
            name: str
            value: int
        """)
        module = collect_symbols(src)
        symbols = {s.name: s.type_ for s in module.symbols}

        assert symbols["Config.name"] is KNOWN
        assert symbols["Config.value"] is KNOWN

    def test_typeddict_alias_attrs(self) -> None:
        src = textwrap.dedent("""
        import typing

        class Config(typing.TypedDict):
            name: str
        """)
        module = collect_symbols(src)
        symbols = {s.name: s.type_ for s in module.symbols}

        assert symbols["Config.name"] is KNOWN

    def test_regular_class_attrs_not_known(self) -> None:
        """Annotated attrs in plain classes should keep their type expression."""
        src = textwrap.dedent("""
        class Foo:
            x: int
        """)
        module = collect_symbols(src)
        symbols = {s.name: s.type_ for s in module.symbols}

        assert symbols["Foo.x"] is not KNOWN
        assert str(symbols["Foo.x"]) == "int"


class TestIsAnnotated:
    def test_markers(self) -> None:
        assert not is_annotated(UNKNOWN)
        assert not is_annotated(KNOWN)
        assert not is_annotated(EXTERNAL)

    def test_expr(self) -> None:
        assert is_annotated(Expr(cst.Name("int")))

    def test_class(self) -> None:
        assert is_annotated(Class("MyClass"))

    def test_function_unannotated(self) -> None:
        """A function with no annotations should not be considered annotated."""
        func = Function(
            "f",
            (
                Overload(
                    (
                        Param("x", ParamKind.POSITIONAL_OR_KEYWORD, UNKNOWN),
                        Param("y", ParamKind.POSITIONAL_OR_KEYWORD, UNKNOWN),
                    ),
                    UNKNOWN,
                ),
            ),
        )
        assert not is_annotated(func)

    def test_function_self_only(self) -> None:
        """A method with only self/cls inferred should not be considered annotated."""
        func = Function(
            "f",
            (
                Overload(
                    (
                        Param("self", ParamKind.POSITIONAL_OR_KEYWORD, KNOWN),
                        Param("x", ParamKind.POSITIONAL_OR_KEYWORD, UNKNOWN),
                    ),
                    UNKNOWN,
                ),
            ),
        )
        assert not is_annotated(func)

    def test_function_with_return(self) -> None:
        """A function with only a return annotation is annotated."""
        func = Function(
            "f",
            (
                Overload(
                    (Param("x", ParamKind.POSITIONAL_OR_KEYWORD, UNKNOWN),),
                    Expr(cst.Name("int")),
                ),
            ),
        )
        assert is_annotated(func)

    def test_function_with_param(self) -> None:
        """A function with at least one annotated param is annotated."""
        func = Function(
            "f",
            (
                Overload(
                    (
                        Param(
                            "x",
                            ParamKind.POSITIONAL_OR_KEYWORD,
                            Expr(cst.Name("int")),
                        ),
                        Param("y", ParamKind.POSITIONAL_OR_KEYWORD, UNKNOWN),
                    ),
                    UNKNOWN,
                ),
            ),
        )
        assert is_annotated(func)
