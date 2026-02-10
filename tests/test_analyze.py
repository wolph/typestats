import textwrap

from typestats.analyze import KNOWN, collect_symbols


def test_imports() -> None:
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


def test_exports_implicit_direct() -> None:
    src = textwrap.dedent("""
    import a
    import b as _b
    import c as c
    """)
    exports = collect_symbols(src).exports_implicit
    assert exports == {"c"}


def test_exports_implicit_from() -> None:
    src = textwrap.dedent("""
    from m import a
    from m import b as _b
    from m import c as c
    """)
    exports = collect_symbols(src).exports_implicit
    assert exports == {"c"}


def test_exports_explicit() -> None:
    src = """__all__ = ["a", "b", "c"]"""
    exports = collect_symbols(src).exports_explicit
    assert exports == {"a", "b", "c"}


def test_exports_explicit_missing() -> None:
    src = """a = 1"""
    exports = collect_symbols(src).exports_explicit
    assert exports is None


def test_type_aliases() -> None:
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


def test_type_alias_indirect() -> None:
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


def test_symbols() -> None:
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


def test_symbols_no_type_alias() -> None:
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


def test_special_typeforms_ignored_aliases() -> None:
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


def test_special_typeforms_ignored_annassign() -> None:
    src = textwrap.dedent("""
    import typing as t

    T: object = t.TypeVar("T")
    D: int = 1
    """)
    symbols = collect_symbols(src).symbols
    assert symbols[0].name == "D"
    assert len(symbols) == 1


def test_ignore_comments() -> None:
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


def test_annotated_unwrap() -> None:
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


def test_annotated_unwrap_indirect_import() -> None:
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


def test_enum_members_are_known() -> None:
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


def test_enum_members_are_known_with_alias() -> None:
    src = textwrap.dedent("""
    from enum import Enum as MyEnum

    class Status(MyEnum):
        READY = 1
    """)
    module = collect_symbols(src)
    symbols = {symbol.name: symbol.type_ for symbol in module.symbols}

    assert symbols["Status.READY"] is KNOWN


def test_assign_imported_name_is_import_alias() -> None:
    """X = Never (imported) should become an import alias, not an UNKNOWN symbol."""
    src = textwrap.dedent("""
    from typing import Never

    Complex64 = Never
    """)
    module = collect_symbols(src)
    imports = dict(module.imports)
    assert imports["Complex64"] == "typing.Never"
    assert len(module.symbols) == 0


def test_assign_local_name_is_type_alias() -> None:
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


def test_dataclass_attrs_are_known() -> None:
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


def test_dataclass_call_attrs_are_known() -> None:
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


def test_dataclass_alias_attrs_are_known() -> None:
    src = textwrap.dedent("""
    import dataclasses

    @dataclasses.dataclass
    class Point:
        x: int
    """)
    module = collect_symbols(src)
    symbols = {s.name: s.type_ for s in module.symbols}

    assert symbols["Point.x"] is KNOWN


def test_namedtuple_attrs_are_known() -> None:
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


def test_typeddict_attrs_are_known() -> None:
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


def test_typeddict_alias_attrs_are_known() -> None:
    src = textwrap.dedent("""
    import typing

    class Config(typing.TypedDict):
        name: str
    """)
    module = collect_symbols(src)
    symbols = {s.name: s.type_ for s in module.symbols}

    assert symbols["Config.name"] is KNOWN


def test_regular_class_attrs_not_known() -> None:
    """Annotated attrs in plain classes should keep their type expression."""
    src = textwrap.dedent("""
    class Foo:
        x: int
    """)
    module = collect_symbols(src)
    symbols = {s.name: s.type_ for s in module.symbols}

    assert symbols["Foo.x"] is not KNOWN
    assert str(symbols["Foo.x"]) == "int"
