import textwrap

from typestats.analyze import collect_symbols


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
    A: TypeAlias = str
    B = TypeAliasType("B", str)
    type C = str
    D = str
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
