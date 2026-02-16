from pathlib import Path

import anyio
import pytest

from typestats import analyze
from typestats.index import (
    _is_excluded_path,
    _resolve_expr_name,
    _resolves_to_any,
    collect_public_symbols,
    sources_to_module_paths,
)

_FIXTURES: Path = Path(__file__).parent / "fixtures"
_PROJECT: Path = _FIXTURES / "project"


@pytest.mark.parametrize(
    ("path", "prefix", "expected"),
    [
        # Paths within an sdist (no prefix needed)
        ("numpy/_core/tests/test_abc.py", "", True),
        ("numpy/tests/__init__.py", "", True),
        ("benchmarks/bench_core.py", "", True),
        ("benchmarks/benchmarks/bench_app.py", "", True),
        ("doc/source/conf.py", "", True),
        ("docs/conf.py", "", True),
        ("tools/changelog.py", "", True),
        ("examples/tutorial.py", "", True),
        (".spin/cmds.py", "", True),
        ("numpy/random/_examples/cffi/extending.py", "", True),
        ("numpy/conftest.py", "", True),
        ("conftest.py", "", True),
        ("numpy/__init__.py", "", False),
        ("numpy/_core/__init__.py", "", False),
        ("numpy/linalg/__init__.pyi", "", False),
        ("numpy/testing/__init__.pyi", "", False),
        ("numpy/f2py/__init__.pyi", "", False),
        # Prefix stripping: project under a tests/ directory should not match
        (
            "tests/fixtures/project/pkg/__init__.py",
            "tests/fixtures/project/",
            False,
        ),
        (
            "tests/fixtures/project/pkg/a.py",
            "tests/fixtures/project/",
            False,
        ),
    ],
)
def test_is_excluded_path(path: str, prefix: str, expected: bool) -> None:
    assert _is_excluded_path(path, prefix=prefix) == expected


def _public_symbol_names(project_dir: Path) -> set[str]:
    """Collect all public symbol names from a fixture project."""

    async def _run() -> set[str]:
        symbols_by_path = await collect_public_symbols(project_dir)
        return {
            symbol.name for symbols in symbols_by_path.values() for symbol in symbols
        }

    return anyio.run(_run)


def test_sources_to_module_paths_package_and_module() -> None:
    pkg = anyio.Path("pkg")

    result = sources_to_module_paths([
        pkg / "__init__.py",
        pkg / "mod.py",
        pkg / "mod.pyi",
        anyio.Path("single.py"),
    ])

    assert result["pkg"] == frozenset({pkg / "__init__.py"})
    assert result["pkg.mod"] == frozenset({pkg / "mod.py", pkg / "mod.pyi"})
    assert result["single"] == frozenset({anyio.Path("single.py")})


def test_sources_to_module_paths_excludes_non_package_subdirs() -> None:
    """Files in directories without __init__ nested inside a package are excluded."""
    pkg = anyio.Path("numpy")
    linalg = pkg / "linalg"
    vendored = linalg / "lapack_lite"

    result = sources_to_module_paths([
        pkg / "__init__.py",
        linalg / "__init__.py",
        linalg / "linalg.py",
        # These are in a non-package subdir (no __init__.py in lapack_lite/)
        vendored / "clapack_scrub.py",
        vendored / "fortran.py",
        vendored / "make_lite.py",
        # This stub sits alongside the directory, not inside it
        linalg / "lapack_lite.pyi",
    ])

    # The vendored files should be excluded
    assert all("clapack_scrub" not in k for k in result)
    assert all("fortran" not in k for k in result)
    assert all("make_lite" not in k for k in result)
    # The stub file at the package level should remain
    assert "numpy.linalg.lapack_lite" in result
    # Normal package contents should remain
    assert "numpy" in result
    assert "numpy.linalg" in result
    assert "numpy.linalg.linalg" in result


def test_sources_to_module_paths_stubs_only() -> None:
    stubs = anyio.Path("proj-stubs")

    result = sources_to_module_paths([
        stubs / "__init__.pyi",
        stubs / "util.pyi",
    ])

    assert result["proj"] == frozenset({stubs / "__init__.pyi"})
    assert result["proj.util"] == frozenset({stubs / "util.pyi"})


def test_sources_to_module_paths_stubs_extra() -> None:
    app = anyio.Path("src/app")
    app_stubs = anyio.Path("src/app-stubs")
    lib_stubs = anyio.Path("typings/lib-stubs")

    result = sources_to_module_paths([
        app / "__init__.py",
        app / "util.py",
        app_stubs / "__init__.pyi",
        app_stubs / "util.pyi",
        lib_stubs / "__init__.pyi",
        lib_stubs / "util.pyi",
    ])

    assert result["app"] == frozenset({app / "__init__.py", app_stubs / "__init__.pyi"})
    assert result["app.util"] == frozenset({app / "util.py", app_stubs / "util.pyi"})

    assert result["lib"] == frozenset({lib_stubs / "__init__.pyi"})
    assert result["lib.util"] == frozenset({lib_stubs / "util.pyi"})


def test_collect_public_symbols_respects_imports() -> None:
    names = _public_symbol_names(_PROJECT)

    assert "pkg.__version__" in names
    assert "pkg.a.public_func" in names
    assert "pkg.a._private_func" in names  # in pkg.__all__, traced to origin
    assert "pkg.a.spam" in names
    assert "pkg.b" not in names  # module reference, skipped
    assert "pkg._b.spam" in names  # origin of pkg's re-export
    assert "pkg.sin" in names  # EXTERNAL

    assert "pkg.public_func" not in names  # attributed to origin pkg.a
    assert "pkg.spam" not in names  # attributed to origin pkg._b
    assert "pkg._private_func" not in names  # attributed to origin pkg.a
    assert "pkg.ext" not in names


def test_collect_public_symbols_implicit_reexports() -> None:
    names = _public_symbol_names(_PROJECT)

    assert "pkg.a.spam" in names
    assert "pkg.api.__version__" in names

    assert "pkg.api.spam" not in names  # attributed to origin pkg.a.spam
    assert "pkg.api.a" not in names
    assert "pkg.api.public_func" not in names


def test_collect_public_symbols_explicit_private_reexports() -> None:
    names = _public_symbol_names(_PROJECT)

    assert "mylib.__version__" in names
    assert "mylib._core._can.CanAdd" in names
    assert "mylib._core._can.CanSub" in names
    assert "mylib._core._do.do_add" in names
    assert "mylib._core._ops.do_mul.do_mul" in names

    assert "mylib.CanAdd" not in names  # attributed to origin
    assert "mylib.CanSub" not in names
    assert "mylib.do_add" not in names
    assert "mylib.do_mul" not in names
    assert "mylib._core.CanAdd" not in names  # intermediate, not origin
    assert "mylib._core.CanSub" not in names
    assert "mylib._core.do_add" not in names
    assert "mylib._core.do_mul" not in names


def test_collect_public_symbols_pyi_relative_imports() -> None:
    """Stub files with relative imports should resolve symbols correctly."""
    names = _public_symbol_names(_PROJECT)

    # The .pyi stub uses relative imports (from ._core._can import CanAdd)
    # and has an explicit __all__; symbols are traced to their origin.
    assert "mylib_pyi.__version__" in names
    assert "mylib_pyi._core._can.CanAdd" in names
    assert "mylib_pyi._core._can.CanSub" in names
    assert "mylib_pyi._core._do.do_add" in names

    # Re-exporting module names should not appear
    assert "mylib_pyi.CanAdd" not in names
    assert "mylib_pyi.CanSub" not in names
    assert "mylib_pyi.do_add" not in names


def _public_symbol_types(project_dir: Path) -> dict[str, analyze.TypeForm]:
    """Collect public symbol names mapped to their resolved types."""

    async def _run() -> dict[str, analyze.TypeForm]:
        symbols_by_path = await collect_public_symbols(project_dir)
        return {
            symbol.name: symbol.type_
            for symbols in symbols_by_path.values()
            for symbol in symbols
        }

    return anyio.run(_run)


def test_collect_public_symbols_external_reexport() -> None:
    """Re-exports from external packages should be marked EXTERNAL."""
    types = _public_symbol_types(_PROJECT)

    # `sin` is imported from stdlib `math` and listed in __all__
    assert "pkg.sin" in types
    assert types["pkg.sin"] is analyze.EXTERNAL


def test_collect_public_symbols_pyi_stub_types_not_unknown() -> None:
    """Symbols typed only in .pyi stubs should not be reported as UNKNOWN."""
    types = _public_symbol_types(_PROJECT)

    assert "stubpkg._typeforms.AnnotatedAlias" in types
    assert "stubpkg._typeforms.GenericType" in types
    assert types["stubpkg._typeforms.AnnotatedAlias"] is not analyze.UNKNOWN
    assert types["stubpkg._typeforms.GenericType"] is not analyze.UNKNOWN


def test_collect_public_symbols_unresolved_all_names_unknown() -> None:
    """Names in __all__ that can't be resolved should be UNKNOWN."""
    types = _public_symbol_types(_PROJECT)

    # spam is imported from _b; origin is pkg._b.spam
    assert "pkg._b.spam" in types
    assert types["pkg._b.spam"] is not analyze.UNKNOWN
    assert "pkg.lazy.spam" not in types  # attributed to origin

    # dynamic_a and dynamic_b are listed in __all__ but not defined
    # anywhere resolvable, so they should be UNKNOWN
    assert "pkg.lazy.dynamic_a" in types
    assert types["pkg.lazy.dynamic_a"] is analyze.UNKNOWN

    assert "pkg.lazy.dynamic_b" in types
    assert types["pkg.lazy.dynamic_b"] is analyze.UNKNOWN


def test_collect_public_symbols_same_name_module_not_unknown() -> None:
    """Functions re-exported from a submodule with the same name should not be UNKNOWN.

    When `from ._private import func` is used, and `_private` re-exports `func`
    from a submodule also named `func` (e.g. `_private/func.py`), the import chain
    is followed to the original definition.
    """
    types = _public_symbol_types(_PROJECT)

    assert "mylib._core._ops.do_mul.do_mul" in types
    assert types["mylib._core._ops.do_mul.do_mul"] is not analyze.UNKNOWN


class TestResolveExprName:
    def test_direct_import(self) -> None:
        assert _resolve_expr_name("Any", {"Any": "typing.Any"}, "mod") == "typing.Any"

    def test_dotted_import(self) -> None:
        assert (
            _resolve_expr_name("typing.Any", {"typing": "typing"}, "mod")
            == "typing.Any"
        )

    def test_aliased_import(self) -> None:
        assert _resolve_expr_name("t.Any", {"t": "typing"}, "mod") == "typing.Any"

    def test_local_fallback(self) -> None:
        assert _resolve_expr_name("Unknown", {}, "mymod") == "mymod.Unknown"


class TestResolvesToAny:
    def test_typing_any(self) -> None:
        assert _resolves_to_any("typing.Any", {})

    def test_typing_extensions_any(self) -> None:
        assert _resolves_to_any("typing_extensions.Any", {})

    def test_not_any(self) -> None:
        assert not _resolves_to_any("builtins.int", {})

    def test_alias_chain(self) -> None:
        aliases = {
            "mod.Unknown": "typing.Any",
        }
        assert _resolves_to_any("mod.Unknown", aliases)

    def test_chained_aliases(self) -> None:
        aliases = {
            "mod.Chained": "mod.Unknown",
            "mod.Unknown": "typing.Any",
        }
        assert _resolves_to_any("mod.Chained", aliases)

    def test_circular_alias_not_any(self) -> None:
        aliases = {
            "mod.A": "mod.B",
            "mod.B": "mod.A",
        }
        assert not _resolves_to_any("mod.A", aliases)

    def test_alias_to_non_any(self) -> None:
        aliases = {
            "mod.MyInt": "builtins.int",
        }
        assert not _resolves_to_any("mod.MyInt", aliases)

    def test_typeshed_incomplete(self) -> None:
        assert _resolves_to_any("_typeshed.Incomplete", {})

    def test_typeshed_maybe_none(self) -> None:
        assert _resolves_to_any("_typeshed.MaybeNone", {})

    def test_typeshed_sentinel(self) -> None:
        assert _resolves_to_any("_typeshed.sentinel", {})

    def test_typeshed_annotation_form(self) -> None:
        assert _resolves_to_any("_typeshed.AnnotationForm", {})

    def test_alias_to_typeshed_incomplete(self) -> None:
        aliases = {"mod.X": "_typeshed.Incomplete"}
        assert _resolves_to_any("mod.X", aliases)


def test_collect_public_symbols_direct_any_is_any() -> None:
    """Symbols annotated with `typing.Any` should be ANY."""
    types = _public_symbol_types(_PROJECT)

    assert "anypkg.mod.any_var" in types
    assert types["anypkg.mod.any_var"] is analyze.ANY


def test_collect_public_symbols_alias_to_any_is_any() -> None:
    """Symbols annotated with a type alias that resolves to Any should be ANY."""
    types = _public_symbol_types(_PROJECT)

    # Unknown is defined as `type Unknown = Any` in _defs
    assert "anypkg.mod.unknown_var" in types
    assert types["anypkg.mod.unknown_var"] is analyze.ANY


def test_collect_public_symbols_chained_alias_to_any_is_any() -> None:
    """Aliases through multiple levels should still resolve to ANY."""
    types = _public_symbol_types(_PROJECT)

    # Chained is defined as `type Chained = Unknown`, Unknown → Any
    assert "anypkg.mod.chained_var" in types
    assert types["anypkg.mod.chained_var"] is analyze.ANY


def test_collect_public_symbols_cross_module_alias_to_any_is_any() -> None:
    """Type aliases imported from other modules should be resolved."""
    types = _public_symbol_types(_PROJECT)

    # Remote is `type Remote = Any` in _defs, used in mod
    assert "anypkg.mod.remote_var" in types
    assert types["anypkg.mod.remote_var"] is analyze.ANY


def test_collect_public_symbols_non_any_alias_not_any() -> None:
    """Type aliases that don't resolve to Any should remain annotated."""
    types = _public_symbol_types(_PROJECT)

    # NotAny is `type NotAny = int`, so not_any_alias_var should still be annotated
    assert "anypkg.mod.not_any_alias_var" in types
    assert types["anypkg.mod.not_any_alias_var"] is not analyze.ANY

    assert "anypkg.mod.normal_var" in types
    assert types["anypkg.mod.normal_var"] is not analyze.ANY


def test_collect_public_symbols_function_any_params_unfolded() -> None:
    """Function params annotated with Any aliases should be unfolded to ANY."""
    types = _public_symbol_types(_PROJECT)

    assert "anypkg.mod.annotated_func" in types
    func = types["anypkg.mod.annotated_func"]
    assert isinstance(func, analyze.Function)

    overload = func.overloads[0]
    # param `a: Unknown` should be ANY (Unknown → Any)
    assert overload.params[0].name == "a"
    assert overload.params[0].annotation is analyze.ANY
    # param `b: int` should remain annotated
    assert overload.params[1].name == "b"
    assert overload.params[1].annotation is not analyze.ANY
    # return `str` should remain annotated
    assert overload.returns is not analyze.ANY
