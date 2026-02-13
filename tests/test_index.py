from pathlib import Path

import anyio
import pytest

from typestats import analyze
from typestats.index import (
    _is_excluded_path,
    collect_public_symbols,
    sources_to_module_paths,
)

_FIXTURES: Path = Path(__file__).parent / "fixtures"


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
            "tests/fixtures/public_project/pkg/__init__.py",
            "tests/fixtures/public_project/",
            False,
        ),
        (
            "tests/fixtures/public_project/pkg/a.py",
            "tests/fixtures/public_project/",
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
    names = _public_symbol_names(_FIXTURES / "public_project")

    assert "pkg.public_func" in names
    assert "pkg.__version__" in names
    assert "pkg.a.public_func" in names
    assert "pkg.b" in names
    assert "pkg.spam" in names

    assert "pkg._private_func" not in names
    assert "pkg.ext" not in names
    assert "pkg._b.spam" not in names


def test_collect_public_symbols_implicit_reexports() -> None:
    names = _public_symbol_names(_FIXTURES / "public_project")

    assert "pkg.a.spam" in names
    assert "pkg.api.spam" in names
    assert "pkg.api.__version__" in names

    assert "pkg.api.a" not in names
    assert "pkg.api.public_func" not in names


def test_collect_public_symbols_explicit_private_reexports() -> None:
    names = _public_symbol_names(_FIXTURES / "private_reexport")

    assert "mylib.__version__" in names
    assert "mylib.CanAdd" in names
    assert "mylib.CanSub" in names
    assert "mylib.do_add" in names
    assert "mylib.do_mul" in names

    assert "mylib._core._can.CanAdd" not in names
    assert "mylib._core._can.CanSub" not in names
    assert "mylib._core._do.do_add" not in names
    assert "mylib._core._ops.do_mul" not in names
    assert "mylib._core.CanAdd" not in names
    assert "mylib._core.CanSub" not in names
    assert "mylib._core.do_add" not in names
    assert "mylib._core.do_mul" not in names


def test_collect_public_symbols_pyi_relative_imports() -> None:
    """Stub files with relative imports should resolve symbols correctly."""
    names = _public_symbol_names(_FIXTURES / "private_reexport_pyi")

    # The .pyi stub uses relative imports (from ._core._can import CanAdd)
    # and has an explicit __all__; all listed symbols should be public.
    assert "mylib.__version__" in names
    assert "mylib.CanAdd" in names
    assert "mylib.CanSub" in names
    assert "mylib.do_add" in names

    # Private module symbols should not leak
    assert "mylib._core._can.CanAdd" not in names
    assert "mylib._core._do.do_add" not in names


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
    types = _public_symbol_types(_FIXTURES / "public_project")

    # `sin` is imported from stdlib `math` and listed in __all__
    assert "pkg.sin" in types
    assert types["pkg.sin"] is analyze.EXTERNAL


def test_collect_public_symbols_pyi_stub_types_not_unknown() -> None:
    """Symbols typed only in .pyi stubs should not be reported as UNKNOWN."""
    types = _public_symbol_types(_FIXTURES / "stub_typed_private")

    assert "stubpkg.AnnotatedAlias" in types
    assert "stubpkg.GenericType" in types
    assert types["stubpkg.AnnotatedAlias"] is not analyze.UNKNOWN
    assert types["stubpkg.GenericType"] is not analyze.UNKNOWN


def test_collect_public_symbols_unresolved_all_names_unknown() -> None:
    """Names in __all__ that can't be resolved should be UNKNOWN."""
    types = _public_symbol_types(_FIXTURES / "public_project")

    # spam is imported from _b and should be resolved normally
    assert "pkg.lazy.spam" in types
    assert types["pkg.lazy.spam"] is not analyze.UNKNOWN

    # dynamic_a and dynamic_b are listed in __all__ but not defined
    # anywhere resolvable, so they should be UNKNOWN
    assert "pkg.lazy.dynamic_a" in types
    assert types["pkg.lazy.dynamic_a"] is analyze.UNKNOWN

    assert "pkg.lazy.dynamic_b" in types
    assert types["pkg.lazy.dynamic_b"] is analyze.UNKNOWN


def test_collect_public_symbols_same_name_module_not_unknown() -> None:
    """Functions re-exported from a submodule with the same name should not be UNKNOWN.

    When `from ._private import func` is used, and `_private` re-exports `func`
    from a submodule also named `func` (e.g. `_private/func.py`), the import
    target matches both a module path and a symbol name. The resolver should
    recognise the re-exported symbol rather than treating it as a private module
    import.
    """
    types = _public_symbol_types(_FIXTURES / "private_reexport")

    assert "mylib.do_mul" in types
    assert types["mylib.do_mul"] is not analyze.UNKNOWN
