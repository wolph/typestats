from pathlib import Path

import anyio

from typestats.index import collect_public_symbols, sources_to_module_paths

_FIXTURES: Path = Path(__file__).parent / "fixtures"


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

    assert "mylib._core._can.CanAdd" not in names
    assert "mylib._core._can.CanSub" not in names
    assert "mylib._core._do.do_add" not in names
    assert "mylib._core.CanAdd" not in names
    assert "mylib._core.CanSub" not in names
    assert "mylib._core.do_add" not in names


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
