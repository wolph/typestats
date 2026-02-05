import anyio

from typestats.index import sources_to_module_paths


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
