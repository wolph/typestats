import pytest

from typestats._pypi import (
    FileDetail,
    ProjectDetail,
    _latest_distribution,
    _latest_sdist,
    _latest_wheel,
    _NoDistributionFoundError,
)


def _make_file(filename: str, *, yanked: bool = False) -> FileDetail:
    return FileDetail(
        filename=filename,
        hashes={"sha256": "abc123"},
        size=1000,
        url=f"https://example.com/{filename}",
        yanked=yanked,
    )


def _make_project(filenames: list[str]) -> ProjectDetail:
    return ProjectDetail(
        name="example",
        files=[_make_file(f) for f in filenames],
        meta={"api-version": "1.0"},
        versions=["1.0.0"],
    )


class TestLatestSdist:
    def test_tar_gz(self) -> None:
        details = _make_project(["example-1.0.0.tar.gz"])
        sdist = _latest_sdist(details)
        assert sdist["filename"] == "example-1.0.0.tar.gz"

    def test_zip(self) -> None:
        details = _make_project(["example-1.0.0.zip"])
        sdist = _latest_sdist(details)
        assert sdist["filename"] == "example-1.0.0.zip"

    def test_latest_version(self) -> None:
        details = _make_project([
            "example-1.0.0.tar.gz",
            "example-2.0.0.tar.gz",
        ])
        sdist = _latest_sdist(details)
        assert sdist["filename"] == "example-2.0.0.tar.gz"

    def test_no_sdist_raises(self) -> None:
        details = _make_project(["example-1.0.0-py3-none-any.whl"])
        with pytest.raises(_NoDistributionFoundError, match="No sdist found"):
            _latest_sdist(details)

    def test_yanked_excluded(self) -> None:
        details = _make_project(["example-1.0.0.tar.gz"])
        details["files"][0]["yanked"] = True
        with pytest.raises(_NoDistributionFoundError, match="No sdist found"):
            _latest_sdist(details)


class TestLatestWheel:
    def test_basic(self) -> None:
        details = _make_project(["example-1.0.0-py3-none-any.whl"])
        wheel = _latest_wheel(details)
        assert wheel["filename"] == "example-1.0.0-py3-none-any.whl"

    def test_prefers_pure_python(self) -> None:
        details = _make_project([
            "example-1.0.0-cp314-cp314-linux_x86_64.whl",
            "example-1.0.0-py3-none-any.whl",
        ])
        wheel = _latest_wheel(details)
        assert "none-any" in wheel["filename"]

    def test_latest_version(self) -> None:
        details = _make_project([
            "example-1.0.0-py3-none-any.whl",
            "example-2.0.0-py3-none-any.whl",
        ])
        wheel = _latest_wheel(details)
        assert wheel["filename"] == "example-2.0.0-py3-none-any.whl"

    def test_no_wheel_raises(self) -> None:
        details = _make_project(["example-1.0.0.tar.gz"])
        with pytest.raises(_NoDistributionFoundError, match="No wheel found"):
            _latest_wheel(details)

    def test_yanked_excluded(self) -> None:
        details = _make_project(["example-1.0.0-py3-none-any.whl"])
        details["files"][0]["yanked"] = True
        with pytest.raises(_NoDistributionFoundError, match="No wheel found"):
            _latest_wheel(details)

    def test_latest_pure_version(self) -> None:
        details = _make_project([
            "example-1.0.0-py3-none-any.whl",
            "example-2.0.0-cp314-cp314-linux_x86_64.whl",
            "example-2.0.0-py3-none-any.whl",
        ])
        wheel = _latest_wheel(details)
        assert wheel["filename"] == "example-2.0.0-py3-none-any.whl"


class TestLatestDistribution:
    def test_prefers_sdist(self) -> None:
        details = _make_project([
            "example-1.0.0.tar.gz",
            "example-1.0.0-py3-none-any.whl",
        ])
        file_detail, kind = _latest_distribution(details)
        assert kind == "sdist"
        assert file_detail["filename"] == "example-1.0.0.tar.gz"

    def test_falls_back_to_wheel(self) -> None:
        details = _make_project(["example-1.0.0-py3-none-any.whl"])
        file_detail, kind = _latest_distribution(details)
        assert kind == "wheel"
        assert file_detail["filename"] == "example-1.0.0-py3-none-any.whl"

    def test_no_valid_distributions_raises(self) -> None:
        details = _make_project(["example-1.0.0-readme.txt"])
        details["files"] = []
        with pytest.raises(_NoDistributionFoundError):
            _latest_distribution(details)
