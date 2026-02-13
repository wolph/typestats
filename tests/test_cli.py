"""Tests for the typestats CLI."""

from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from typestats.__main__ import app

runner = CliRunner()


class TestCLI:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "typestats" in result.output.lower() or "type" in result.output.lower()

    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "typestats" in result.output

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app)
        assert result.exit_code == 2  # no_args_is_help exits with code 2
        assert "Usage" in result.output or "usage" in result.output

    def test_check_help(self) -> None:
        result = runner.invoke(app, ["check", "--help"])
        assert result.exit_code == 0
        assert "package" in result.output.lower()

    def test_check_invokes_async(self) -> None:
        with patch("typestats.__main__._check_async", new_callable=AsyncMock) as mock:
            result = runner.invoke(app, ["check", "example-pkg"])
            assert result.exit_code == 0
            mock.assert_called_once()
            args = mock.call_args.args
            assert args[0] == "example-pkg"
