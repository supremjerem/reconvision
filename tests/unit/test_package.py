"""Smoke tests proving the package is installed and importable."""

from typer.testing import CliRunner

from reconvision import __version__
from reconvision.cli import app


def test_version_is_semver() -> None:
    major, minor, patch = __version__.split(".")
    assert all(part.isdigit() for part in (major, minor, patch))


def test_version_command_reports_the_package_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout
