"""Command line entry point."""

from __future__ import annotations

import typer

from reconvision import __version__

app = typer.Typer(
    name="reconvision",
    help="Face recognition on home video streams.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Face recognition on home video streams."""


@app.command()
def version() -> None:
    """Print the installed version."""
    print(f"reconvision {__version__}")


if __name__ == "__main__":
    app()
