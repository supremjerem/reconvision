"""Command line entry point."""

from __future__ import annotations

import signal
from pathlib import Path
from types import FrameType
from typing import Annotated

import typer
from pydantic import ValidationError

from reconvision import __version__
from reconvision.adapters.video.sources import VideoSourceError, create_frame_source
from reconvision.application.config import (
    MissingCameraCredentialError,
    Settings,
    load_camera_manifest,
)
from reconvision.application.ingest import FrameIngestor
from reconvision.application.telemetry import configure_telemetry

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


@app.command()
def run(
    source: str = typer.Option(
        ...,
        "--source",
        "-s",
        help="webcam:0, an rtsp:// URL, or a path to a video file.",
    ),
    camera_name: str = typer.Option("camera", "--name", help="Name recorded on events."),
    sample_every: int = typer.Option(3, "--sample-every", min=1, help="Analyse one frame in N."),
    max_frames: int | None = typer.Option(
        None, "--max-frames", help="Stop after this many analysed frames."
    ),
) -> None:
    """Watch a source and report what reaches the detector.

    Detection and recognition are wired in the next stage; for now this measures
    the ingestion budget, which is what determines how many cameras a machine can
    carry.
    """
    try:
        settings = Settings()
        frame_source = create_frame_source(source, name=camera_name)
    except (VideoSourceError, ValidationError, ValueError) as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    telemetry = configure_telemetry(settings)
    ingestor = FrameIngestor(frame_source, telemetry, sample_every_n_frames=sample_every)

    stopping = False

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True
        frame_source.close()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    typer.echo(f"Watching {source} as {camera_name!r}. Ctrl-C to stop.")
    try:
        for analysed, _frame in enumerate(ingestor.analysable_frames(), start=1):
            if stopping or (max_frames is not None and analysed >= max_frames):
                break
    finally:
        frame_source.close()
        ingestor.log_throughput()
        stats = ingestor.stats
        typer.echo(
            f"\n{stats.decoded} frames decoded ({stats.decoded_fps:.1f}/s), "
            f"{stats.analysed} analysed ({stats.analysed_fps:.1f}/s), "
            f"{stats.skipped_ratio:.0%} skipped before any model ran."
        )


@app.command()
def check(
    cameras: Annotated[Path, typer.Option("--cameras", help="Path to the camera manifest.")] = Path(
        "cameras.yaml"
    ),
) -> None:
    """Validate the configuration without opening a single camera.

    Catches the mistakes that would otherwise surface as an unreadable connection
    error hours into a deployment: a missing credential, a duplicate camera name,
    a threshold outside the valid range.
    """
    try:
        settings = Settings()
        manifest = load_camera_manifest(cameras)
    except (MissingCameraCredentialError, ValidationError, ValueError) as error:
        # This command exists to surface exactly these mistakes. Reporting them
        # as a stack trace would defeat the point, so they are presented plainly
        # and the exit code is left non-zero for scripts and healthchecks.
        typer.secho(f"Configuration is invalid:\n{error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Data directory:   {settings.data_dir}")
    typer.echo(f"Match threshold:  {settings.match_threshold} (margin {settings.min_match_margin})")
    typer.echo(f"Notifiers:        {', '.join(settings.enabled_notifiers) or 'none configured'}")
    typer.echo(f"Telemetry:        {settings.telemetry_exporter.value}")
    typer.echo(f"Cameras enabled:  {len(manifest.enabled)} of {len(manifest.cameras)}")
    for camera in manifest.cameras:
        state = "enabled" if camera.enabled else "disabled"
        typer.echo(f"  - {camera.name} ({state}, 1 frame in {camera.sample_every_n_frames})")


if __name__ == "__main__":
    app()
