"""Command line entry point."""

from __future__ import annotations

import signal
from collections import Counter
from pathlib import Path
from types import FrameType
from typing import Annotated

import typer
from pydantic import ValidationError

from reconvision import __version__
from reconvision.adapters.storage.memory import InMemoryGallery
from reconvision.adapters.video.sources import VideoSourceError
from reconvision.application.assembly import build_pipeline
from reconvision.application.config import (
    CameraConfig,
    MissingCameraCredentialError,
    Settings,
    load_camera_manifest,
)
from reconvision.application.telemetry import configure_telemetry
from reconvision.domain.events import EventVerdict, RecognitionEvent

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


@app.command("export-models")
def export_models(
    detector_weights: Annotated[
        str, typer.Option("--detector", help="Ultralytics weights to convert.")
    ] = "yolo11s.pt",
    skip_faces: Annotated[
        bool, typer.Option("--skip-faces", help="Only export the object detector.")
    ] = False,
) -> None:
    """Download and convert the model weights the pipeline needs.

    Run once after installing. The weights are roughly 360 MB and are written to
    the data directory rather than the repository, which is public.
    """
    # Imported here, not at module scope: this pulls in Ultralytics and therefore
    # PyTorch, which the runtime container deliberately does not carry, so every
    # other command must remain usable without it.
    try:
        from reconvision.tooling.model_export import (
            export_detector,
            fetch_face_models,
            write_manifest,
        )
    except ImportError:
        typer.secho(
            "Model export needs the export extra: uv sync --extra export",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None

    settings = Settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Exporting detector {detector_weights} -> ONNX ...")
    exported = [export_detector(settings.models_dir, detector_weights)]
    typer.echo(f"  {exported[0].path.name}  {exported[0].size_mb:.1f} MB")

    if not skip_faces:
        typer.echo(f"Fetching face models ({settings.face_model_pack}) ...")
        exported.append(fetch_face_models(settings.models_dir, settings.face_model_pack))
        typer.echo(f"  {exported[-1].name}  {exported[-1].size_mb:.1f} MB")

    manifest = write_manifest(settings.models_dir, exported)
    typer.echo(f"\nWrote {manifest}")
    typer.secho("Models ready.", fg=typer.colors.GREEN)


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
    """Watch a source and report who and what passes in front of it.

    Until identities are enrolled the gallery is empty, so every person is
    reported as unknown. Animal detection needs no enrolment and works
    immediately.
    """
    try:
        settings = Settings()
        camera = CameraConfig(name=camera_name, source=source, sample_every_n_frames=sample_every)
    except (VideoSourceError, ValidationError, ValueError) as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    telemetry = configure_telemetry(settings)

    try:
        pipeline = build_pipeline(
            camera=camera,
            settings=settings,
            gallery=InMemoryGallery(),
            telemetry=telemetry,
        )
    except (FileNotFoundError, VideoSourceError) as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    stopping = False

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True
        pipeline.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    typer.echo(f"Watching {source} as {camera_name!r}. Ctrl-C to stop.\n")
    counts: Counter[str] = Counter()
    try:
        for observed in pipeline.events():
            counts[observed.event.verdict.value] += 1
            typer.echo(_describe(observed.event))
            if stopping or (max_frames is not None and pipeline.analysed_frames >= max_frames):
                break
    finally:
        pipeline.close()
        stats = pipeline.ingest_stats
        typer.echo(
            f"\n{stats.decoded} frames decoded ({stats.decoded_fps:.1f}/s), "
            f"{stats.analysed} analysed, "
            f"{stats.skipped_ratio:.0%} skipped before any model ran."
        )
        summary = ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items()))
        typer.echo(f"Events: {summary or 'none'}")


def _describe(event: RecognitionEvent) -> str:
    """One human-readable line per passage."""
    when = event.started_at.strftime("%H:%M:%S")
    if event.verdict is EventVerdict.ANIMAL:
        return f"{when}  animal ({event.animal_label})"
    if event.verdict is EventVerdict.KNOWN_PERSON:
        return (
            f"{when}  {event.identity_id} "
            f"(similarity {event.best_similarity:.2f}, seen in {event.observations} frames)"
        )
    if event.verdict is EventVerdict.UNKNOWN_PERSON:
        return f"{when}  unknown person (seen in {event.observations} frames)"
    return f"{when}  person, no usable face (seen in {event.observations} frames)"


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
