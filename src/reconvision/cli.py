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
from reconvision.adapters.faces.insightface_analyzer import InsightFaceAnalyzer
from reconvision.adapters.faces.lfw import (
    DatasetUnavailableError,
    ensure_lfw,
    iter_people,
)
from reconvision.adapters.images import read_image
from reconvision.adapters.video.sources import VideoSourceError
from reconvision.application.assembly import (
    build_enrollment_service,
    build_pipeline,
    open_gallery,
)
from reconvision.application.config import (
    CameraConfig,
    MissingCameraCredentialError,
    Settings,
    load_camera_manifest,
)
from reconvision.application.enrollment import EnrollmentReport, find_photos
from reconvision.application.evaluation import (
    LabelledEmbedding,
    evaluate,
    format_distribution,
)
from reconvision.application.telemetry import configure_telemetry
from reconvision.domain.events import EventVerdict, RecognitionEvent
from reconvision.domain.models import Identity

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

    Anyone enrolled with `reconvision enroll` is recognised by name. With an empty
    gallery every person is reported as unknown, while animal detection needs no
    enrolment and works immediately.
    """
    try:
        settings = Settings()
        camera = CameraConfig(name=camera_name, source=source, sample_every_n_frames=sample_every)
    except (VideoSourceError, ValidationError, ValueError) as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    telemetry = configure_telemetry(settings)

    try:
        gallery = open_gallery(settings)
        enrolled = len(gallery.list_identities())
        typer.echo(
            f"{enrolled} identity(ies) enrolled."
            if enrolled
            else "No identities enrolled yet; every person will be reported as unknown."
        )
        pipeline = build_pipeline(
            camera=camera,
            settings=settings,
            gallery=gallery,
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


@app.command()
def enroll(
    identity: Annotated[
        str, typer.Option("--identity", "-i", help="Identifier for the person, e.g. jeremie.")
    ],
    photos: Annotated[
        Path, typer.Option("--photos", "-p", help="Folder of photographs of this person.")
    ],
    display_name: Annotated[
        str | None, typer.Option("--name", help="Human-readable name. Defaults to the id.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would be enrolled, store nothing.")
    ] = False,
) -> None:
    """Teach the system to recognise someone from a folder of photographs.

    Every photograph is reported individually. A photograph containing more than
    one face is skipped rather than guessed at: enrolling the wrong face corrupts
    every later comparison and produces no error, so the failure would only ever
    surface as the system confidently naming the wrong person.
    """
    settings = Settings()
    configure_telemetry(settings)

    if not photos.is_dir():
        typer.secho(f"No such folder: {photos}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    found = find_photos(photos)
    if not found:
        typer.secho(f"No images found in {photos}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        service = build_enrollment_service(settings)
    except FileNotFoundError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Inspecting {len(found)} photo(s) in {photos} ...\n")
    results = list(service.inspect(found))
    for result in results:
        colour = typer.colors.GREEN if result.accepted else typer.colors.YELLOW
        typer.secho(f"  {result.path.name:<40} {result.describe()}", fg=colour)

    person = Identity(identity_id=identity, display_name=display_name or identity)
    report = EnrollmentReport(identity=person, results=results)

    if not dry_run and report.accepted_count:
        service.enroll(person, found, results)

    typer.echo(f"\n{report.accepted_count} of {len(results)} photo(s) usable.")
    for warning in report.warnings():
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)

    if dry_run:
        typer.echo("\nDry run: nothing was stored.")
    elif report.accepted_count:
        typer.secho(f"\nEnrolled {identity}.", fg=typer.colors.GREEN)
        typer.echo("Next: run `reconvision eval` to choose a threshold from measurement.")
    else:
        raise typer.Exit(code=1)


@app.command("eval")
def evaluate_threshold(
    lfw_people: Annotated[
        int,
        typer.Option(
            "--people",
            min=2,
            help="How many LFW identities to measure against. More is slower and tighter.",
        ),
    ] = 400,
    skip_public: Annotated[
        bool,
        typer.Option("--no-public", help="Measure only enrolled identities, skip LFW."),
    ] = False,
    show_distribution: Annotated[
        bool, typer.Option("--distribution/--no-distribution", help="Print score histograms.")
    ] = True,
) -> None:
    """Measure recognition accuracy and recommend a threshold.

    The threshold decides whether the system greets you by name or reports you as
    an intruder, and no value is correct in the abstract: it depends on the model,
    the cameras and who is enrolled. This measures the score distribution of
    same-person and different-person pairs and reports the threshold that holds
    wrong identifications to a chosen rate.

    Runs against LFW because a false-accept rate of one in a thousand cannot be
    observed with the two or three people in a household.
    """
    settings = Settings()
    configure_telemetry(settings)

    try:
        analyzer = InsightFaceAnalyzer(settings.models_dir, settings.face_model_pack)
    except FileNotFoundError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    labelled: list[LabelledEmbedding] = []

    gallery = open_gallery(settings)
    enrolled = list(gallery.load_entries())
    labelled += [
        LabelledEmbedding(identity_id=entry.identity_id, embedding=entry.embedding)
        for entry in enrolled
    ]
    if enrolled:
        typer.echo(f"Enrolled: {len(enrolled)} embedding(s) across your own identities.")

    if not skip_public:
        typer.echo(f"Encoding up to {lfw_people} LFW identities (first run downloads ~180 MB) ...")
        try:
            labelled += _encode_lfw(analyzer, settings.data_dir, lfw_people)
        except DatasetUnavailableError as error:
            typer.secho(f"Public dataset unavailable: {error}", fg=typer.colors.YELLOW, err=True)
            typer.echo("Continuing with enrolled identities only.")

    report = evaluate(labelled)

    if not report.genuine_pairs or not report.impostor_pairs:
        typer.secho(
            "Not enough data to measure: at least two identities, each with at least "
            "two usable photographs, are needed.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"\n{report.identities} identities, "
        f"{report.genuine_pairs:,} same-person pairs, "
        f"{report.impostor_pairs:,} different-person pairs."
    )
    typer.echo(
        f"Mean separation: {report.separation:.3f}   "
        f"Equal error rate: {report.equal_error_rate:.2%}"
    )

    if show_distribution:
        typer.echo("\nSame person:")
        for line in format_distribution(report.genuine_scores):
            typer.echo(f"  {line}")
        typer.echo("\nDifferent people:")
        for line in format_distribution(report.impostor_scores):
            typer.echo(f"  {line}")

    typer.echo("\nOperating points:")
    for point in report.operating_points:
        typer.echo(f"  {point.describe()}")

    recommended = report.recommended()
    if recommended is not None:
        typer.secho(
            f"\nRecommended: RECONVISION_MATCH_THRESHOLD={recommended.threshold:.2f}",
            fg=typer.colors.GREEN,
        )
        typer.echo(
            f"At that threshold roughly {recommended.true_accept_rate:.0%} of genuine faces "
            f"are recognised, and about 1 stranger in 1000 comparisons is wrongly named."
        )


def _encode_lfw(
    analyzer: InsightFaceAnalyzer, data_dir: Path, people: int
) -> list[LabelledEmbedding]:
    """Embed a slice of LFW, skipping photographs with no clearly usable face."""
    root = ensure_lfw(data_dir)
    encoded: list[LabelledEmbedding] = []

    with typer.progressbar(list(iter_people(root, limit=people)), label="Encoding") as people_bar:
        for person in people_bar:
            for photo in person.photos:
                image = read_image(photo)
                if image is None:
                    continue
                faces = analyzer.analyse(image)
                # One face only: a group photograph would attach a bystander's
                # descriptor to this person's label and corrupt the measurement.
                if len(faces) == 1:
                    encoded.append(
                        LabelledEmbedding(
                            identity_id=person.identity_id, embedding=faces[0].embedding
                        )
                    )
    return encoded
