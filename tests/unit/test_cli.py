"""The CLI is the operator's interface, so its failures must be readable."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from reconvision import __version__
from reconvision.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def quiet_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep console exporters out of the captured CLI output."""
    monkeypatch.setenv("RECONVISION_TELEMETRY_EXPORTER", "none")


def write_manifest(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cameras.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_version_reports_the_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_check_summarises_a_valid_configuration(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path, "cameras:\n  - name: hall\n    source: rtsp://cam.local/1\n"
    )

    result = runner.invoke(app, ["check", "--cameras", str(manifest)])

    assert result.exit_code == 0
    assert "hall" in result.stdout
    assert "Cameras enabled:  1 of 1" in result.stdout


def test_check_never_prints_a_camera_url(tmp_path: Path) -> None:
    """Its output is the natural thing to paste into a bug report, so it must not
    contain the credentials embedded in a stream URL."""
    manifest = write_manifest(
        tmp_path,
        "cameras:\n  - name: hall\n    source: rtsp://admin:secret@cam.local/1\n",
    )

    result = runner.invoke(app, ["check", "--cameras", str(manifest)])

    assert result.exit_code == 0
    assert "secret" not in result.stdout
    assert "cam.local" not in result.stdout


def test_check_reports_a_missing_credential_without_a_stack_trace(tmp_path: Path) -> None:
    """This command exists to catch exactly this mistake; a traceback would be a
    worse answer than the one it already knows how to give."""
    manifest = write_manifest(
        tmp_path, "cameras:\n  - name: hall\n    source: ${ABSENT_VARIABLE}\n"
    )

    result = runner.invoke(app, ["check", "--cameras", str(manifest)])

    assert result.exit_code == 1
    assert "ABSENT_VARIABLE" in result.output
    assert "Traceback" not in result.output


def test_check_accepts_a_configuration_with_no_cameras_yet(tmp_path: Path) -> None:
    """A fresh install is a valid state, not an error."""
    result = runner.invoke(app, ["check", "--cameras", str(tmp_path / "absent.yaml")])

    assert result.exit_code == 0
    assert "Cameras enabled:  0 of 0" in result.stdout


def test_run_reports_throughput_for_a_recorded_clip(tmp_path: Path) -> None:
    from tests.fixtures.video import still_frame, walk_past, write_video

    clip = write_video(tmp_path / "clip.mp4", [still_frame()] * 20 + walk_past(steps=10))

    result = runner.invoke(app, ["run", "--source", str(clip), "--sample-every", "1"])

    assert result.exit_code == 0
    assert "frames decoded" in result.stdout
    assert "skipped before any model ran" in result.stdout


def test_run_rejects_a_malformed_source_without_a_stack_trace() -> None:
    result = runner.invoke(app, ["run", "--source", "webcam:front"])

    assert result.exit_code == 1
    assert "webcam:<index>" in result.output
    assert "Traceback" not in result.output


def test_run_stops_at_the_requested_frame_count(tmp_path: Path) -> None:
    """Bounded runs are what make the pipeline testable against real footage."""
    from tests.fixtures.video import frame_with_subject, write_video

    clip = write_video(tmp_path / "clip.mp4", [frame_with_subject(x) for x in range(0, 300, 10)])

    result = runner.invoke(
        app, ["run", "--source", str(clip), "--sample-every", "1", "--max-frames", "5"]
    )

    assert result.exit_code == 0
