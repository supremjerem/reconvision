"""Configuration is a system boundary, so it validates rather than trusts."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from reconvision.application.config import (
    CameraManifest,
    MissingCameraCredentialError,
    TelemetryExporter,
    load_camera_manifest,
)
from tests.fakes import build_settings


def write_manifest(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cameras.yaml"
    path.write_text(body, encoding="utf-8")
    return path


settings = build_settings


def test_defaults_are_usable_without_any_environment() -> None:
    assert settings().data_dir == Path("./data")
    assert settings().telemetry_exporter is TelemetryExporter.CONSOLE


def test_data_paths_derive_from_a_single_directory() -> None:
    """One volume to mount on the NAS, one place to back up."""
    configured = settings(data_dir=Path("/srv/reconvision"))

    assert configured.models_dir == Path("/srv/reconvision/models")
    assert configured.snapshots_dir == Path("/srv/reconvision/snapshots")
    assert configured.database_path == Path("/srv/reconvision/reconvision.db")


def test_a_threshold_outside_cosine_range_is_refused() -> None:
    with pytest.raises(ValidationError):
        settings(match_threshold=1.4)


def test_a_channel_is_enabled_by_being_configured() -> None:
    """No separate enable flag, so nothing can drift out of sync with the config."""
    configured = settings(ntfy_topic="my-topic", webhook_url="https://example.test/hook")

    assert configured.enabled_notifiers == ("ntfy", "webhook")


def test_no_channels_are_enabled_out_of_the_box() -> None:
    assert settings().enabled_notifiers == ()


def test_a_manifest_resolves_credentials_from_the_environment(tmp_path: Path) -> None:
    """The point of the indirection: the real URL never touches a versioned file."""
    path = write_manifest(
        tmp_path,
        "cameras:\n  - name: living_room\n    source: ${LIVING_ROOM_STREAM_URL}\n",
    )

    manifest = load_camera_manifest(path, environ={"LIVING_ROOM_STREAM_URL": "rtsp://cam/1"})

    assert manifest.cameras[0].source == "rtsp://cam/1"


def test_a_missing_credential_fails_immediately_and_says_which(tmp_path: Path) -> None:
    """Left unresolved, a missing password surfaces hours later as an unreadable
    connection error. It fails here instead, naming the variable to add."""
    path = write_manifest(tmp_path, "cameras:\n  - name: living_room\n    source: ${NOT_SET}\n")

    with pytest.raises(MissingCameraCredentialError) as raised:
        load_camera_manifest(path, environ={})

    assert raised.value.variable == "NOT_SET"
    assert "living_room" in str(raised.value)
    assert ".env" in str(raised.value)


def test_a_missing_manifest_is_not_an_error(tmp_path: Path) -> None:
    """A fresh install has no cameras yet, which is a valid state, not a crash."""
    assert load_camera_manifest(tmp_path / "absent.yaml").cameras == []


def test_duplicate_camera_names_are_refused() -> None:
    """Names key the events, so a duplicate would silently merge two rooms."""
    with pytest.raises(ValidationError, match="unique"):
        CameraManifest.model_validate(
            {
                "cameras": [
                    {"name": "hall", "source": "a"},
                    {"name": "hall", "source": "b"},
                ]
            }
        )


def test_a_camera_name_unsuitable_for_a_metric_label_is_refused() -> None:
    with pytest.raises(ValidationError):
        CameraManifest.model_validate({"cameras": [{"name": "Living Room!", "source": "a"}]})


def test_disabled_cameras_are_parsed_but_not_run(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path,
        "cameras:\n"
        "  - name: hall\n    source: rtsp://a\n"
        "  - name: garage\n    source: rtsp://b\n    enabled: false\n",
    )

    manifest = load_camera_manifest(path, environ={})

    assert len(manifest.cameras) == 2
    assert [camera.name for camera in manifest.enabled] == ["hall"]


def test_an_absurd_sampling_rate_is_refused() -> None:
    with pytest.raises(ValidationError):
        CameraManifest.model_validate(
            {"cameras": [{"name": "hall", "source": "a", "sample_every_n_frames": 0}]}
        )
