"""The two screens, driven through HTTP with every model replaced by a double.

What is under test is the contract between the routes and the templates: that a
verdict reaches the page, that a correction is recorded and reported back, and
that a crafted URL cannot read files it should not. None of that needs weights.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reconvision.adapters.storage.snapshots import FileSnapshotStore
from reconvision.adapters.storage.sqlite_events import SqliteEvents
from reconvision.adapters.storage.sqlite_gallery import SqliteGallery, connect
from reconvision.api.dependencies import Services
from reconvision.api.main import create_app
from reconvision.application.enrollment import EnrollmentService
from reconvision.application.feedback import FeedbackService
from reconvision.domain.events import EventVerdict, RecognitionEvent
from reconvision.domain.models import (
    BoundingBox,
    Face,
    FaceQuality,
    Frame,
    GalleryEntry,
    Identity,
    SubjectKind,
)
from reconvision.domain.quality import QualityPolicy
from tests.fakes import FakeClock, ScriptedFaceAnalyzer, build_settings

NOW = datetime(2026, 8, 30, 3, 12, tzinfo=UTC)


def frame() -> Frame:
    return np.full((80, 120, 3), 60, dtype=np.uint8)


def usable_face(rng: np.random.Generator) -> Face:
    vector = rng.standard_normal(512).astype(np.float32)
    return Face(
        box=BoundingBox(left=10, top=10, right=140, bottom=170),
        embedding=vector / np.linalg.norm(vector),
        quality=FaceQuality(
            pixel_height=160, sharpness=180.0, yaw_degrees=4.0, detection_confidence=0.97
        ),
    )


def event(**overrides: object) -> RecognitionEvent:
    defaults: dict[str, object] = {
        "camera_name": "hall",
        "verdict": EventVerdict.UNKNOWN_PERSON,
        "started_at": NOW,
        "ended_at": NOW + timedelta(seconds=6),
        "subject_kind": SubjectKind.PERSON,
        "observations": 19,
        "best_similarity": 0.29,
    }
    defaults.update(overrides)
    return RecognitionEvent(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def services(tmp_path: Path, rng: np.random.Generator) -> Services:
    """Real storage, scripted models."""
    settings = build_settings(data_dir=tmp_path, match_threshold=0.42)
    connection = connect(settings.database_path)
    gallery = SqliteGallery(connection)
    events = SqliteEvents(connection)
    snapshots = FileSnapshotStore(settings.snapshots_dir)
    analyzer = ScriptedFaceAnalyzer([usable_face(rng)])
    clock = FakeClock(NOW)

    enrollment = EnrollmentService(analyzer, gallery, clock, QualityPolicy())
    return Services(
        settings=settings,
        gallery=gallery,
        events=events,
        snapshots=snapshots,
        enrollment=enrollment,
        feedback=FeedbackService(events, snapshots, analyzer, enrollment, clock),
    )


@pytest.fixture
def client(services: Services) -> TestClient:
    return TestClient(create_app(settings=services.settings, services=services))


# --- review screen ---------------------------------------------------------------


def test_the_review_screen_lists_events(client: TestClient, services: Services) -> None:
    services.events.save(event())

    body = client.get("/").text

    assert "Unknown person" in body
    assert "hall" in body


def test_a_recognised_person_is_named(client: TestClient, services: Services) -> None:
    services.gallery.add_identity(Identity("jeremie", "Jeremie"))
    services.events.save(
        event(verdict=EventVerdict.KNOWN_PERSON, identity_id="jeremie", best_similarity=0.71)
    )

    body = client.get("/").text

    assert "jeremie" in body
    assert "verdict--known_person" in body


def test_the_scale_places_the_score_against_the_threshold(
    client: TestClient, services: Services
) -> None:
    """The signature of the screen: not just what was decided, but how close it was."""
    services.events.save(event(best_similarity=0.71))

    body = client.get("/").text

    assert "scale-threshold" in body
    assert "left: 42.0%" in body  # the calibrated threshold
    assert "left: 71.0" in body  # where this face landed


def test_an_event_with_no_comparison_shows_no_scale(client: TestClient, services: Services) -> None:
    """Nobody enrolled means no comparison happened. Drawing a dot at zero would
    read as "measured, and very low", which is a different and untrue statement."""
    services.events.save(event(verdict=EventVerdict.UNIDENTIFIED, best_similarity=-1.0))

    body = client.get("/").text

    assert "not compared" in body
    assert "scale-mark" not in body


def test_an_animal_offers_nothing_to_correct(client: TestClient, services: Services) -> None:
    services.events.save(
        event(
            verdict=EventVerdict.ANIMAL,
            subject_kind=SubjectKind.ANIMAL,
            animal_label="cat",
            best_similarity=-1.0,
        )
    )

    body = client.get("/").text

    assert "cat" in body
    assert "nothing to correct" in body


def test_events_can_be_filtered_by_camera(client: TestClient, services: Services) -> None:
    services.events.save(event(camera_name="hall"))
    services.events.save(event(camera_name="garage"))

    body = client.get("/", params={"camera": "garage"}).text

    assert "garage" in body


def test_an_empty_system_explains_what_to_do(client: TestClient) -> None:
    """An empty screen is an invitation to act, not a blank."""
    body = client.get("/").text

    assert "Nothing recorded yet" in body
    assert "reconvision run" in body


# --- corrections ------------------------------------------------------------------


def test_confirming_an_event_learns_from_its_snapshot(
    client: TestClient, services: Services
) -> None:
    """The whole reason the screen exists: a correction adds a capture taken by the
    real camera, which is worth more than any enrolment portrait."""
    services.gallery.add_identity(Identity("jeremie", "Jeremie"))
    snapshot_id = services.snapshots.save(frame(), "e1")
    stored = event(
        verdict=EventVerdict.KNOWN_PERSON, identity_id="jeremie", snapshot_id=snapshot_id
    )
    services.events.save(stored)

    response = client.post(f"/events/{stored.event_id}/feedback", data={"label": "confirmed"})

    assert response.status_code == 200
    assert "added a real capture" in response.text
    assert services.gallery.count_entries("jeremie") == 1


def test_correcting_an_identity_attributes_the_capture_to_them(
    client: TestClient, services: Services
) -> None:
    services.gallery.add_identity(Identity("jeremie", "Jeremie"))
    snapshot_id = services.snapshots.save(frame(), "e2")
    stored = event(snapshot_id=snapshot_id)
    services.events.save(stored)

    client.post(
        f"/events/{stored.event_id}/feedback",
        data={"label": "wrong_identity", "identity_id": "jeremie"},
    )

    assert services.gallery.count_entries("jeremie") == 1


def test_a_correction_is_remembered_across_reloads(client: TestClient, services: Services) -> None:
    stored = event()
    services.events.save(stored)
    client.post(f"/events/{stored.event_id}/feedback", data={"label": "not_a_person"})

    body = client.get("/").text

    assert "already corrected" in body


def test_correcting_a_missing_event_is_a_404(client: TestClient) -> None:
    response = client.post("/events/never-existed/feedback", data={"label": "confirmed"})

    assert response.status_code == 404


def test_an_unknown_correction_label_is_rejected(client: TestClient, services: Services) -> None:
    stored = event()
    services.events.save(stored)

    response = client.post(f"/events/{stored.event_id}/feedback", data={"label": "maybe"})

    assert response.status_code == 422


def test_a_correction_naming_nobody_is_rejected(client: TestClient, services: Services) -> None:
    """ "That was not me" is not actionable; the type refuses the useless half."""
    stored = event()
    services.events.save(stored)

    response = client.post(f"/events/{stored.event_id}/feedback", data={"label": "wrong_identity"})

    assert response.status_code == 422


# --- enrolment --------------------------------------------------------------------


def test_photographs_can_be_enrolled_through_the_screen(
    client: TestClient, services: Services, tmp_path: Path
) -> None:
    import cv2

    photo = tmp_path / "one.jpg"
    cv2.imwrite(str(photo), np.full((200, 200, 3), 120, dtype=np.uint8))

    response = client.post(
        "/people",
        data={"identity_id": "jeremie", "display_name": "Jeremie"},
        files=[("photos", ("one.jpg", photo.read_bytes(), "image/jpeg"))],
    )

    assert response.status_code == 200
    assert "accepted" in response.text
    assert services.gallery.count_entries("jeremie") == 1


def test_a_thin_gallery_is_flagged_on_the_screen(client: TestClient, tmp_path: Path) -> None:
    import cv2

    photo = tmp_path / "one.jpg"
    cv2.imwrite(str(photo), np.full((200, 200, 3), 120, dtype=np.uint8))

    response = client.post(
        "/people",
        data={"identity_id": "jeremie"},
        files=[("photos", ("one.jpg", photo.read_bytes(), "image/jpeg"))],
    )

    assert "10 or more" in response.text


def test_forgetting_a_person_removes_their_descriptors(
    client: TestClient, services: Services, rng: np.random.Generator
) -> None:
    services.gallery.add_identity(Identity("guest", "Guest"))
    vector = rng.standard_normal(512).astype(np.float32)
    services.gallery.add_entry(GalleryEntry("guest", vector / np.linalg.norm(vector)))

    response = client.post("/people/guest/forget")

    assert response.status_code == 200
    assert services.gallery.count_entries("guest") == 0


# --- media and health --------------------------------------------------------------


def test_a_snapshot_is_served(client: TestClient, services: Services) -> None:
    snapshot_id = services.snapshots.save(frame(), "e3")

    response = client.get(f"/snapshots/{snapshot_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("crafted", ["../../../../etc/passwd", "2026-08-30/../../../../etc/passwd"])
def test_a_crafted_snapshot_path_cannot_read_the_filesystem(
    client: TestClient, crafted: str
) -> None:
    """These URLs are reachable by anyone who reaches the screens."""
    assert client.get(f"/snapshots/{crafted}").status_code in (404, 400)


def test_health_and_metrics_are_available(client: TestClient, services: Services) -> None:
    services.events.save(event())

    assert client.get("/healthz").text == "ok"
    assert "reconvision_events_stored 1" in client.get("/metrics").text
