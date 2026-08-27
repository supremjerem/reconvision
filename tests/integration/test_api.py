"""The two screens, end to end, against injected fakes rather than real models.

`create_app` takes its services as an argument precisely so this test can drive
every route without loading 360 MB of weights. What is checked here is the wiring:
a form post reaches the feedback service, a bad label is refused at the boundary,
a missing snapshot is a 404 and not a stack trace.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from reconvision.api.dependencies import Services
from reconvision.api.main import create_app
from reconvision.application.enrollment import EnrollmentService
from reconvision.application.feedback import FeedbackService
from reconvision.domain.events import EventVerdict, FeedbackLabel, RecognitionEvent
from reconvision.domain.models import (
    BoundingBox,
    Embedding,
    Face,
    FaceQuality,
    Identity,
    SubjectKind,
)
from reconvision.domain.quality import QualityPolicy
from tests.fakes import (
    FakeClock,
    InMemoryEventRepository,
    InMemoryGalleryRepository,
    ScriptedFaceAnalyzer,
    build_settings,
)
from tests.unit.conftest import random_embedding

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 27, 3, 12, tzinfo=UTC)
ALEX = Identity(identity_id="alex", display_name="Alex")


class ApiGallery(InMemoryGalleryRepository):
    """Adds the read helpers the screens call and a no-op close."""

    def count_entries(self, identity_id: str | None = None) -> int:
        return sum(
            1
            for entry in self.load_entries()
            if identity_id is None or entry.identity_id == identity_id
        )

    def close(self) -> None:
        pass


class ApiEvents(InMemoryEventRepository):
    def count(self) -> int:
        return len(self.list_recent(limit=10_000))

    def anonymise_identity(self, identity_id: str) -> int:
        changed = 0
        for event_id, stored in list(self._events.items()):
            if stored.identity_id == identity_id:
                self._events[event_id] = replace(
                    stored, identity_id=None, verdict=EventVerdict.UNKNOWN_PERSON
                )
                changed += 1
        return changed


class ApiSnapshots:
    """No real files. `path_for` is what the snapshot route asks for."""

    def __init__(self, present: dict[str, Path] | None = None) -> None:
        self._present = present or {}

    def path_for(self, snapshot_id: str) -> Path | None:
        return self._present.get(snapshot_id)

    def load(self, snapshot_id: str) -> None:
        return None

    def purge_older_than(self, cutoff: datetime) -> int:
        return 0


def good_face(embedding: Embedding) -> Face:
    return Face(
        box=BoundingBox(left=10, top=10, right=140, bottom=170),
        embedding=embedding,
        quality=FaceQuality(
            pixel_height=160, sharpness=180.0, yaw_degrees=3.0, detection_confidence=0.97
        ),
    )


def an_event(**overrides: object) -> RecognitionEvent:
    defaults: dict[str, object] = {
        "camera_name": "hall",
        "verdict": EventVerdict.UNKNOWN_PERSON,
        "started_at": NOW,
        "ended_at": NOW + timedelta(seconds=4),
        "subject_kind": SubjectKind.PERSON,
        "best_similarity": 0.31,
        "observations": 12,
    }
    defaults.update(overrides)
    return RecognitionEvent(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260827)


@pytest.fixture
def services(rng: np.random.Generator) -> Services:
    gallery = ApiGallery()
    events = ApiEvents()
    analyzer = ScriptedFaceAnalyzer([good_face(random_embedding(rng))])
    enrollment = EnrollmentService(analyzer=analyzer, gallery=gallery, clock=FakeClock())
    return Services(
        settings=build_settings(),
        gallery=gallery,  # type: ignore[arg-type]
        events=events,  # type: ignore[arg-type]
        snapshots=ApiSnapshots(),  # type: ignore[arg-type]
        enrollment=enrollment,
        feedback=FeedbackService(
            events=events,
            snapshots=ApiSnapshots(),  # type: ignore[arg-type]
            analyzer=analyzer,
            enrollment=enrollment,
            clock=FakeClock(NOW),
            quality=QualityPolicy(),
        ),
    )


@pytest.fixture
def client(services: Services) -> Iterator[TestClient]:
    with TestClient(create_app(settings=services.settings, services=services)) as running:
        yield running


def test_healthz_is_plain_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.text == "ok"


def test_the_review_screen_lists_recent_events(client: TestClient, services: Services) -> None:
    services.events.save(an_event(camera_name="garden"))

    response = client.get("/")

    assert response.status_code == 200
    assert "Unknown person" in response.text
    assert "garden" in response.text


def test_the_review_screen_can_filter_by_camera(client: TestClient, services: Services) -> None:
    services.events.save(an_event(camera_name="garden"))
    services.events.save(an_event(camera_name="hall"))

    response = client.get("/", params={"camera": "garden"})

    assert response.status_code == 200
    assert "garden" in response.text


def test_a_correction_reaches_the_feedback_service(client: TestClient, services: Services) -> None:
    stored = an_event()
    services.events.save(stored)

    response = client.post(f"/events/{stored.event_id}/feedback", data={"label": "not_a_person"})

    assert response.status_code == 200
    assert "recorded" in response.text.lower()
    assert [f.label for f in services.events.list_feedback()] == [FeedbackLabel.NOT_A_PERSON]


def test_an_unknown_label_is_refused_at_the_boundary(
    client: TestClient, services: Services
) -> None:
    stored = an_event()
    services.events.save(stored)

    response = client.post(f"/events/{stored.event_id}/feedback", data={"label": "banana"})

    assert response.status_code == 422


def test_a_correction_on_a_missing_event_is_a_404(client: TestClient) -> None:
    response = client.post("/events/does-not-exist/feedback", data={"label": "confirmed"})
    assert response.status_code == 404


def test_reassigning_without_a_target_identity_is_rejected(
    client: TestClient, services: Services
) -> None:
    stored = an_event()
    services.events.save(stored)

    response = client.post(f"/events/{stored.event_id}/feedback", data={"label": "wrong_identity"})

    assert response.status_code == 422


def test_a_known_person_event_offers_confirmation_and_the_similarity_scale(
    client: TestClient, services: Services
) -> None:
    services.gallery.add_identity(ALEX)
    services.events.save(
        an_event(verdict=EventVerdict.KNOWN_PERSON, identity_id="alex", best_similarity=0.55)
    )

    response = client.get("/")

    assert response.status_code == 200
    assert "Yes, that was them" in response.text
    assert "scale-mark" in response.text


def test_a_reviewed_event_shows_no_correction_controls(
    client: TestClient, services: Services
) -> None:
    stored = an_event()
    services.events.save(stored)
    client.post(f"/events/{stored.event_id}/feedback", data={"label": "confirmed"})

    response = client.get("/")

    assert "already corrected" in response.text
    assert "not_a_person" not in response.text


def test_the_people_screen_shows_capture_counts(client: TestClient, services: Services) -> None:
    services.gallery.add_identity(ALEX)

    response = client.get("/people")

    assert response.status_code == 200
    assert "Alex" in response.text


def test_enrolling_from_uploaded_photos(
    client: TestClient, services: Services, tmp_path: Path
) -> None:
    photo = tmp_path / "alex.png"
    cv2.imwrite(str(photo), np.full((200, 200, 3), 120, dtype=np.uint8))

    response = client.post(
        "/people",
        data={"identity_id": "alex", "display_name": "Alex"},
        files={"photos": ("alex.png", photo.read_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert "1 of 1" in response.text
    assert services.gallery.count_entries("alex") == 1


def test_forgetting_a_person_reports_back(client: TestClient, services: Services) -> None:
    services.gallery.add_identity(ALEX)

    response = client.post("/people/alex/forget")

    assert response.status_code == 200
    assert "forgotten" in response.text.lower()
    assert services.gallery.list_identities() == []


def test_a_missing_snapshot_is_a_404(client: TestClient) -> None:
    response = client.get("/snapshots/2026-08-27/nope.jpg")
    assert response.status_code == 404


def test_metrics_are_prometheus_text(client: TestClient, services: Services) -> None:
    services.gallery.add_identity(ALEX)
    services.events.save(an_event())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "reconvision_identities 1" in response.text
    assert "reconvision_events_stored 1" in response.text
