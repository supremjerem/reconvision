"""The correction loop: a user's "that was me" has to make the system better.

Two things have to hold. A correction is always recorded, because the labelled
set it builds is worth as much as the gallery entry. And a correction only adds a
descriptor when that descriptor is good enough to help - a correct statement about
who was there, captured on a blurry frame, is still a bad thing to store.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from reconvision.application.enrollment import EnrollmentService
from reconvision.application.feedback import FeedbackService, UnknownEventError
from reconvision.domain.events import EventVerdict, FeedbackLabel, RecognitionEvent
from reconvision.domain.models import (
    BoundingBox,
    Embedding,
    Face,
    FaceQuality,
    Frame,
    GalleryEntrySource,
    Identity,
    SubjectKind,
)
from reconvision.domain.quality import QualityPolicy
from tests.fakes import (
    FakeClock,
    InMemoryEventRepository,
    InMemoryGalleryRepository,
    ScriptedFaceAnalyzer,
)
from tests.unit.conftest import random_embedding

NOW = datetime(2026, 8, 27, 3, 12, tzinfo=UTC)
SNAPSHOT_ID = "2026-08-27/passage.jpg"
ALEX = Identity(identity_id="alex", display_name="Alex")
SAM = Identity(identity_id="sam", display_name="Sam")


class InMemorySnapshots:
    """Snapshot storage without a filesystem."""

    def __init__(self) -> None:
        self.saved: dict[str, Frame] = {}

    def add(self, snapshot_id: str, frame: Frame) -> None:
        self.saved[snapshot_id] = frame

    def save(self, frame: Frame, event_id: str) -> str:
        snapshot_id = f"2026-08-27/{event_id}.jpg"
        self.saved[snapshot_id] = frame
        return snapshot_id

    def load(self, snapshot_id: str) -> Frame | None:
        return self.saved.get(snapshot_id)

    def purge_older_than(self, cutoff: datetime) -> int:
        return 0


def frame() -> Frame:
    return np.full((80, 80, 3), 64, dtype=np.uint8)


def good_face(embedding: Embedding) -> Face:
    return Face(
        box=BoundingBox(left=10, top=10, right=140, bottom=170),
        embedding=embedding,
        quality=FaceQuality(
            pixel_height=160, sharpness=180.0, yaw_degrees=3.0, detection_confidence=0.97
        ),
    )


def poor_face(embedding: Embedding) -> Face:
    return Face(
        box=BoundingBox(left=10, top=10, right=50, bottom=55),
        embedding=embedding,
        quality=FaceQuality(
            pixel_height=40, sharpness=8.0, yaw_degrees=12.0, detection_confidence=0.6
        ),
    )


def event(**overrides: object) -> RecognitionEvent:
    defaults: dict[str, object] = {
        "camera_name": "hall",
        "verdict": EventVerdict.UNKNOWN_PERSON,
        "started_at": NOW,
        "ended_at": NOW + timedelta(seconds=4),
        "subject_kind": SubjectKind.PERSON,
        "snapshot_id": SNAPSHOT_ID,
    }
    defaults.update(overrides)
    return RecognitionEvent(**defaults)  # type: ignore[arg-type]


def build(
    faces: list[Face] | None = None,
) -> tuple[FeedbackService, InMemoryEventRepository, InMemoryGalleryRepository, InMemorySnapshots]:
    events = InMemoryEventRepository()
    gallery = InMemoryGalleryRepository()
    snapshots = InMemorySnapshots()
    analyzer = ScriptedFaceAnalyzer(faces or [])
    enrollment = EnrollmentService(analyzer=analyzer, gallery=gallery, clock=FakeClock())
    service = FeedbackService(
        events=events,
        snapshots=snapshots,
        analyzer=analyzer,
        enrollment=enrollment,
        clock=FakeClock(NOW),
        quality=QualityPolicy(min_pixel_height=80),
    )
    return service, events, gallery, snapshots


def test_a_correction_on_an_unknown_event_is_rejected() -> None:
    service, *_ = build()

    with pytest.raises(UnknownEventError):
        service.submit(event_id="nope", label=FeedbackLabel.CONFIRMED)


def test_every_correction_is_recorded_even_when_nothing_is_learned() -> None:
    service, events, _, _ = build()
    stored = event()
    events.save(stored)

    outcome = service.submit(event_id=stored.event_id, label=FeedbackLabel.NOT_A_PERSON)

    assert outcome.recorded is True
    assert outcome.improved_recognition is False
    saved = events.list_feedback()
    assert len(saved) == 1
    assert saved[0].label is FeedbackLabel.NOT_A_PERSON
    assert saved[0].submitted_at == NOW


def test_confirming_a_known_person_adds_a_real_capture(rng: np.random.Generator) -> None:
    embedding = random_embedding(rng)
    service, events, gallery, snapshots = build([good_face(embedding)])
    stored = event(verdict=EventVerdict.KNOWN_PERSON, identity_id="alex")
    events.save(stored)
    gallery.add_identity(ALEX)
    snapshots.add(SNAPSHOT_ID, frame())

    outcome = service.submit(event_id=stored.event_id, label=FeedbackLabel.CONFIRMED)

    assert outcome.improved_recognition is True
    entries = gallery.load_entries()
    assert len(entries) == 1
    assert entries[0].identity_id == "alex"
    assert entries[0].source is GalleryEntrySource.CORRECTED_EVENT
    assert np.allclose(entries[0].embedding, embedding)


def test_reassigning_to_another_person_learns_for_that_person(rng: np.random.Generator) -> None:
    service, events, gallery, snapshots = build([good_face(random_embedding(rng))])
    stored = event(verdict=EventVerdict.KNOWN_PERSON, identity_id="alex")
    events.save(stored)
    gallery.add_identity(ALEX)
    gallery.add_identity(SAM)
    snapshots.add(SNAPSHOT_ID, frame())

    outcome = service.submit(
        event_id=stored.event_id,
        label=FeedbackLabel.WRONG_IDENTITY,
        corrected_identity_id="sam",
    )

    assert outcome.improved_recognition is True
    assert [entry.identity_id for entry in gallery.load_entries()] == ["sam"]


def test_a_poor_face_is_recorded_but_never_stored(rng: np.random.Generator) -> None:
    service, events, gallery, snapshots = build([poor_face(random_embedding(rng))])
    stored = event(verdict=EventVerdict.KNOWN_PERSON, identity_id="alex")
    events.save(stored)
    gallery.add_identity(ALEX)
    snapshots.add(SNAPSHOT_ID, frame())

    outcome = service.submit(event_id=stored.event_id, label=FeedbackLabel.CONFIRMED)

    assert outcome.recorded is True
    assert outcome.improved_recognition is False
    assert "too poor" in outcome.reason
    assert gallery.load_entries() == []


def test_several_faces_in_the_snapshot_stop_learning(rng: np.random.Generator) -> None:
    service, events, gallery, snapshots = build(
        [good_face(random_embedding(rng)), good_face(random_embedding(rng))]
    )
    stored = event(verdict=EventVerdict.KNOWN_PERSON, identity_id="alex")
    events.save(stored)
    gallery.add_identity(ALEX)
    snapshots.add(SNAPSHOT_ID, frame())

    outcome = service.submit(event_id=stored.event_id, label=FeedbackLabel.CONFIRMED)

    assert outcome.improved_recognition is False
    assert "single clear face" in outcome.reason
    assert gallery.load_entries() == []


def test_a_missing_snapshot_stops_learning_but_not_recording() -> None:
    service, events, gallery, _ = build([])
    stored = event(verdict=EventVerdict.KNOWN_PERSON, identity_id="alex", snapshot_id=None)
    events.save(stored)
    gallery.add_identity(ALEX)

    outcome = service.submit(event_id=stored.event_id, label=FeedbackLabel.CONFIRMED)

    assert outcome.recorded is True
    assert outcome.improved_recognition is False
    assert gallery.load_entries() == []


def test_confirming_an_unknown_person_records_without_learning() -> None:
    service, events, gallery, snapshots = build([])
    stored = event(verdict=EventVerdict.UNKNOWN_PERSON)
    events.save(stored)
    snapshots.add(SNAPSHOT_ID, frame())

    outcome = service.submit(event_id=stored.event_id, label=FeedbackLabel.CONFIRMED)

    assert outcome.recorded is True
    assert outcome.improved_recognition is False
    assert gallery.load_entries() == []
