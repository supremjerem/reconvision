"""Event persistence, snapshots and the retention that limits both."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from reconvision.adapters.storage.snapshots import FileSnapshotStore
from reconvision.adapters.storage.sqlite_events import SqliteEvents
from reconvision.adapters.storage.sqlite_gallery import SqliteGallery, connect
from reconvision.application.assembly import forget_identity
from reconvision.domain.events import (
    EventFeedback,
    EventVerdict,
    FeedbackLabel,
    RecognitionEvent,
)
from reconvision.domain.models import Frame, Identity, SubjectKind
from reconvision.domain.ports import EventRepository, SnapshotStore

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


@pytest.fixture
def events(tmp_path: Path) -> SqliteEvents:
    return SqliteEvents(connect(tmp_path / "reconvision.db"))


@pytest.fixture
def snapshots(tmp_path: Path) -> FileSnapshotStore:
    return FileSnapshotStore(tmp_path / "snapshots")


def frame() -> Frame:
    return np.full((90, 120, 3), 70, dtype=np.uint8)


def event(**overrides: object) -> RecognitionEvent:
    defaults: dict[str, object] = {
        "camera_name": "hall",
        "verdict": EventVerdict.UNKNOWN_PERSON,
        "started_at": NOW,
        "ended_at": NOW + timedelta(seconds=5),
        "subject_kind": SubjectKind.PERSON,
        "observations": 12,
    }
    defaults.update(overrides)
    return RecognitionEvent(**defaults)  # type: ignore[arg-type]


# --- events ---------------------------------------------------------------------


def test_the_repository_satisfies_its_port(events: SqliteEvents) -> None:
    port: EventRepository = events

    assert isinstance(port, EventRepository)


def test_an_event_survives_the_round_trip(events: SqliteEvents) -> None:
    original = event(
        verdict=EventVerdict.ANIMAL, subject_kind=SubjectKind.ANIMAL, animal_label="cat"
    )

    events.save(original)

    stored = events.get(original.event_id)
    assert stored is not None
    assert stored.verdict is EventVerdict.ANIMAL
    assert stored.animal_label == "cat"
    assert stored.started_at == NOW


def test_rewriting_an_event_replaces_it(events: SqliteEvents) -> None:
    """A restart mid-passage can write the same id twice; losing the second write
    would discard the more complete record."""
    first = event(observations=3)
    events.save(first)

    events.save(replace(first, observations=40))

    stored = events.get(first.event_id)
    assert stored is not None
    assert stored.observations == 40


def test_events_are_listed_newest_first(events: SqliteEvents) -> None:
    """How the review screen reads them, and the only order anyone wants."""
    for minutes in (0, 30, 60):
        events.save(
            event(
                started_at=NOW + timedelta(minutes=minutes),
                ended_at=NOW + timedelta(minutes=minutes, seconds=5),
            )
        )

    listed = events.list_recent()

    assert [e.started_at for e in listed] == sorted((e.started_at for e in listed), reverse=True)


def test_events_can_be_filtered_by_camera(events: SqliteEvents) -> None:
    events.save(event(camera_name="hall"))
    events.save(event(camera_name="garage"))

    assert len(events.list_recent(camera_name="garage")) == 1


def test_events_can_be_filtered_by_time(events: SqliteEvents) -> None:
    events.save(event(started_at=NOW - timedelta(days=2), ended_at=NOW - timedelta(days=2)))
    events.save(event())

    assert len(events.list_recent(since=NOW - timedelta(hours=1))) == 1


def test_an_unknown_event_is_none(events: SqliteEvents) -> None:
    assert events.get("never-existed") is None


def test_a_correction_is_stored_against_its_event(events: SqliteEvents) -> None:
    stored = event()
    events.save(stored)

    events.save_feedback(
        EventFeedback(event_id=stored.event_id, label=FeedbackLabel.CONFIRMED, submitted_at=NOW)
    )

    (feedback,) = events.list_feedback()
    assert feedback.label is FeedbackLabel.CONFIRMED


def test_a_correction_naming_a_person_keeps_that_name(tmp_path: Path) -> None:
    connection = connect(tmp_path / "reconvision.db")
    SqliteGallery(connection).add_identity(Identity("jeremie", "Jeremie"))
    events = SqliteEvents(connection)
    stored = event()
    events.save(stored)

    events.save_feedback(
        EventFeedback(
            event_id=stored.event_id,
            label=FeedbackLabel.WRONG_IDENTITY,
            submitted_at=NOW,
            corrected_identity_id="jeremie",
        )
    )

    (feedback,) = events.list_feedback()
    assert feedback.corrected_identity_id == "jeremie"


def test_forgetting_a_person_unnames_their_history_without_erasing_it(
    tmp_path: Path,
) -> None:
    """Their face descriptors go; the record that someone passed through stays, as
    an unknown person. Nulling the name alone would leave a row claiming to be a
    known person with nobody attached, which the domain refuses to represent."""
    connection = connect(tmp_path / "reconvision.db")
    gallery = SqliteGallery(connection)
    events = SqliteEvents(connection)
    gallery.add_identity(Identity("guest", "Guest"))
    stored = event(verdict=EventVerdict.KNOWN_PERSON, identity_id="guest")
    events.save(stored)

    changed = forget_identity(gallery, events, "guest")

    assert changed == 1
    reloaded = events.get(stored.event_id)
    assert reloaded is not None
    assert reloaded.identity_id is None
    assert reloaded.verdict is EventVerdict.UNKNOWN_PERSON
    assert gallery.count_entries("guest") == 0


def test_old_events_are_deleted(events: SqliteEvents) -> None:
    """Retention is enforced, not offered: a camera on a hallway builds a record of
    everyone who lives there."""
    events.save(event(started_at=NOW - timedelta(days=90), ended_at=NOW - timedelta(days=90)))
    events.save(event())

    removed = events.delete_older_than(NOW - timedelta(days=30))

    assert removed == 1
    assert events.count() == 1


# --- snapshots ------------------------------------------------------------------


def test_the_snapshot_store_satisfies_its_port(snapshots: FileSnapshotStore) -> None:
    port: SnapshotStore = snapshots

    assert isinstance(port, SnapshotStore)


def test_a_snapshot_survives_the_round_trip(snapshots: FileSnapshotStore) -> None:
    snapshot_id = snapshots.save(frame(), "event-1")

    loaded = snapshots.load(snapshot_id)

    assert loaded is not None
    assert loaded.shape == (90, 120, 3)


def test_snapshots_are_grouped_by_day(snapshots: FileSnapshotStore) -> None:
    """So retention is a directory removal rather than a scan of a year of images."""
    snapshot_id = snapshots.save(frame(), "event-1")

    assert snapshot_id.count("/") == 1
    assert snapshot_id.endswith("event-1.jpg")


def test_a_missing_snapshot_is_none(snapshots: FileSnapshotStore) -> None:
    assert snapshots.load("2020-01-01/absent.jpg") is None


@pytest.mark.parametrize(
    "crafted",
    ["../../../etc/passwd", "2026-08-27/../../../../etc/passwd"],
)
def test_a_crafted_snapshot_id_cannot_escape_the_store(
    snapshots: FileSnapshotStore, crafted: str
) -> None:
    """Snapshot ids arrive from HTTP requests, so a crafted one must not be able to
    read arbitrary files off the machine."""
    assert snapshots.load(crafted) is None
    assert snapshots.path_for(crafted) is None


def test_old_snapshots_are_purged(tmp_path: Path) -> None:
    store = FileSnapshotStore(tmp_path / "snapshots")
    old_day = tmp_path / "snapshots" / "2020-01-01"
    old_day.mkdir(parents=True)
    (old_day / "ancient.jpg").write_bytes(b"\xff\xd8\xff")
    store.save(frame(), "recent")

    removed = store.purge_older_than(datetime(2026, 1, 1, tzinfo=UTC))

    assert removed == 1
    assert not old_day.exists()


def test_purging_ignores_unexpected_directories(tmp_path: Path) -> None:
    """A stray folder in the data volume is not a reason to abort the sweep."""
    store = FileSnapshotStore(tmp_path / "snapshots")
    (tmp_path / "snapshots" / "not-a-date").mkdir(parents=True)

    assert store.purge_older_than(datetime(2026, 1, 1, tzinfo=UTC)) == 0
