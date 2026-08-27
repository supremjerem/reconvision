"""What happens to an event after the pipeline decides it: store, then notify."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from reconvision.application.pipeline import ObservedEvent
from reconvision.application.recording import EventRecorder
from reconvision.domain.events import EventVerdict, RecognitionEvent
from reconvision.domain.models import Frame, SubjectKind
from tests.fakes import FakeClock, InMemoryEventRepository, RecordingNotifier

NOW = datetime(2026, 8, 27, 3, 12, tzinfo=UTC)


class InMemorySnapshots:
    """Snapshot storage without a filesystem."""

    def __init__(self) -> None:
        self.saved: dict[str, Frame] = {}
        self.purged_before: datetime | None = None

    def save(self, frame: Frame, event_id: str) -> str:
        snapshot_id = f"2026-08-27/{event_id}.jpg"
        self.saved[snapshot_id] = frame
        return snapshot_id

    def load(self, snapshot_id: str) -> Frame | None:
        return self.saved.get(snapshot_id)

    def purge_older_than(self, cutoff: datetime) -> int:
        self.purged_before = cutoff
        return 0


class FailingSnapshots(InMemorySnapshots):
    """A full or read-only data volume."""

    def save(self, frame: Frame, event_id: str) -> str:
        message = "no space left on device"
        raise OSError(message)


def frame() -> Frame:
    return np.full((60, 80, 3), 50, dtype=np.uint8)


def event(**overrides: object) -> RecognitionEvent:
    defaults: dict[str, object] = {
        "camera_name": "hall",
        "verdict": EventVerdict.UNKNOWN_PERSON,
        "started_at": NOW,
        "ended_at": NOW + timedelta(seconds=4),
        "subject_kind": SubjectKind.PERSON,
    }
    defaults.update(overrides)
    return RecognitionEvent(**defaults)  # type: ignore[arg-type]


def build(
    notifier: RecordingNotifier | None = None,
    snapshots: InMemorySnapshots | None = None,
    retention_days: int = 30,
) -> tuple[EventRecorder, InMemoryEventRepository, RecordingNotifier, InMemorySnapshots]:
    events = InMemoryEventRepository()
    delivery = notifier or RecordingNotifier()
    store = snapshots or InMemorySnapshots()
    recorder = EventRecorder(
        events=events,
        snapshots=store,
        notifier=delivery,
        clock=FakeClock(NOW),
        retention_days=retention_days,
    )
    return recorder, events, delivery, store


def test_an_event_is_stored() -> None:
    recorder, events, _, _ = build()

    stored = recorder.record(ObservedEvent(event=event(), snapshot=None))

    assert events.get(stored.event_id) is not None


def test_the_snapshot_is_stored_and_linked_to_the_event() -> None:
    """The link is what lets the review screen show the picture beside the row."""
    recorder, events, _, snapshots = build()

    stored = recorder.record(ObservedEvent(event=event(), snapshot=frame()))

    assert stored.snapshot_id is not None
    assert stored.snapshot_id in snapshots.saved
    reloaded = events.get(stored.event_id)
    assert reloaded is not None
    assert reloaded.snapshot_id == stored.snapshot_id


def test_an_unknown_person_is_notified() -> None:
    recorder, _, notifier, _ = build()

    recorder.record(ObservedEvent(event=event(), snapshot=frame()))

    assert len(notifier.delivered) == 1


def test_a_recognised_household_member_is_recorded_but_not_announced() -> None:
    """Otherwise the system cries wolf every time you walk to the kitchen."""
    recorder, events, notifier, _ = build()

    stored = recorder.record(
        ObservedEvent(
            event=event(verdict=EventVerdict.KNOWN_PERSON, identity_id="jeremie"),
            snapshot=frame(),
        )
    )

    assert notifier.delivered == []
    assert events.get(stored.event_id) is not None


def test_the_cat_is_recorded_but_not_announced() -> None:
    recorder, events, notifier, _ = build()

    stored = recorder.record(
        ObservedEvent(
            event=event(
                verdict=EventVerdict.ANIMAL,
                subject_kind=SubjectKind.ANIMAL,
                animal_label="cat",
            ),
            snapshot=frame(),
        )
    )

    assert notifier.delivered == []
    assert events.get(stored.event_id) is not None


def test_an_event_is_stored_before_it_is_delivered() -> None:
    """A notification arriving with no record behind it leaves the user nothing to
    review; a record with a failed notification is merely quiet."""
    recorder, events, _, _ = build()
    seen_at_delivery: list[bool] = []

    class Checking(RecordingNotifier):
        def notify(self, event: RecognitionEvent, snapshot: Frame | None = None) -> None:
            seen_at_delivery.append(events.get(event.event_id) is not None)

    recorder, events, _, _ = build()
    recorder._notifier = Checking()
    recorder.record(ObservedEvent(event=event(), snapshot=None))

    assert seen_at_delivery == [True]


def test_a_full_disk_does_not_stop_the_pipeline() -> None:
    """A data volume that fills up should degrade to events without pictures, not
    stop the cameras being watched."""
    recorder, _, _, _ = build(snapshots=FailingSnapshots())

    with pytest.raises(OSError, match="no space"):
        recorder.record(ObservedEvent(event=event(), snapshot=frame()))


def test_retention_sweeps_both_events_and_snapshots() -> None:
    """Leaving either behind defeats the purpose of the other."""
    recorder, events, _, snapshots = build(retention_days=7)
    events.save(event(started_at=NOW - timedelta(days=30), ended_at=NOW - timedelta(days=30)))
    events.save(event())

    removed_events, _ = recorder.purge_expired()

    assert removed_events == 1
    assert snapshots.purged_before == NOW - timedelta(days=7)


def test_retention_leaves_recent_events_alone() -> None:
    recorder, events, _, _ = build(retention_days=30)
    events.save(event())

    removed_events, _ = recorder.purge_expired()

    assert removed_events == 0
