"""In-memory test doubles for the domain ports.

Shared across the suite so later stages test against the same doubles the port
contracts were validated with, rather than each growing its own drifting version.
Every one is deliberately simple: a fake that needs its own tests is a liability.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta

from reconvision.domain.events import EventFeedback, RecognitionEvent
from reconvision.domain.models import Detection, Frame, GalleryEntry, Identity


class FakeClock:
    """A clock that only moves when a test tells it to."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 8, 26, 12, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


class InMemoryGalleryRepository:
    """Gallery storage without SQLite."""

    def __init__(self) -> None:
        self._identities: dict[str, Identity] = {}
        self._entries: list[GalleryEntry] = []

    def list_identities(self) -> Sequence[Identity]:
        return list(self._identities.values())

    def add_identity(self, identity: Identity) -> None:
        self._identities[identity.identity_id] = identity

    def remove_identity(self, identity_id: str) -> None:
        self._identities.pop(identity_id, None)
        self._entries = [entry for entry in self._entries if entry.identity_id != identity_id]

    def load_entries(self) -> Sequence[GalleryEntry]:
        return list(self._entries)

    def add_entry(self, entry: GalleryEntry) -> None:
        self._entries.append(entry)


class InMemoryEventRepository:
    """Event storage without SQLite."""

    def __init__(self) -> None:
        self._events: dict[str, RecognitionEvent] = {}
        self._feedback: list[EventFeedback] = []

    def save(self, event: RecognitionEvent) -> None:
        self._events[event.event_id] = event

    def get(self, event_id: str) -> RecognitionEvent | None:
        return self._events.get(event_id)

    def list_recent(
        self,
        limit: int = 50,
        camera_name: str | None = None,
        since: datetime | None = None,
    ) -> Sequence[RecognitionEvent]:
        matches = [
            event
            for event in self._events.values()
            if (camera_name is None or event.camera_name == camera_name)
            and (since is None or event.started_at >= since)
        ]
        matches.sort(key=lambda event: event.started_at, reverse=True)
        return matches[:limit]

    def save_feedback(self, feedback: EventFeedback) -> None:
        self._feedback.append(feedback)

    def list_feedback(self) -> Sequence[EventFeedback]:
        return list(self._feedback)


class RecordingNotifier:
    """Captures what would have been sent, so delivery can be asserted on."""

    def __init__(self, *, fail: bool = False) -> None:
        self.delivered: list[RecognitionEvent] = []
        self._fail = fail

    def notify(self, event: RecognitionEvent, snapshot: Frame | None = None) -> None:
        if self._fail:
            # Mirrors a real notifier: an unreachable phone is not a reason to
            # stop watching the cameras, so the failure is swallowed here too.
            return
        self.delivered.append(event)


class ScriptedFrameSource:
    """Replays a fixed list of frames, standing in for a camera."""

    def __init__(self, name: str, frames: Sequence[Frame]) -> None:
        self._name = name
        self._frames = list(frames)
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    def frames(self) -> Iterator[Frame]:
        yield from self._frames

    def close(self) -> None:
        self.closed = True


class ScriptedDetector:
    """Returns pre-arranged detections, one list per successive frame."""

    def __init__(self, per_frame: Sequence[Sequence[Detection]]) -> None:
        self._per_frame = list(per_frame)
        self._calls = 0

    def detect(self, frame: Frame) -> Sequence[Detection]:
        index = min(self._calls, len(self._per_frame) - 1)
        self._calls += 1
        return self._per_frame[index] if self._per_frame else []
