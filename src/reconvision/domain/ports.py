"""The boundaries between the recognition rules and the outside world.

Every port is a `Protocol`, so adapters satisfy them structurally: nothing in
`adapters/` imports a base class from here, and nothing here knows an adapter
exists. That is what keeps the domain suite runnable with no model, no camera and
no database, and what confines a change of storage or tracker to one file.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from reconvision.domain.events import EventFeedback, RecognitionEvent
from reconvision.domain.models import (
    BoundingBox,
    Detection,
    Face,
    Frame,
    GalleryEntry,
    Identity,
    TrackedDetection,
)


@runtime_checkable
class FrameSource(Protocol):
    """A stream of frames from one camera, file or capture device."""

    @property
    def name(self) -> str:
        """Stable identifier used to attribute events to a camera."""

    def frames(self) -> Iterator[Frame]:
        """Yield decoded frames until the source ends or is closed.

        Implementations reconnect internally on transient failure, and drop frames
        rather than queue them when the consumer falls behind: for live
        recognition a stale frame has no value, so latency is preserved over
        completeness.
        """

    def close(self) -> None: ...


@runtime_checkable
class ObjectDetector(Protocol):
    """Locates people and animals, the cheap gate in front of the face stage."""

    def detect(self, frame: Frame) -> Sequence[Detection]: ...


@runtime_checkable
class FaceAnalyzer(Protocol):
    """Detects, aligns and embeds faces."""

    def analyse(self, frame: Frame, region: BoundingBox | None = None) -> Sequence[Face]:
        """Return the faces found, optionally restricted to one person's box.

        Restricting to a region is what makes the face stage affordable: the
        detector has already said where the person is, so there is no reason to
        search the whole frame again.
        """


@runtime_checkable
class Tracker(Protocol):
    """Associates detections across frames so identity is decided per path.

    Track expiry is deliberately not part of this contract. Trackers reclaim
    their own state, and the pipeline decides when a track has *ended* by
    watching for its id to stop appearing - which is a policy question about how
    long someone may be occluded, not something a tracker should answer.
    """

    def update(self, detections: Sequence[Detection]) -> Sequence[TrackedDetection]: ...


@runtime_checkable
class GalleryRepository(Protocol):
    """Stores enrolled identities and their embeddings."""

    def list_identities(self) -> Sequence[Identity]: ...

    def add_identity(self, identity: Identity) -> None: ...

    def remove_identity(self, identity_id: str) -> None:
        """Remove an identity and every embedding belonging to it."""

    def load_entries(self) -> Sequence[GalleryEntry]: ...

    def add_entry(self, entry: GalleryEntry) -> None: ...


@runtime_checkable
class EventRepository(Protocol):
    """Stores recognition events and the corrections made against them."""

    def save(self, event: RecognitionEvent) -> None: ...

    def get(self, event_id: str) -> RecognitionEvent | None: ...

    def list_recent(
        self,
        limit: int = 50,
        camera_name: str | None = None,
        since: datetime | None = None,
    ) -> Sequence[RecognitionEvent]: ...

    def save_feedback(self, feedback: EventFeedback) -> None: ...

    def list_feedback(self) -> Sequence[EventFeedback]: ...


@runtime_checkable
class SnapshotStore(Protocol):
    """Persists the single best frame of an event.

    Separate from the event repository because images are large binary blobs with
    their own retention rules, and because a database is the wrong place for them.
    """

    def save(self, frame: Frame, event_id: str) -> str:
        """Store a frame and return its snapshot id."""

    def load(self, snapshot_id: str) -> Frame | None: ...

    def purge_older_than(self, cutoff: datetime) -> int:
        """Delete snapshots taken before the cutoff, returning how many went."""


@runtime_checkable
class Notifier(Protocol):
    """Delivers an event somewhere a human will see it."""

    def notify(self, event: RecognitionEvent, snapshot: Frame | None = None) -> None:
        """Deliver the event.

        Implementations must not raise on delivery failure: a phone that is
        unreachable is not a reason to stop watching the cameras. They log and
        move on.
        """


@runtime_checkable
class Clock(Protocol):
    """The current time, injected so event timing is testable."""

    def now(self) -> datetime: ...
