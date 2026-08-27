"""The correction loop: turning a user's "that was me" into a better system.

This is what the review screen exists for, and the reason it is not a dashboard.
A correction does two things that nothing else can:

- it adds a descriptor captured by the real camera, at the real angle, in the real
  light, which is worth far more to the matcher than any enrolment portrait;
- it builds a labelled set from the household's own footage, so the threshold can
  later be recalibrated against reality rather than against a public benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from reconvision.application.enrollment import EnrollmentService
from reconvision.domain.events import EventFeedback, FeedbackLabel
from reconvision.domain.ports import (
    Clock,
    EventRepository,
    FaceAnalyzer,
    SnapshotStore,
)
from reconvision.domain.quality import QualityPolicy

logger = structlog.get_logger(__name__)


class UnknownEventError(LookupError):
    """A correction referred to an event that does not exist."""


@dataclass(frozen=True, slots=True)
class FeedbackOutcome:
    """What a correction changed."""

    recorded: bool
    gallery_entry_added: bool
    reason: str

    @property
    def improved_recognition(self) -> bool:
        return self.gallery_entry_added


class FeedbackService:
    """Applies a user correction to an event."""

    def __init__(
        self,
        events: EventRepository,
        snapshots: SnapshotStore,
        analyzer: FaceAnalyzer,
        enrollment: EnrollmentService,
        clock: Clock,
        quality: QualityPolicy | None = None,
    ) -> None:
        self._events = events
        self._snapshots = snapshots
        self._analyzer = analyzer
        self._enrollment = enrollment
        self._clock = clock
        self._quality = quality or QualityPolicy()

    def submit(
        self,
        event_id: str,
        label: FeedbackLabel,
        corrected_identity_id: str | None = None,
    ) -> FeedbackOutcome:
        """Record a correction and, where it helps, learn from it."""
        event = self._events.get(event_id)
        if event is None:
            message = f"No event {event_id!r}"
            raise UnknownEventError(message)

        feedback = EventFeedback(
            event_id=event_id,
            label=label,
            corrected_identity_id=corrected_identity_id,
            submitted_at=self._clock.now(),
        )
        self._events.save_feedback(feedback)

        identity_id = corrected_identity_id or (
            event.identity_id if label is FeedbackLabel.CONFIRMED else None
        )
        if identity_id is None:
            # "That was not a person" carries no face to learn from, and a
            # confirmed unknown person is not someone to enrol.
            return FeedbackOutcome(True, False, "recorded; nothing to learn from")

        added, reason = self._learn_from(event.snapshot_id, identity_id)
        logger.info(
            "feedback_applied",
            event_id=event_id,
            label=label.value,
            identity=identity_id,
            learned=added,
        )
        return FeedbackOutcome(True, added, reason)

    def _learn_from(self, snapshot_id: str | None, identity_id: str) -> tuple[bool, str]:
        """Add the event's face to a person's gallery, if it is good enough.

        The same quality bar as enrolment applies. A correction on a blurry frame
        is a correct statement about who was there and a bad descriptor to store,
        and storing it would degrade every future comparison.
        """
        if snapshot_id is None:
            return False, "recorded; no snapshot to learn from"

        snapshot = self._snapshots.load(snapshot_id)
        if snapshot is None:
            return False, "recorded; snapshot no longer available"

        faces = self._analyzer.analyse(snapshot)
        if len(faces) != 1:
            # Zero faces means nothing to learn; several means no way to tell which
            # one the correction was about.
            return False, "recorded; no single clear face in the snapshot"

        face = faces[0]
        if not self._quality.accepts(face.quality):
            return False, "recorded; the face is too poor to learn from"

        self._enrollment.add_capture(identity_id, face.embedding)
        return True, f"recorded; added a real capture to {identity_id}"
