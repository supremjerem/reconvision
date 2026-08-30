"""What the pipeline emits once a tracked subject has left or been decided.

One event per passage, never one per frame. Everything downstream - storage,
notifications, the correction screen - consumes this type.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from reconvision.domain.models import SubjectKind


class EventVerdict(StrEnum):
    """The three answers the system exists to give, plus an honest fourth."""

    KNOWN_PERSON = "known_person"
    UNKNOWN_PERSON = "unknown_person"
    ANIMAL = "animal"
    #: A person was present but no usable face was ever captured. Reported rather
    #: than silently dropped: a camera producing these constantly is misaimed.
    UNIDENTIFIED = "unidentified"


class FeedbackLabel(StrEnum):
    """A correction supplied by the user from the review screen."""

    CONFIRMED = "confirmed"
    WRONG_IDENTITY = "wrong_identity"
    NOT_A_PERSON = "not_a_person"


@dataclass(frozen=True, slots=True)
class RecognitionEvent:
    """A single subject's passage in front of a single camera."""

    camera_name: str
    verdict: EventVerdict
    started_at: datetime
    ended_at: datetime
    subject_kind: SubjectKind
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    identity_id: str | None = None
    #: Share of the track's weight behind the verdict, not a probability.
    confidence: float = 0.0
    #: Best cosine similarity seen along the track, for tuning the threshold later.
    best_similarity: float = -1.0
    observations: int = 0
    snapshot_id: str | None = None
    animal_label: str | None = None

    def __post_init__(self) -> None:
        if self.ended_at < self.started_at:
            message = f"Event ends before it starts: {self.started_at} -> {self.ended_at}"
            raise ValueError(message)
        if (self.verdict is EventVerdict.KNOWN_PERSON) != (self.identity_id is not None):
            message = f"{self.verdict} is inconsistent with identity_id={self.identity_id!r}"
            raise ValueError(message)

    @property
    def duration_seconds(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def is_noteworthy(self) -> bool:
        """Whether this deserves a push notification.

        A recognised household member walking through their own home at midday is
        not news. An unknown person is. Callers may still apply their own rules on
        top - time of day, camera - but this is the default worth waking up for.
        """
        return self.verdict in (EventVerdict.UNKNOWN_PERSON, EventVerdict.UNIDENTIFIED)


@dataclass(frozen=True, slots=True)
class EventFeedback:
    """A user correction on a past event.

    Two jobs, both essential. It enriches the gallery with a capture taken by the
    real camera in real conditions, which is worth more than any enrolment selfie;
    and it builds a labelled set from your own footage so the threshold can be
    recalibrated against reality rather than against a public benchmark.
    """

    event_id: str
    label: FeedbackLabel
    submitted_at: datetime
    #: Who it actually was. Required when correcting to a known person.
    corrected_identity_id: str | None = None

    def __post_init__(self) -> None:
        needs_identity = self.label is FeedbackLabel.WRONG_IDENTITY
        if needs_identity and self.corrected_identity_id is None:
            message = "Correcting an identity requires the identity it should have been"
            raise ValueError(message)
        if self.label is FeedbackLabel.NOT_A_PERSON and self.corrected_identity_id is not None:
            message = "A non-person cannot be corrected to an identity"
            raise ValueError(message)
