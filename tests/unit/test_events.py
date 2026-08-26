"""Events are the pipeline's only output, and corrections are its feedback loop."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from reconvision.domain.events import (
    EventFeedback,
    EventVerdict,
    FeedbackLabel,
    RecognitionEvent,
)
from reconvision.domain.models import SubjectKind

START = datetime(2026, 8, 26, 3, 12, tzinfo=UTC)


def event(**overrides: Any) -> RecognitionEvent:
    defaults: dict[str, Any] = {
        "camera_name": "living_room",
        "verdict": EventVerdict.UNKNOWN_PERSON,
        "started_at": START,
        "ended_at": START + timedelta(seconds=4),
        "subject_kind": SubjectKind.PERSON,
    }
    defaults.update(overrides)
    return RecognitionEvent(**defaults)


def test_an_event_reports_how_long_the_subject_was_present() -> None:
    assert event().duration_seconds == pytest.approx(4.0)


def test_an_event_cannot_end_before_it_starts() -> None:
    with pytest.raises(ValueError, match="ends before it starts"):
        event(ended_at=START - timedelta(seconds=1))


def test_a_known_person_verdict_requires_an_identity() -> None:
    """Guards the inconsistency that would show 'recognised: nobody' in the UI."""
    with pytest.raises(ValueError, match="inconsistent"):
        event(verdict=EventVerdict.KNOWN_PERSON, identity_id=None)


def test_an_unknown_person_verdict_must_not_carry_an_identity() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        event(verdict=EventVerdict.UNKNOWN_PERSON, identity_id="jeremie")


def test_events_get_distinct_identifiers() -> None:
    assert event().event_id != event().event_id


def test_a_stranger_is_worth_a_notification() -> None:
    assert event(verdict=EventVerdict.UNKNOWN_PERSON).is_noteworthy


def test_a_person_whose_face_was_never_usable_is_worth_a_notification() -> None:
    """Someone was there and the system could not say who. That is the case a
    household most wants to hear about, so it must not be swallowed."""
    assert event(verdict=EventVerdict.UNIDENTIFIED).is_noteworthy


def test_a_recognised_household_member_is_not_worth_a_notification() -> None:
    """Otherwise the system cries wolf every time you walk to the kitchen."""
    assert not event(verdict=EventVerdict.KNOWN_PERSON, identity_id="jeremie").is_noteworthy


def test_the_cat_is_not_worth_a_notification() -> None:
    assert not event(
        verdict=EventVerdict.ANIMAL,
        subject_kind=SubjectKind.ANIMAL,
        animal_label="cat",
    ).is_noteworthy


def test_confirming_an_event_needs_no_identity() -> None:
    feedback = EventFeedback(event_id="e1", label=FeedbackLabel.CONFIRMED, submitted_at=START)

    assert feedback.corrected_identity_id is None


def test_correcting_an_identity_requires_saying_who_it_was() -> None:
    """'That was not me' is not actionable; 'that was me' adds a real capture to
    the gallery. The type refuses the useless half of the correction."""
    with pytest.raises(ValueError, match="requires the identity"):
        EventFeedback(event_id="e1", label=FeedbackLabel.WRONG_IDENTITY, submitted_at=START)


def test_a_non_person_cannot_be_corrected_to_a_person() -> None:
    with pytest.raises(ValueError, match="cannot be corrected"):
        EventFeedback(
            event_id="e1",
            label=FeedbackLabel.NOT_A_PERSON,
            submitted_at=START,
            corrected_identity_id="jeremie",
        )
