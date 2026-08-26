"""Tracking is what lets identity be decided once per passage instead of per frame."""

from __future__ import annotations

from collections.abc import Sequence

from reconvision.adapters.tracking.bytetrack import ByteTrackAdapter, TrackingPolicy
from reconvision.domain.models import BoundingBox, Detection, TrackedDetection
from reconvision.domain.ports import Tracker

QUICK_TO_CONFIRM = TrackingPolicy(minimum_consecutive_frames=2)


def person_at(x: float, label: str = "person") -> Detection:
    return Detection(
        box=BoundingBox(left=x, top=100, right=x + 80, bottom=400),
        label=label,
        confidence=0.9,
    )


def walk(tracker: ByteTrackAdapter, steps: Sequence[int], stride: int = 25) -> list[int]:
    """Feed a subject moving steadily and collect the ids reported."""
    seen: list[int] = []
    for step in steps:
        seen.extend(t.track_id for t in tracker.update([person_at(50 + step * stride)]))
    return seen


def test_the_adapter_satisfies_the_tracker_port() -> None:
    tracker: Tracker = ByteTrackAdapter()

    assert isinstance(tracker, Tracker)


def test_one_person_crossing_the_room_keeps_one_identity() -> None:
    """The whole point. Without this, one passage becomes hundreds of events."""
    ids = walk(ByteTrackAdapter(QUICK_TO_CONFIRM), range(20))

    assert len(set(ids)) == 1


def test_two_people_at_once_get_separate_identities() -> None:
    """If their tracks merged, their recognition votes would merge too."""
    tracker = ByteTrackAdapter(QUICK_TO_CONFIRM)
    seen: list[int] = []
    for step in range(15):
        approaching, leaving = 40 + step * 20, 600 - step * 20
        seen.extend(
            t.track_id for t in tracker.update([person_at(approaching), person_at(leaving)])
        )

    assert len(set(seen)) == 2


def test_an_unconfirmed_track_is_never_reported() -> None:
    """ByteTrack reports -1 until a track is confirmed. Emitting that would pool
    every unconfirmed person into one shared track, votes included."""
    ids = walk(ByteTrackAdapter(TrackingPolicy(minimum_consecutive_frames=4)), range(10))

    assert all(track_id >= 0 for track_id in ids)


def test_identity_survives_a_brief_occlusion() -> None:
    """Someone walking behind a sofa should come out the other side as themselves,
    not as a second, unknown person."""
    tracker = ByteTrackAdapter(TrackingPolicy(minimum_consecutive_frames=2, lost_track_buffer=30))
    seen: list[int] = []
    for step in range(24):
        hidden = 10 <= step <= 13
        detections = [] if hidden else [person_at(50 + step * 25)]
        seen.extend(t.track_id for t in tracker.update(detections))

    assert len(set(seen)) == 1


def test_someone_returning_much_later_is_a_new_passage() -> None:
    """Otherwise a morning and an evening visit would be one endless event."""
    tracker = ByteTrackAdapter(
        TrackingPolicy(minimum_consecutive_frames=2, lost_track_buffer=5, frame_rate=10)
    )
    seen = walk(tracker, range(10))
    for _ in range(40):
        tracker.update([])
    seen += walk(tracker, range(10))

    assert len(set(seen)) == 2


def test_empty_frames_still_age_the_tracks() -> None:
    """A tracker that is not told about empty frames never expires anyone."""
    tracker = ByteTrackAdapter(QUICK_TO_CONFIRM)

    assert tracker.update([]) == []


def test_labels_survive_the_round_trip() -> None:
    """The tracker works in integers; the pipeline needs to know it is still a cat."""
    tracker = ByteTrackAdapter(QUICK_TO_CONFIRM)
    tracked: Sequence[TrackedDetection] = []
    for step in range(6):
        tracked = tracker.update([person_at(50 + step * 15, label="cat")])

    assert tracked
    assert all(t.detection.label == "cat" for t in tracked)
    assert all(t.detection.is_animal for t in tracked)


def test_a_stricter_confirmation_threshold_reports_fewer_frames() -> None:
    """The trade-off the setting exists for: a higher threshold filters the
    single-frame phantoms a detector produces on noise, at the cost of taking
    longer to acknowledge someone who really is there."""

    def reported_with(minimum: int) -> int:
        policy = TrackingPolicy(minimum_consecutive_frames=minimum)
        return len(walk(ByteTrackAdapter(policy), range(12)))

    assert reported_with(1) > reported_with(5)
