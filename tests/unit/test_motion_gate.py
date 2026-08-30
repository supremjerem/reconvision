"""The motion gate removes the largest waste in the pipeline: analysing an empty room."""

from __future__ import annotations

import numpy as np
import pytest

from reconvision.adapters.video.motion import MotionGate, MotionPolicy
from tests.fixtures.video import frame_with_subject, noisy_frame, still_frame


def test_the_first_frame_always_counts_as_motion() -> None:
    """Nothing to compare against, so the safe answer is to look rather than skip."""
    assert MotionGate().has_motion(still_frame())


def test_an_unchanged_room_is_skipped() -> None:
    """The saving that makes several cameras affordable."""
    gate = MotionGate()
    gate.has_motion(still_frame())

    assert not gate.has_motion(still_frame())


def test_someone_walking_in_triggers_the_gate() -> None:
    gate = MotionGate()
    gate.has_motion(still_frame())

    assert gate.has_motion(frame_with_subject(x_position=40))


def test_sensor_grain_in_a_dark_room_does_not_trigger_the_gate() -> None:
    """Without the blur this is exactly what trips on every frame at night,
    and the whole saving disappears."""
    rng = np.random.default_rng(seed=20260826)
    gate = MotionGate()
    gate.has_motion(noisy_frame(rng))

    assert not any(gate.has_motion(noisy_frame(rng)) for _ in range(10))


def test_continued_movement_keeps_triggering() -> None:
    gate = MotionGate()
    gate.has_motion(still_frame())

    assert all(gate.has_motion(frame_with_subject(x)) for x in (20, 90, 160, 230))


def test_a_subject_that_stops_moving_stops_triggering() -> None:
    """Frame differencing reports change, not presence. Someone standing perfectly
    still is invisible to it, which is why detection runs on a schedule too."""
    gate = MotionGate()
    gate.has_motion(frame_with_subject(100))

    assert not gate.has_motion(frame_with_subject(100))


def test_resetting_makes_the_next_frame_count_again() -> None:
    """Used after a reconnection, where the previous frame is stale by minutes."""
    gate = MotionGate()
    gate.has_motion(still_frame())
    assert not gate.has_motion(still_frame())

    gate.reset()

    assert gate.has_motion(still_frame())


def test_a_change_of_aspect_ratio_counts_as_motion() -> None:
    """A camera renegotiating to a different shape leaves nothing comparable, so
    the gate opens rather than comparing frames that do not line up."""
    gate = MotionGate()
    gate.has_motion(still_frame())

    assert gate.has_motion(np.full((360, 640, 3), 40, dtype=np.uint8))


def test_the_same_scene_at_a_higher_resolution_is_not_mistaken_for_motion() -> None:
    """Both frames reduce to the same working size, so they stay comparable and a
    resolution change alone does not manufacture a false alarm."""
    gate = MotionGate()
    gate.has_motion(still_frame())

    assert not gate.has_motion(np.full((480, 640, 3), 40, dtype=np.uint8))


def test_sensitivity_is_configurable() -> None:
    strict = MotionGate(MotionPolicy(min_changed_ratio=0.9))
    strict.has_motion(still_frame())

    assert not strict.has_motion(frame_with_subject(x_position=40))


@pytest.mark.parametrize(
    ("threshold", "ratio"),
    [(0, 0.005), (256, 0.005), (25, -0.1), (25, 1.5)],
)
def test_an_impossible_motion_policy_is_refused(threshold: int, ratio: float) -> None:
    with pytest.raises(ValueError):
        MotionPolicy(pixel_delta_threshold=threshold, min_changed_ratio=ratio)
