"""Quality gating is the cheapest defence against confident wrong answers."""

from __future__ import annotations

import pytest

from reconvision.domain.models import FaceQuality
from reconvision.domain.quality import QualityPolicy, RejectionReason


def usable(**overrides: float) -> FaceQuality:
    """A face that passes every default requirement, minus whatever is overridden."""
    defaults: dict[str, float] = {
        "pixel_height": 140,
        "sharpness": 100.0,
        "yaw_degrees": 5.0,
        "detection_confidence": 0.95,
    }
    defaults.update(overrides)
    return FaceQuality(
        pixel_height=int(defaults["pixel_height"]),
        sharpness=defaults["sharpness"],
        yaw_degrees=defaults["yaw_degrees"],
        detection_confidence=defaults["detection_confidence"],
    )


def test_a_good_face_is_accepted() -> None:
    assert QualityPolicy().accepts(usable())


def test_a_distant_face_is_rejected() -> None:
    """The most common rejection on a wide-angle room camera."""
    assert QualityPolicy().rejection_reason(usable(pixel_height=40)) is RejectionReason.TOO_SMALL


def test_a_motion_blurred_face_is_rejected() -> None:
    assert QualityPolicy().rejection_reason(usable(sharpness=5.0)) is RejectionReason.TOO_BLURRY


@pytest.mark.parametrize("yaw", [70.0, -70.0])
def test_a_face_in_profile_is_rejected_whichever_way_it_turns(yaw: float) -> None:
    assert QualityPolicy().rejection_reason(usable(yaw_degrees=yaw)) is RejectionReason.TOO_ANGLED


def test_a_doubtful_detection_is_rejected() -> None:
    assert (
        QualityPolicy().rejection_reason(usable(detection_confidence=0.2))
        is RejectionReason.LOW_CONFIDENCE
    )


def test_thresholds_are_configurable_for_a_camera_that_sees_faces_small() -> None:
    small = usable(pixel_height=50)

    assert not QualityPolicy().accepts(small)
    assert QualityPolicy(min_pixel_height=45).accepts(small)


def test_frontality_falls_from_head_on_to_profile() -> None:
    assert usable(yaw_degrees=0.0).frontality == pytest.approx(1.0)
    assert usable(yaw_degrees=90.0).frontality == pytest.approx(0.0)
    assert usable(yaw_degrees=45.0).frontality == pytest.approx(0.5)


def test_a_better_face_carries_more_weight_than_a_worse_one() -> None:
    """Weight is what lets good frames outvote bad ones along a track."""
    close_and_sharp = usable(pixel_height=200, sharpness=200.0, yaw_degrees=0.0)
    far_and_blurry = usable(pixel_height=85, sharpness=45.0, yaw_degrees=40.0)

    assert close_and_sharp.weight() > far_and_blurry.weight()


def test_weight_stays_within_zero_and_one() -> None:
    """It scales votes, so a single enormous face must not dominate without bound."""
    enormous = usable(pixel_height=2000, sharpness=9000.0, yaw_degrees=0.0)

    assert 0.0 <= enormous.weight() <= 1.0
