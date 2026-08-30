"""Deciding whether a detected face is worth recognising at all.

Home cameras produce a majority of faces that are too small, too blurry or too
far off-axis to identify. Feeding those to the matcher does not produce a wrong
answer occasionally; it produces confident wrong answers systematically, because
a degraded embedding drifts toward the centre of the space where it sits near
everybody. Declining to answer is the correct behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from reconvision.domain.models import FaceQuality


class RejectionReason(StrEnum):
    """Why a face was not submitted for recognition. Reported as a metric so the
    camera placement can be corrected rather than guessed at."""

    TOO_SMALL = "too_small"
    TOO_BLURRY = "too_blurry"
    TOO_ANGLED = "too_angled"
    LOW_CONFIDENCE = "low_confidence"


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Minimum requirements a face must meet before it reaches the matcher.

    The defaults are deliberate starting points, not tuned constants: run
    `reconvision eval` against captures from your own cameras and adjust.
    """

    #: ArcFace is trained on 112x112 crops. Below roughly 80 pixels of face height
    #: the crop is upscaled from too little detail and the embedding degrades.
    min_pixel_height: int = 80
    #: Variance of the Laplacian. Scene-dependent, so treat this as a floor to tune.
    min_sharpness: float = 40.0
    #: Beyond about 45 degrees of yaw, half the face is occluded by the other half.
    max_yaw_degrees: float = 45.0
    min_detection_confidence: float = 0.6

    def rejection_reason(self, quality: FaceQuality) -> RejectionReason | None:
        """The first unmet requirement, or None when the face is usable."""
        if quality.pixel_height < self.min_pixel_height:
            return RejectionReason.TOO_SMALL
        if quality.sharpness < self.min_sharpness:
            return RejectionReason.TOO_BLURRY
        if abs(quality.yaw_degrees) > self.max_yaw_degrees:
            return RejectionReason.TOO_ANGLED
        if quality.detection_confidence < self.min_detection_confidence:
            return RejectionReason.LOW_CONFIDENCE
        return None

    def accepts(self, quality: FaceQuality) -> bool:
        return self.rejection_reason(quality) is None
