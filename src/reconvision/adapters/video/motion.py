"""The cheapest stage: deciding whether anything moved at all.

A home camera watches an empty room most of the time. Running an object detector
on those frames is the single largest waste in a naive pipeline, and this gate
removes it for a few hundred microseconds per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from reconvision.domain.models import Frame


@dataclass(frozen=True, slots=True)
class MotionPolicy:
    """How much change counts as motion."""

    #: Per-pixel intensity change that counts as changed, on a 0-255 scale. Low
    #: enough to catch a person in dim light, high enough to ignore sensor noise.
    pixel_delta_threshold: int = 25
    #: Share of the frame that must change. A person entering a room moves well
    #: over 0.5% of the pixels; compression noise and a flickering bulb do not.
    min_changed_ratio: float = 0.005
    #: Frames are downscaled to this width before comparison. Motion is a
    #: coarse question, and 320px answers it roughly sixteen times cheaper.
    working_width: int = 320

    def __post_init__(self) -> None:
        if not 0 < self.pixel_delta_threshold < 256:
            message = f"Pixel threshold must be in (0, 256), got {self.pixel_delta_threshold}"
            raise ValueError(message)
        if not 0.0 <= self.min_changed_ratio <= 1.0:
            message = f"Changed ratio must be in [0, 1], got {self.min_changed_ratio}"
            raise ValueError(message)


class MotionGate:
    """Reports whether a frame differs meaningfully from the previous one.

    Deliberately a frame-to-frame difference rather than a background subtractor:
    a learned background model spends its first minutes wrong after every restart
    and drifts when the light changes, which on a home camera means either missed
    people or a stream of false alarms at dusk.
    """

    def __init__(self, policy: MotionPolicy | None = None) -> None:
        self._policy = policy or MotionPolicy()
        self._previous: Frame | None = None

    def reset(self) -> None:
        """Forget the reference frame, after a reconnection for instance."""
        self._previous = None

    def has_motion(self, frame: Frame) -> bool:
        """Whether this frame differs enough from the last to be worth analysing.

        The first frame after a reset always counts as motion: with nothing to
        compare against, the safe answer is to look rather than to skip.
        """
        current = self._prepare(frame)
        previous, self._previous = self._previous, current

        if previous is None or previous.shape != current.shape:
            return True

        difference = cv2.absdiff(previous, current)
        _, changed = cv2.threshold(
            difference, self._policy.pixel_delta_threshold, 255, cv2.THRESH_BINARY
        )
        changed_ratio = float(np.count_nonzero(changed)) / changed.size

        return changed_ratio >= self._policy.min_changed_ratio

    def _prepare(self, frame: Frame) -> Frame:
        """Downscale, desaturate and blur, so only real movement survives.

        The blur is what separates a person from sensor noise: without it, grain
        in a dark room trips the gate on every frame and the saving disappears.
        """
        height, width = frame.shape[:2]
        working = frame
        if width > self._policy.working_width:
            scale = self._policy.working_width / width
            working = cast(
                "Frame",
                cv2.resize(frame, (self._policy.working_width, max(1, int(height * scale)))),
            )

        grey = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY) if working.ndim == 3 else working
        return cast(
            "np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", cv2.GaussianBlur(grey, (21, 21), 0)
        )
