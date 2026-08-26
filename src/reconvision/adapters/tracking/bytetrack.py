"""Associating detections across frames with ByteTrack.

The stage that makes recognition both cheaper and more accurate: instead of
deciding who someone is on every frame and flickering between answers, the
pipeline accumulates evidence along a track and decides once.

ByteTrack lives in the `trackers` package rather than in supervision, where it
was deprecated in 0.28 and is removed in 0.31.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import structlog
import supervision as sv
from trackers import ByteTrackTracker

from reconvision.domain.models import BoundingBox, Detection, TrackedDetection

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrackingPolicy:
    """How forgiving the tracker is about losing sight of someone."""

    #: Frames a track survives without a matching detection. Someone walking
    #: behind a sofa should keep their identity when they come out the other
    #: side, rather than being reported as a second, unknown person.
    lost_track_buffer: int = 30
    #: Frames a track must persist before it is reported at all. Filters the
    #: single-frame phantoms that a detector produces on noise.
    minimum_consecutive_frames: int = 3
    track_activation_threshold: float = 0.25
    minimum_iou_threshold: float = 0.3
    #: Sampling rate the pipeline actually feeds the tracker, which is lower than
    #: the camera's. Passing the camera's rate would make the buffer expire
    #: several times too fast.
    frame_rate: float = 10.0


class ByteTrackAdapter:
    """Assigns stable ids to detections across frames."""

    def __init__(self, policy: TrackingPolicy | None = None) -> None:
        self._policy = policy or TrackingPolicy()
        self._tracker = ByteTrackTracker(
            lost_track_buffer=self._policy.lost_track_buffer,
            minimum_consecutive_frames=self._policy.minimum_consecutive_frames,
            track_activation_threshold=self._policy.track_activation_threshold,
            minimum_iou_threshold=self._policy.minimum_iou_threshold,
            frame_rate=self._policy.frame_rate,
        )
        self._labels_by_class_id: dict[int, str] = {}

    def update(self, detections: Sequence[Detection]) -> Sequence[TrackedDetection]:
        """Match this frame's detections against the tracks in flight."""
        if not detections:
            # ByteTrack still needs to age its tracks on empty frames, or someone
            # who steps out of view briefly never expires.
            self._tracker.update(sv.Detections.empty())
            return []

        tracked = self._tracker.update(self._to_supervision(detections))
        if tracked.tracker_id is None:
            return []

        return [
            TrackedDetection(
                track_id=int(track_id),
                detection=Detection(
                    box=BoundingBox(
                        left=float(box[0]),
                        top=float(box[1]),
                        right=float(box[2]),
                        bottom=float(box[3]),
                    ),
                    label=self._labels_by_class_id.get(int(class_id), "person"),
                    confidence=float(confidence),
                ),
            )
            for box, track_id, class_id, confidence in zip(
                tracked.xyxy,
                tracked.tracker_id,
                tracked.class_id
                if tracked.class_id is not None
                else np.zeros(len(tracked), dtype=int),
                tracked.confidence if tracked.confidence is not None else np.ones(len(tracked)),
                strict=True,
            )
            # ByteTrack reports -1 for a track it has seen but not yet confirmed.
            # Treating that as an id would pool every unconfirmed person into a
            # single track, and their recognition votes along with them.
            if track_id >= 0
        ]

    def _to_supervision(self, detections: Sequence[Detection]) -> sv.Detections:
        """Convert to the array form the tracker expects.

        Labels are strings in the domain and integers in the tracker, so the
        mapping is built as labels are encountered and kept for the reverse trip.
        """
        for detection in detections:
            if detection.label not in self._labels_by_class_id.values():
                self._labels_by_class_id[len(self._labels_by_class_id)] = detection.label

        class_ids = {label: index for index, label in self._labels_by_class_id.items()}

        return sv.Detections(
            xyxy=np.array(
                [[d.box.left, d.box.top, d.box.right, d.box.bottom] for d in detections],
                dtype=np.float32,
            ),
            confidence=np.array([d.confidence for d in detections], dtype=np.float32),
            class_id=np.array([class_ids[d.label] for d in detections], dtype=int),
        )
