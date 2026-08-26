"""Value objects shared across the recognition pipeline.

Everything here is immutable and free of I/O. Embeddings are plain NumPy arrays:
NumPy is a numeric primitive, not an infrastructure concern, so the domain is
allowed to depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

#: A decoded video frame in BGR order, as produced by every video adapter.
type Frame = NDArray[np.uint8]

#: A face descriptor. Always L2-normalised, so a dot product is a cosine similarity.
type Embedding = NDArray[np.float32]

#: Sentinel identity for a face that matched nobody in the gallery.
UNKNOWN_IDENTITY: str | None = None


class SubjectKind(StrEnum):
    """What the object detector believes it is looking at."""

    PERSON = "person"
    ANIMAL = "animal"


#: COCO labels the detector may emit, mapped to the only distinction that matters
#: downstream. A person continues to the face stage; an animal never does.
SUBJECT_KIND_BY_LABEL: dict[str, SubjectKind] = {
    "person": SubjectKind.PERSON,
    "cat": SubjectKind.ANIMAL,
    "dog": SubjectKind.ANIMAL,
    "bird": SubjectKind.ANIMAL,
    "horse": SubjectKind.ANIMAL,
    "sheep": SubjectKind.ANIMAL,
    "cow": SubjectKind.ANIMAL,
    "bear": SubjectKind.ANIMAL,
}


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned box in pixel coordinates, top-left origin."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if self.right < self.left or self.bottom < self.top:
            message = f"Box has negative extent: {self}"
            raise ValueError(message)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersection_over_union(self, other: BoundingBox) -> float:
        """Overlap ratio, used to associate a face with the person containing it."""
        overlap_width = max(0.0, min(self.right, other.right) - max(self.left, other.left))
        overlap_height = max(0.0, min(self.bottom, other.bottom) - max(self.top, other.top))
        intersection = overlap_width * overlap_height
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def contains_centre_of(self, other: BoundingBox) -> bool:
        """Whether another box is centred inside this one.

        More reliable than IoU for pairing a small face box with the much larger
        person box around it, where the IoU is low even when the pairing is obvious.
        """
        centre_x = (other.left + other.right) / 2
        centre_y = (other.top + other.bottom) / 2
        return self.left <= centre_x <= self.right and self.top <= centre_y <= self.bottom


@dataclass(frozen=True, slots=True)
class Detection:
    """An object located in a frame by the detector."""

    box: BoundingBox
    label: str
    confidence: float

    @property
    def kind(self) -> SubjectKind | None:
        """The coarse category, or None for a label the pipeline does not care about."""
        return SUBJECT_KIND_BY_LABEL.get(self.label)

    @property
    def is_person(self) -> bool:
        return self.kind is SubjectKind.PERSON

    @property
    def is_animal(self) -> bool:
        return self.kind is SubjectKind.ANIMAL


@dataclass(frozen=True, slots=True)
class FaceQuality:
    """How usable a detected face is for recognition.

    Recognising a blurry, distant or side-on face is the dominant source of false
    matches, because a degraded embedding drifts towards the centre of the space
    and lands near everything. Measuring quality lets the pipeline decline to
    answer instead of answering badly.
    """

    pixel_height: int
    sharpness: float
    yaw_degrees: float
    detection_confidence: float

    @property
    def frontality(self) -> float:
        """1.0 looking straight at the camera, decaying to 0.0 at full profile."""
        return max(0.0, 1.0 - abs(self.yaw_degrees) / 90.0)

    def weight(
        self, reference_pixel_height: int = 160, reference_sharpness: float = 120.0
    ) -> float:
        """A 0-1 score used to weight this observation in a track's vote.

        A large, sharp, frontal face should outvote a small, blurry, angled one
        rather than merely counting the same.
        """
        size = min(1.0, self.pixel_height / reference_pixel_height)
        sharpness = min(1.0, self.sharpness / reference_sharpness)
        return size * sharpness * self.frontality * self.detection_confidence


@dataclass(frozen=True, slots=True)
class Face:
    """A face located in a frame, with its descriptor and quality assessment."""

    box: BoundingBox
    embedding: Embedding = field(compare=False)
    quality: FaceQuality


@dataclass(frozen=True, slots=True)
class Identity:
    """A person the system has been taught to recognise."""

    identity_id: str
    display_name: str


class GalleryEntrySource(StrEnum):
    """Where a gallery embedding came from.

    Kept because the two are not equivalent: an enrolled photo is usually a
    well-lit portrait, while a correction is a real capture from the actual
    camera at the actual angle, which is far more valuable for matching.
    """

    ENROLLED_PHOTO = "enrolled_photo"
    CORRECTED_EVENT = "corrected_event"


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    """One enrolled embedding belonging to one identity."""

    identity_id: str
    embedding: Embedding = field(compare=False)
    source: GalleryEntrySource = GalleryEntrySource.ENROLLED_PHOTO
    captured_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The outcome of comparing one embedding against the gallery."""

    identity_id: str | None
    similarity: float
    runner_up_similarity: float
    runner_up_identity_id: str | None = None

    @property
    def is_match(self) -> bool:
        return self.identity_id is not None

    @property
    def margin(self) -> float:
        """Distance to the next best identity.

        A thin margin means two enrolled people scored almost the same, which is
        a weak answer even when the top score clears the threshold.
        """
        return self.similarity - self.runner_up_similarity


@dataclass(frozen=True, slots=True)
class TrackVerdict:
    """The decision for a whole tracked path, rather than for a single frame."""

    identity_id: str | None
    confidence: float
    observations: int
    is_conclusive: bool

    @property
    def is_known_person(self) -> bool:
        return self.is_conclusive and self.identity_id is not None
