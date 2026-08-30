"""Turning photographs of a person into a gallery the matcher can search.

Enrolment quality decides everything downstream. A gallery containing one crop of
somebody else silently poisons every comparison, and nothing in the system will
report an error - it will simply start recognising the wrong person. So this
module reports exactly what it found in each photograph and lets a human confirm
it, rather than quietly doing its best.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

from reconvision.adapters.images import read_image
from reconvision.domain.models import (
    Embedding,
    Face,
    GalleryEntry,
    GalleryEntrySource,
    Identity,
)
from reconvision.domain.ports import Clock, FaceAnalyzer, GalleryRepository
from reconvision.domain.quality import QualityPolicy, RejectionReason

logger = structlog.get_logger(__name__)

#: Image types worth trying to read from an enrolment folder.
PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

#: Below this, a gallery is too thin to survive a change of lighting or angle.
RECOMMENDED_MINIMUM_PHOTOS = 10


class EnrollmentOutcome(str):
    """Why a photograph did or did not contribute to the gallery."""

    __slots__ = ()


ACCEPTED = EnrollmentOutcome("accepted")
NO_FACE = EnrollmentOutcome("no_face_found")
SEVERAL_FACES = EnrollmentOutcome("several_faces")
UNREADABLE = EnrollmentOutcome("unreadable")


@dataclass(frozen=True, slots=True)
class PhotoResult:
    """What happened to one enrolment photograph."""

    path: Path
    outcome: str
    face: Face | None = None
    rejection: RejectionReason | None = None

    @property
    def accepted(self) -> bool:
        return self.outcome == ACCEPTED

    def describe(self) -> str:
        """A single line explaining the outcome, for the CLI and the web screen."""
        if self.accepted and self.face is not None:
            quality = self.face.quality
            return (
                f"accepted  ({quality.pixel_height}px, "
                f"yaw {quality.yaw_degrees:+.0f}°, weight {quality.weight():.2f})"
            )
        if self.rejection is not None:
            return f"skipped   (face {self.rejection.value.replace('_', ' ')})"
        return f"skipped   ({self.outcome.replace('_', ' ')})"


@dataclass(frozen=True, slots=True)
class EnrollmentReport:
    """The result of enrolling one identity from a folder of photographs."""

    identity: Identity
    results: Sequence[PhotoResult]

    @property
    def accepted(self) -> Sequence[PhotoResult]:
        return [result for result in self.results if result.accepted]

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def is_thin(self) -> bool:
        """Whether the gallery is too small to be reliable.

        A handful of photographs taken in one sitting all share a lighting and an
        angle, so they describe that sitting rather than the person. The system
        then works in the hallway at noon and fails in the hallway at night.
        """
        return self.accepted_count < RECOMMENDED_MINIMUM_PHOTOS

    def warnings(self) -> list[str]:
        """Problems worth telling a human about before they trust the result."""
        problems: list[str] = []
        if self.accepted_count == 0:
            problems.append("No usable face was found; nothing was enrolled.")
        elif self.is_thin:
            problems.append(
                f"Only {self.accepted_count} photo(s) enrolled. "
                f"{RECOMMENDED_MINIMUM_PHOTOS} or more, across different lighting and "
                f"angles, make recognition far more reliable."
            )

        several = sum(1 for result in self.results if result.outcome == SEVERAL_FACES)
        if several:
            problems.append(
                f"{several} photo(s) contained more than one face and were skipped. "
                f"Enrolling the wrong face silently corrupts every later comparison, "
                f"so crop them to one person and try again."
            )
        return problems


def find_photos(folder: Path) -> list[Path]:
    """Every readable image in a folder, in a stable order."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in PHOTO_SUFFIXES
    )


class EnrollmentService:
    """Builds a gallery from photographs, reporting what it did with each one."""

    def __init__(
        self,
        analyzer: FaceAnalyzer,
        gallery: GalleryRepository,
        clock: Clock,
        quality: QualityPolicy | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._gallery = gallery
        self._clock = clock
        self._quality = quality or QualityPolicy()

    def inspect(self, photos: Sequence[Path]) -> Iterator[PhotoResult]:
        """Report what each photograph contains, without storing anything.

        Separated from enrolling so a user can review the crops first. This is the
        step that catches a sibling in the background before they become you.
        """
        for path in photos:
            yield self._inspect_one(path)

    def enroll(
        self,
        identity: Identity,
        photos: Sequence[Path],
        results: Sequence[PhotoResult] | None = None,
    ) -> EnrollmentReport:
        """Store the usable faces from these photographs against an identity.

        Accepts an already-reviewed set of results, so the web screen can drop a
        bad crop before anything is written rather than after.
        """
        reviewed = list(results) if results is not None else list(self.inspect(photos))
        now = self._clock.now()

        self._gallery.add_identity(identity)
        for result in reviewed:
            if result.accepted and result.face is not None:
                self._gallery.add_entry(
                    GalleryEntry(
                        identity_id=identity.identity_id,
                        embedding=result.face.embedding,
                        source=GalleryEntrySource.ENROLLED_PHOTO,
                        captured_at=now,
                    )
                )

        report = EnrollmentReport(identity=identity, results=reviewed)
        logger.info(
            "enrolled",
            identity=identity.identity_id,
            photos=len(reviewed),
            accepted=report.accepted_count,
        )
        return report

    def add_capture(
        self,
        identity_id: str,
        embedding: Embedding,
        captured_at: datetime | None = None,
    ) -> None:
        """Add an embedding captured by a camera, from a user correction.

        Worth more than an enrolment photograph: it is the actual camera, the
        actual angle, the actual light, which is what the matcher will face.
        """
        self._gallery.add_entry(
            GalleryEntry(
                identity_id=identity_id,
                embedding=embedding,
                source=GalleryEntrySource.CORRECTED_EVENT,
                captured_at=captured_at or self._clock.now(),
            )
        )

    def _inspect_one(self, path: Path) -> PhotoResult:
        image = read_image(path)
        if image is None:
            return PhotoResult(path=path, outcome=UNREADABLE)

        faces = self._analyzer.analyse(image)
        if not faces:
            return PhotoResult(path=path, outcome=NO_FACE)
        if len(faces) > 1:
            # Refusing is the whole point. Picking the largest face would be right
            # most of the time and catastrophic the rest, and the failure is
            # invisible: the gallery simply starts containing someone else.
            return PhotoResult(path=path, outcome=SEVERAL_FACES)

        face = faces[0]
        rejection = self._quality.rejection_reason(face.quality)
        if rejection is not None:
            return PhotoResult(path=path, outcome=rejection.value, rejection=rejection)

        return PhotoResult(path=path, outcome=ACCEPTED, face=face)
