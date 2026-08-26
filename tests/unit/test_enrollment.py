"""Enrolment quality decides everything downstream.

A gallery containing one crop of somebody else corrupts every later comparison,
and nothing reports an error - the system simply starts naming the wrong person.
These tests are about refusing to guess.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from reconvision.application.enrollment import (
    RECOMMENDED_MINIMUM_PHOTOS,
    EnrollmentService,
    find_photos,
)
from reconvision.domain.models import (
    BoundingBox,
    Embedding,
    Face,
    FaceQuality,
    GalleryEntrySource,
    Identity,
)
from tests.fakes import FakeClock, InMemoryGalleryRepository, ScriptedFaceAnalyzer
from tests.unit.conftest import random_embedding

JEREMIE = Identity(identity_id="jeremie", display_name="Jeremie")


def usable_face(embedding: Embedding) -> Face:
    return Face(
        box=BoundingBox(left=10, top=10, right=140, bottom=170),
        embedding=embedding,
        quality=FaceQuality(
            pixel_height=160, sharpness=180.0, yaw_degrees=4.0, detection_confidence=0.97
        ),
    )


def distant_face(embedding: Embedding) -> Face:
    return Face(
        box=BoundingBox(left=10, top=10, right=50, bottom=55),
        embedding=embedding,
        quality=FaceQuality(
            pixel_height=45, sharpness=10.0, yaw_degrees=10.0, detection_confidence=0.8
        ),
    )


@pytest.fixture
def photos(tmp_path: Path) -> list[Path]:
    """A folder of real image files, so path handling is genuinely exercised."""
    folder = tmp_path / "jeremie"
    folder.mkdir()
    paths = []
    for index in range(3):
        path = folder / f"photo_{index}.jpg"
        cv2.imwrite(str(path), np.full((200, 200, 3), 120, dtype=np.uint8))
        paths.append(path)
    return paths


def service(faces: list[Face], gallery: InMemoryGalleryRepository) -> EnrollmentService:
    return EnrollmentService(
        analyzer=ScriptedFaceAnalyzer(faces),
        gallery=gallery,
        clock=FakeClock(),
    )


def test_photos_are_found_in_a_folder(photos: list[Path]) -> None:
    assert find_photos(photos[0].parent) == photos


def test_non_images_in_the_folder_are_ignored(photos: list[Path]) -> None:
    (photos[0].parent / "notes.txt").write_text("not a photo")

    assert find_photos(photos[0].parent) == photos


def test_a_usable_photo_is_enrolled(photos: list[Path], rng: np.random.Generator) -> None:
    gallery = InMemoryGalleryRepository()

    report = service([usable_face(random_embedding(rng))], gallery).enroll(JEREMIE, photos)

    assert report.accepted_count == 3
    assert len(gallery.load_entries()) == 3
    assert gallery.list_identities() == [JEREMIE]


def test_enrolled_entries_are_marked_as_coming_from_photographs(
    photos: list[Path], rng: np.random.Generator
) -> None:
    """Distinguished from corrections so a later review can tell a posed portrait
    from a real capture off the camera."""
    gallery = InMemoryGalleryRepository()
    service([usable_face(random_embedding(rng))], gallery).enroll(JEREMIE, photos)

    assert all(
        entry.source is GalleryEntrySource.ENROLLED_PHOTO for entry in gallery.load_entries()
    )


def test_a_photo_with_two_faces_is_refused(photos: list[Path], rng: np.random.Generator) -> None:
    """The failure this whole module exists to prevent. Picking the larger face
    would be right most of the time and silently catastrophic the rest: a sibling
    in the background becomes you, and nothing ever reports an error."""
    gallery = InMemoryGalleryRepository()
    two_faces = [usable_face(random_embedding(rng)), usable_face(random_embedding(rng))]

    report = service(two_faces, gallery).enroll(JEREMIE, photos)

    assert report.accepted_count == 0
    assert gallery.load_entries() == []
    assert any("more than one face" in warning for warning in report.warnings())


def test_a_photo_with_no_face_is_skipped(photos: list[Path], rng: np.random.Generator) -> None:
    report = service([], InMemoryGalleryRepository()).enroll(JEREMIE, photos)

    assert report.accepted_count == 0
    assert all(not result.accepted for result in report.results)


def test_a_photo_too_distant_to_identify_is_skipped(
    photos: list[Path], rng: np.random.Generator
) -> None:
    """Enrolling a 45-pixel face teaches the gallery a blur, which then sits near
    everybody in the embedding space."""
    report = service([distant_face(random_embedding(rng))], InMemoryGalleryRepository()).enroll(
        JEREMIE, photos
    )

    assert report.accepted_count == 0
    assert report.results[0].rejection is not None


def test_an_unreadable_file_is_reported_not_raised(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    """A folder of photographs routinely contains a stray file. That is not a
    reason to abandon the other nineteen."""
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "broken.jpg").write_bytes(b"not an image")

    results = list(
        service([usable_face(random_embedding(rng))], InMemoryGalleryRepository()).inspect(
            find_photos(folder)
        )
    )

    assert len(results) == 1
    assert not results[0].accepted


def test_a_thin_gallery_is_flagged(photos: list[Path], rng: np.random.Generator) -> None:
    """Three photographs from one sitting share a light and an angle, so they
    describe that sitting rather than the person: it works in the hallway at noon
    and fails there at night."""
    report = service([usable_face(random_embedding(rng))], InMemoryGalleryRepository()).enroll(
        JEREMIE, photos
    )

    assert report.is_thin
    assert any(str(RECOMMENDED_MINIMUM_PHOTOS) in warning for warning in report.warnings())


def test_inspecting_stores_nothing(photos: list[Path], rng: np.random.Generator) -> None:
    """The dry run behind the review screen: see the crops before committing them."""
    gallery = InMemoryGalleryRepository()

    results = list(service([usable_face(random_embedding(rng))], gallery).inspect(photos))

    assert len(results) == 3
    assert gallery.load_entries() == []


def test_a_reviewed_selection_can_drop_a_bad_crop(
    photos: list[Path], rng: np.random.Generator
) -> None:
    """What the web screen does: the user deletes the photo containing the wrong
    face, and only the rest is stored."""
    gallery = InMemoryGalleryRepository()
    enroller = service([usable_face(random_embedding(rng))], gallery)
    reviewed = list(enroller.inspect(photos))

    enroller.enroll(JEREMIE, photos, results=reviewed[:1])

    assert len(gallery.load_entries()) == 1


def test_a_correction_is_recorded_as_a_real_capture(rng: np.random.Generator) -> None:
    """Worth more than an enrolment photograph: the actual camera, angle and light
    the matcher will face."""
    gallery = InMemoryGalleryRepository()

    service([], gallery).add_capture("jeremie", random_embedding(rng))

    (entry,) = gallery.load_entries()
    assert entry.source is GalleryEntrySource.CORRECTED_EVENT
    assert entry.captured_at is not None


def test_a_photo_result_explains_itself(rng: np.random.Generator) -> None:
    """The line printed per photo, and later shown beside each crop on the screen."""
    from reconvision.application.enrollment import ACCEPTED, PhotoResult

    described = PhotoResult(
        path=Path("a.jpg"), outcome=ACCEPTED, face=usable_face(random_embedding(rng))
    ).describe()

    assert "accepted" in described
    assert "160px" in described
