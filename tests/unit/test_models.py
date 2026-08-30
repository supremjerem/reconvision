"""Geometry and classification primitives the pipeline leans on every frame."""

from __future__ import annotations

import pytest

from reconvision.domain.models import BoundingBox, Detection, SubjectKind


def box(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def test_a_box_reports_its_extent() -> None:
    person = box(10, 20, 110, 220)

    assert person.width == 100
    assert person.height == 200
    assert person.area == 20_000


def test_an_inverted_box_is_refused() -> None:
    """Detector output is external input and is validated at the boundary."""
    with pytest.raises(ValueError, match="negative extent"):
        box(100, 0, 10, 50)


def test_identical_boxes_fully_overlap() -> None:
    assert box(0, 0, 10, 10).intersection_over_union(box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_disjoint_boxes_do_not_overlap() -> None:
    assert box(0, 0, 10, 10).intersection_over_union(box(50, 50, 60, 60)) == 0.0


def test_partial_overlap_is_measured() -> None:
    """Two 100-unit squares sharing a 50x100 strip: 5000 shared of 15000 covered."""
    assert box(0, 0, 100, 100).intersection_over_union(box(50, 0, 150, 100)) == pytest.approx(1 / 3)


def test_a_face_is_paired_with_the_person_containing_it() -> None:
    """The pairing the pipeline actually needs. IoU is near zero here because the
    face is a fraction of the body, so containment is the right test."""
    person = box(100, 100, 300, 700)
    face = box(170, 130, 230, 200)

    assert person.contains_centre_of(face)
    assert person.intersection_over_union(face) < 0.05


def test_a_face_outside_a_person_is_not_paired_with_them() -> None:
    person = box(100, 100, 300, 700)
    someone_elses_face = box(500, 130, 560, 200)

    assert not person.contains_centre_of(someone_elses_face)


def test_a_person_detection_continues_to_the_face_stage() -> None:
    detection = Detection(box=box(0, 0, 10, 10), label="person", confidence=0.9)

    assert detection.is_person
    assert detection.kind is SubjectKind.PERSON


@pytest.mark.parametrize("label", ["cat", "dog", "bird"])
def test_an_animal_detection_stops_before_the_face_stage(label: str) -> None:
    """The saving that makes several streams affordable: no face model for pets."""
    detection = Detection(box=box(0, 0, 10, 10), label=label, confidence=0.9)

    assert detection.is_animal
    assert not detection.is_person


def test_an_irrelevant_label_is_neither_person_nor_animal() -> None:
    """COCO detects 80 classes. A sofa is not an event."""
    detection = Detection(box=box(0, 0, 10, 10), label="couch", confidence=0.9)

    assert detection.kind is None
    assert not detection.is_person
    assert not detection.is_animal
