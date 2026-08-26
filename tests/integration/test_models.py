"""The detector and face analyzer, exercised against the real exported weights.

Marked `models` and skipped automatically when the weights are absent. Real
photographs rather than synthetic shapes, because a convolutional network's
behaviour on a drawn rectangle says nothing about its behaviour on a person.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

from reconvision.adapters.detection.onnx_yolo import (
    DetectorPolicy,
    OnnxYoloDetector,
    select_providers,
)
from reconvision.adapters.faces.insightface_analyzer import InsightFaceAnalyzer
from reconvision.domain.matching import cosine_similarity
from reconvision.domain.models import Frame
from reconvision.domain.ports import FaceAnalyzer, ObjectDetector
from reconvision.domain.quality import QualityPolicy
from tests.fixtures.samples import sample_path

pytestmark = [pytest.mark.integration, pytest.mark.models]


@pytest.fixture(scope="module")
def load(samples_dir: Path) -> Callable[[str], Frame]:
    """Load a sample photograph, skipping the test if it cannot be obtained."""

    def loader(name: str) -> Frame:
        path = sample_path(name, samples_dir)
        if path is None:
            pytest.skip(f"sample {name} could not be fetched")
        frame = cv2.imread(str(path))
        if frame is None:
            pytest.skip(f"sample {name} could not be decoded")
        return cast("Frame", frame)

    return loader


@pytest.fixture(scope="module")
def detector(detector_path: Path) -> OnnxYoloDetector:
    return OnnxYoloDetector(detector_path)


@pytest.fixture(scope="module")
def analyzer(models_dir: Path) -> InsightFaceAnalyzer:
    return InsightFaceAnalyzer(models_dir)


def test_the_detector_satisfies_its_port(detector: OnnxYoloDetector) -> None:
    port: ObjectDetector = detector

    assert isinstance(port, ObjectDetector)


def test_the_analyzer_satisfies_its_port(analyzer: InsightFaceAnalyzer) -> None:
    port: FaceAnalyzer = analyzer

    assert isinstance(port, FaceAnalyzer)


def test_people_are_found_in_a_photograph(
    detector: OnnxYoloDetector, load: Callable[[str], Frame]
) -> None:
    detections = detector.detect(load("people.jpg"))

    assert [d for d in detections if d.is_person]
    assert all(0.0 < d.confidence <= 1.0 for d in detections)


def test_boxes_stay_inside_the_frame(
    detector: OnnxYoloDetector, load: Callable[[str], Frame]
) -> None:
    """Letterbox padding is undone by arithmetic; a sign error there puts boxes
    off-image and every downstream crop comes back empty."""
    frame = load("people.jpg")
    height, width = frame.shape[:2]

    for detection in detector.detect(frame):
        assert 0 <= detection.box.left <= detection.box.right <= width
        assert 0 <= detection.box.top <= detection.box.bottom <= height


def test_a_dog_is_classified_as_an_animal_not_a_person(
    detector: OnnxYoloDetector, load: Callable[[str], Frame]
) -> None:
    """Half of what this system is for, and the branch that keeps it affordable:
    an animal never reaches the face model."""
    detections = detector.detect(load("dog.jpg"))

    assert any(d.is_animal for d in detections)
    assert not any(d.is_person for d in detections)


def test_furniture_and_vehicles_are_ignored(
    detector: OnnxYoloDetector, load: Callable[[str], Frame]
) -> None:
    """The photograph contains a bus. COCO can label 80 classes; a bus is not an
    event and must not reach the pipeline."""
    labels = {d.label for d in detector.detect(load("people.jpg"))}

    assert labels <= {"person", "cat", "dog", "bird", "horse", "sheep", "cow", "bear"}


def test_a_higher_confidence_threshold_yields_fewer_detections(
    detector_path: Path, load: Callable[[str], Frame]
) -> None:
    frame = load("people.jpg")
    permissive = OnnxYoloDetector(detector_path, DetectorPolicy(confidence_threshold=0.2))
    strict = OnnxYoloDetector(detector_path, DetectorPolicy(confidence_threshold=0.9))

    assert len(permissive.detect(frame)) >= len(strict.detect(frame))


def test_an_empty_frame_produces_no_detections(detector: OnnxYoloDetector) -> None:
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_faces_are_found_and_embedded(
    analyzer: InsightFaceAnalyzer, load: Callable[[str], Frame]
) -> None:
    faces = analyzer.analyse(load("zidane.jpg"))

    assert faces
    for face in faces:
        assert face.embedding.shape == (512,)
        assert face.embedding.dtype == np.float32


def test_two_captures_of_the_same_face_are_more_similar_than_two_people(
    analyzer: InsightFaceAnalyzer, load: Callable[[str], Frame]
) -> None:
    """The property the entire system rests on. If a re-encode of one face were
    not closer to itself than to a different person, no threshold could work."""
    frame = load("zidane.jpg")
    faces = analyzer.analyse(frame)
    if len(faces) < 2:
        pytest.skip("sample does not contain two faces")

    # Re-encoding at lower quality stands in for a second capture of the same
    # face: same person, different compression artefacts.
    _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        pytest.skip("re-encoded frame could not be decoded")

    recaptured = analyzer.analyse(cast("Frame", decoded))
    if not recaptured:
        pytest.skip("re-encoded frame yielded no faces")

    same_person = max(cosine_similarity(faces[0].embedding, f.embedding) for f in recaptured)
    different_people = cosine_similarity(faces[0].embedding, faces[1].embedding)

    assert same_person > different_people


def test_restricting_to_a_person_box_finds_that_persons_face(
    detector: OnnxYoloDetector, analyzer: InsightFaceAnalyzer, load: Callable[[str], Frame]
) -> None:
    """The normal path: the detector has already said where the person is, so the
    face stage does not rescan the whole frame."""
    frame = load("zidane.jpg")
    person = next(d for d in detector.detect(frame) if d.is_person)

    faces = analyzer.analyse(frame, person.box)

    assert faces
    for face in faces:
        assert person.box.contains_centre_of(face.box)


def test_a_distant_face_is_declined_by_the_quality_gate(
    detector: OnnxYoloDetector, analyzer: InsightFaceAnalyzer, load: Callable[[str], Frame]
) -> None:
    """A wide-angle shot where people are visible but their faces are 50 pixels
    tall. Recognising these is how a system produces confident nonsense."""
    frame = load("people.jpg")
    policy = QualityPolicy()

    qualities = [
        face.quality
        for person in detector.detect(frame)
        if person.is_person
        for face in analyzer.analyse(frame, person.box)
    ]
    if not qualities:
        pytest.skip("no faces detected in the wide shot")

    assert all(not policy.accepts(quality) for quality in qualities)


def test_a_face_in_profile_is_declined(
    analyzer: InsightFaceAnalyzer, load: Callable[[str], Frame]
) -> None:
    faces = analyzer.analyse(load("zidane.jpg"))
    policy = QualityPolicy()
    angled = [f for f in faces if abs(f.quality.yaw_degrees) > policy.max_yaw_degrees]
    if not angled:
        pytest.skip("sample contains no strongly angled face")

    assert all(not policy.accepts(face.quality) for face in angled)


def test_a_provider_is_always_available() -> None:
    """CoreML on the developer's Mac, plain CPU in the container. The same
    exported model runs on both, which is why ONNX was chosen at all."""
    providers = select_providers()

    assert providers[-1] == "CPUExecutionProvider"


def test_a_bystanders_face_is_not_attributed_to_the_person_in_front(
    detector: OnnxYoloDetector, analyzer: InsightFaceAnalyzer, load: Callable[[str], Frame]
) -> None:
    """Two overlapping people. The crop around one is widened to give the aligner
    context, which lets the other's face into the image; attributing it to the
    wrong track would put someone else's identity on this passage."""
    frame = load("zidane.jpg")
    people = [d for d in detector.detect(frame) if d.is_person]
    if len(people) < 2:
        pytest.skip("sample does not contain two overlapping people")

    for person in people:
        for face in analyzer.analyse(frame, person.box):
            assert person.box.contains_centre_of(face.box)
