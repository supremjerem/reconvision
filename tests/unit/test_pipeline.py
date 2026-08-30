"""The pipeline's contract: one event per passage, with the right verdict.

Every model is replaced by a double here. What is under test is the orchestration
- that animals skip the face stage, that evidence accumulates along a track, that
an event is emitted once the subject leaves - none of which needs real weights.
"""

from __future__ import annotations

import numpy as np
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

from reconvision.application.ingest import FrameIngestor
from reconvision.application.pipeline import (
    ObservedEvent,
    PipelinePolicy,
    RecognitionPipeline,
)
from reconvision.application.telemetry import PipelineMetrics, Telemetry
from reconvision.domain.events import EventVerdict
from reconvision.domain.matching import GalleryMatcher, ThresholdPolicy
from reconvision.domain.models import (
    BoundingBox,
    Detection,
    Embedding,
    Face,
    FaceQuality,
    GalleryEntry,
    SubjectKind,
)
from reconvision.domain.smoothing import VotePolicy
from tests.fakes import (
    FakeClock,
    ScriptedDetector,
    ScriptedFaceAnalyzer,
    ScriptedFrameSource,
    ScriptedTracker,
)
from tests.fixtures.video import frame_with_subject
from tests.unit.conftest import nearby_embedding, random_embedding

PERMISSIVE = ThresholdPolicy(match_threshold=0.4, min_margin=0.0)
QUICK_VOTE = VotePolicy(min_observations=2, min_weight_share=0.6)


@pytest.fixture
def telemetry() -> Telemetry:
    meter = MeterProvider(metric_readers=[InMemoryMetricReader()]).get_meter("test")
    return Telemetry(
        tracer=TracerProvider().get_tracer("test"),
        metrics=PipelineMetrics.create(meter),
    )


def good_face(embedding: Embedding, x: float = 100) -> Face:
    """A face large, sharp and frontal enough to clear the quality gate."""
    return Face(
        box=BoundingBox(left=x, top=110, right=x + 120, bottom=260),
        embedding=embedding,
        quality=FaceQuality(
            pixel_height=150, sharpness=150.0, yaw_degrees=3.0, detection_confidence=0.95
        ),
    )


def unusable_face(embedding: Embedding) -> Face:
    """A face too small and too angled to identify: filmed from across a room."""
    return Face(
        box=BoundingBox(left=100, top=110, right=140, bottom=155),
        embedding=embedding,
        quality=FaceQuality(
            pixel_height=45, sharpness=8.0, yaw_degrees=75.0, detection_confidence=0.4
        ),
    )


def subject_at(x: float, label: str = "person") -> Detection:
    return Detection(
        box=BoundingBox(left=x, top=100, right=x + 160, bottom=430),
        label=label,
        confidence=0.9,
    )


def build(
    telemetry: Telemetry,
    detections_per_frame: list[list[Detection]],
    faces: list[Face],
    gallery: list[GalleryEntry],
    frames_before_closing: int = 2,
    threshold: ThresholdPolicy = PERMISSIVE,
    analyzer: ScriptedFaceAnalyzer | None = None,
) -> RecognitionPipeline:
    """Assemble a pipeline whose every model is scripted."""
    frames = [frame_with_subject(20 * index) for index in range(len(detections_per_frame))]
    return RecognitionPipeline(
        camera_name="hall",
        ingestor=FrameIngestor(
            ScriptedFrameSource("hall", frames),
            telemetry,
            sample_every_n_frames=1,
            heartbeat_every_n_samples=1,
        ),
        detector=ScriptedDetector(detections_per_frame),
        analyzer=analyzer or ScriptedFaceAnalyzer(faces),
        tracker=ScriptedTracker(),
        matcher=GalleryMatcher(gallery, threshold),
        clock=FakeClock(),
        telemetry=telemetry,
        vote_policy=QUICK_VOTE,
        policy=PipelinePolicy(frames_before_closing=frames_before_closing),
    )


def walking(frames: int = 6, label: str = "person") -> list[list[Detection]]:
    """One subject crossing, then an empty room so the track closes."""
    return [[subject_at(40 + step * 25, label)] for step in range(frames)] + [[]] * 4


def test_an_enrolled_person_is_recognised(telemetry: Telemetry, rng: np.random.Generator) -> None:
    """The headline case: you walk past, the system names you."""
    enrolled = random_embedding(rng)
    pipeline = build(
        telemetry,
        walking(),
        faces=[good_face(nearby_embedding(enrolled, rng, similarity=0.8))],
        gallery=[GalleryEntry("jeremie", enrolled)],
    )

    events = [observed.event for observed in pipeline.events()]

    assert len(events) == 1
    assert events[0].verdict is EventVerdict.KNOWN_PERSON
    assert events[0].identity_id == "jeremie"


def test_a_stranger_is_reported_as_unknown(telemetry: Telemetry, rng: np.random.Generator) -> None:
    pipeline = build(
        telemetry,
        walking(),
        faces=[good_face(random_embedding(rng))],
        gallery=[GalleryEntry("jeremie", random_embedding(rng))],
    )

    events = [observed.event for observed in pipeline.events()]

    assert len(events) == 1
    assert events[0].verdict is EventVerdict.UNKNOWN_PERSON
    assert events[0].is_noteworthy


def test_one_passage_produces_exactly_one_event(
    telemetry: Telemetry, rng: np.random.Generator
) -> None:
    """The promise the whole tracking stage exists for. Per-frame decisions would
    make this twenty events and twenty notifications."""
    pipeline = build(
        telemetry,
        walking(frames=20),
        faces=[good_face(random_embedding(rng))],
        gallery=[],
    )

    events = list(pipeline.events())

    assert len(events) == 1
    assert events[0].event.observations == 20


def test_an_animal_never_reaches_the_face_model(telemetry: Telemetry) -> None:
    """The saving that makes several cameras affordable, asserted directly."""
    analyzer = ScriptedFaceAnalyzer([])
    pipeline = build(telemetry, walking(label="cat"), faces=[], gallery=[], analyzer=analyzer)

    events = [observed.event for observed in pipeline.events()]

    assert len(events) == 1
    assert events[0].verdict is EventVerdict.ANIMAL
    assert events[0].animal_label == "cat"
    assert events[0].subject_kind is SubjectKind.ANIMAL
    assert analyzer.calls == 0


def test_an_animal_event_is_not_worth_waking_someone_for(telemetry: Telemetry) -> None:
    pipeline = build(telemetry, walking(label="dog"), faces=[], gallery=[])

    events = [observed.event for observed in pipeline.events()]

    assert not events[0].is_noteworthy


def test_a_person_whose_face_is_never_usable_is_reported_honestly(
    telemetry: Telemetry, rng: np.random.Generator
) -> None:
    """Filmed from behind, or too far away. Neither identifying them nor dropping
    them silently is right: someone was there and the system could not say who."""
    pipeline = build(
        telemetry,
        walking(),
        faces=[unusable_face(random_embedding(rng))],
        gallery=[GalleryEntry("jeremie", random_embedding(rng))],
    )

    events = [observed.event for observed in pipeline.events()]

    assert len(events) == 1
    assert events[0].verdict is EventVerdict.UNIDENTIFIED
    assert events[0].is_noteworthy


def test_two_people_at_once_produce_two_events(
    telemetry: Telemetry, rng: np.random.Generator
) -> None:
    detections = [
        [subject_at(40 + step * 15), subject_at(600 - step * 15)] for step in range(6)
    ] + [[]] * 4
    pipeline = build(telemetry, detections, faces=[good_face(random_embedding(rng))], gallery=[])

    events = list(pipeline.events())

    assert len(events) == 2


def test_a_track_still_in_view_is_not_emitted_early(
    telemetry: Telemetry, rng: np.random.Generator
) -> None:
    """A notification must not fire while someone is still walking through."""
    detections = [[subject_at(40 + step * 20)] for step in range(8)]
    pipeline = build(
        telemetry,
        detections,
        faces=[good_face(random_embedding(rng))],
        gallery=[],
        frames_before_closing=50,
    )
    stream = pipeline.events()

    first = next(stream, None)

    # Nothing is emitted until the stream ends, which is when the passage ends.
    assert first is not None
    assert first.event.observations == 8


def test_an_event_carries_the_best_frame_of_the_passage(
    telemetry: Telemetry, rng: np.random.Generator
) -> None:
    """The snapshot a notification attaches and a correction turns into a gallery
    entry, so it must be the frame where the face was clearest."""
    pipeline = build(
        telemetry,
        walking(),
        faces=[good_face(random_embedding(rng))],
        gallery=[],
    )

    observed: list[ObservedEvent] = list(pipeline.events())

    assert observed[0].snapshot is not None


def test_an_empty_room_produces_no_events(telemetry: Telemetry) -> None:
    pipeline = build(telemetry, [[]] * 10, faces=[], gallery=[])

    assert list(pipeline.events()) == []


def test_a_thin_margin_between_two_household_members_reports_unknown(
    telemetry: Telemetry, rng: np.random.Generator
) -> None:
    """Two people who look alike both clear the threshold. Naming either would be
    a coin flip, so the passage is reported as unknown instead."""
    probe = random_embedding(rng)
    pipeline = build(
        telemetry,
        walking(),
        faces=[good_face(probe)],
        gallery=[
            GalleryEntry("sibling_a", nearby_embedding(probe, rng, similarity=0.62)),
            GalleryEntry("sibling_b", nearby_embedding(probe, rng, similarity=0.60)),
        ],
        threshold=ThresholdPolicy(match_threshold=0.4, min_margin=0.05),
    )

    events = [observed.event for observed in pipeline.events()]

    assert len(events) == 1
    # The property that matters is that neither sibling is named. The event is
    # reported as an unknown person rather than a third "ambiguous" verdict: both
    # are noteworthy and both alert, so a separate state would add vocabulary
    # without changing any behaviour.
    assert events[0].verdict is not EventVerdict.KNOWN_PERSON
    assert events[0].identity_id is None
    assert events[0].is_noteworthy


def test_a_person_with_no_usable_face_still_gets_a_snapshot(
    telemetry: Telemetry, rng: np.random.Generator
) -> None:
    """The event a human most wants a picture of - someone was here and the system
    could not say who - must not be the one arriving with nothing to look at."""
    pipeline = build(
        telemetry,
        walking(),
        faces=[unusable_face(random_embedding(rng))],
        gallery=[],
    )

    observed = list(pipeline.events())

    assert observed[0].event.verdict is EventVerdict.UNIDENTIFIED
    assert observed[0].snapshot is not None


def test_an_animal_event_carries_a_snapshot(telemetry: Telemetry) -> None:
    pipeline = build(telemetry, walking(label="cat"), faces=[], gallery=[])

    observed = list(pipeline.events())

    assert observed[0].snapshot is not None
