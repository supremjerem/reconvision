"""Ingestion decides how much work the expensive stages are asked to do."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

from reconvision.adapters.video.motion import MotionGate, MotionPolicy
from reconvision.application.ingest import FrameIngestor
from reconvision.application.telemetry import PipelineMetrics, Telemetry
from reconvision.domain.models import Frame
from tests.fakes import ScriptedFrameSource
from tests.fixtures.video import frame_with_subject, still_frame


@pytest.fixture
def telemetry() -> Telemetry:
    meter = MeterProvider(metric_readers=[InMemoryMetricReader()]).get_meter("test")
    return Telemetry(
        tracer=TracerProvider().get_tracer("test"),
        metrics=PipelineMetrics.create(meter),
    )


class AlwaysMoving(MotionGate):
    """Isolates the sampling rule from the motion rule."""

    def has_motion(self, frame: Frame) -> bool:
        return True


def test_an_empty_room_is_almost_entirely_skipped(telemetry: Telemetry) -> None:
    """The saving the whole design rests on: 30 frames of nothing cost 3 detections."""
    source = ScriptedFrameSource("hall", [still_frame()] * 30)
    ingestor = FrameIngestor(
        source, telemetry, sample_every_n_frames=1, heartbeat_every_n_samples=12
    )

    analysed = list(ingestor.analysable_frames())

    assert ingestor.stats.decoded == 30
    assert len(analysed) <= 3


def test_a_still_scene_is_rechecked_on_a_heartbeat(telemetry: Telemetry) -> None:
    """Frame differencing reports change, not presence. Someone who walks in and
    then stands still vanishes from the gate, so without a heartbeat they could
    stand in the room indefinitely without the detector ever looking again."""
    source = ScriptedFrameSource("hall", [still_frame()] * 30)
    ingestor = FrameIngestor(
        source, telemetry, sample_every_n_frames=1, heartbeat_every_n_samples=10
    )

    list(ingestor.analysable_frames())

    assert ingestor.stats.heartbeats >= 2


def test_the_heartbeat_interval_controls_how_often_a_still_scene_is_rechecked(
    telemetry: Telemetry,
) -> None:
    def analysed_with(heartbeat: int) -> int:
        ingestor = FrameIngestor(
            ScriptedFrameSource("hall", [still_frame()] * 60),
            telemetry,
            sample_every_n_frames=1,
            heartbeat_every_n_samples=heartbeat,
        )
        return len(list(ingestor.analysable_frames()))

    assert analysed_with(5) > analysed_with(30)


def test_an_impossible_heartbeat_is_refused(telemetry: Telemetry) -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        FrameIngestor(ScriptedFrameSource("hall", []), telemetry, heartbeat_every_n_samples=0)


def test_someone_walking_through_reaches_the_detector(telemetry: Telemetry) -> None:
    walking = [frame_with_subject(x) for x in range(0, 200, 20)]
    source = ScriptedFrameSource("hall", walking)
    ingestor = FrameIngestor(source, telemetry, sample_every_n_frames=1)

    assert len(list(ingestor.analysable_frames())) == len(walking)


def test_sampling_keeps_one_frame_in_n(telemetry: Telemetry) -> None:
    source = ScriptedFrameSource("hall", [still_frame()] * 30)
    ingestor = FrameIngestor(source, telemetry, sample_every_n_frames=3, motion_gate=AlwaysMoving())

    list(ingestor.analysable_frames())

    assert ingestor.stats.decoded == 30
    assert ingestor.stats.analysed == 10


def test_the_two_filters_compose(telemetry: Telemetry) -> None:
    """Sampling runs first because it is free; the motion gate only pays for what
    sampling let through."""
    frames = [still_frame()] * 20 + [frame_with_subject(x) for x in range(0, 100, 10)]
    ingestor = FrameIngestor(
        ScriptedFrameSource("hall", frames), telemetry, sample_every_n_frames=2
    )

    analysed = list(ingestor.analysable_frames())

    assert ingestor.stats.decoded == 30
    assert ingestor.stats.sampled == 15
    assert len(analysed) < ingestor.stats.sampled


def test_throughput_is_reported_for_capacity_planning(telemetry: Telemetry) -> None:
    """`run` prints these, and they are what decides how many cameras fit."""
    ingestor = FrameIngestor(
        ScriptedFrameSource("hall", [still_frame()] * 40), telemetry, sample_every_n_frames=4
    )

    list(ingestor.analysable_frames())

    assert ingestor.stats.decoded_fps > 0
    assert ingestor.stats.skipped_ratio == pytest.approx(1 - 1 / 40)
    ingestor.log_throughput()


def test_statistics_are_safe_before_any_frame_arrives(telemetry: Telemetry) -> None:
    """A camera that never connects must not make the stats line divide by zero."""
    ingestor = FrameIngestor(ScriptedFrameSource("hall", []), telemetry)

    list(ingestor.analysable_frames())

    assert ingestor.stats.skipped_ratio == 0.0
    assert ingestor.stats.decoded_fps == 0.0


def test_a_sensitive_gate_lets_more_through_than_a_strict_one(
    telemetry: Telemetry,
) -> None:
    frames = [frame_with_subject(x, size=20) for x in range(0, 120, 12)]

    def analysed_with(ratio: float) -> int:
        ingestor = FrameIngestor(
            ScriptedFrameSource("hall", frames),
            telemetry,
            sample_every_n_frames=1,
            motion_gate=MotionGate(MotionPolicy(min_changed_ratio=ratio)),
        )
        return len(list(ingestor.analysable_frames()))

    assert analysed_with(0.001) > analysed_with(0.5)


def test_a_sampling_rate_that_would_discard_everything_is_refused(
    telemetry: Telemetry,
) -> None:
    with pytest.raises(ValueError, match="at least every frame"):
        FrameIngestor(ScriptedFrameSource("hall", []), telemetry, sample_every_n_frames=0)
