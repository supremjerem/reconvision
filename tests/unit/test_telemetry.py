"""Telemetry must record what happened without ever becoming a failure mode.

Instrumentation that can crash the pipeline is worse than no instrumentation, so
these tests care as much about what telemetry does not break as about what it
records.
"""

from __future__ import annotations

import json
import logging

import pytest
import structlog
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from reconvision.application.config import TelemetryExporter
from reconvision.application.telemetry import (
    FACE_DETECTION,
    OBJECT_DETECTION,
    PipelineMetrics,
    Telemetry,
    configure_logging,
    configure_telemetry,
)
from tests.fakes import build_settings


@pytest.fixture
def spans() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def telemetry(spans: InMemorySpanExporter) -> Telemetry:
    """A self-contained Telemetry that records into memory."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("test")
    return Telemetry(tracer=provider.get_tracer("test"), metrics=PipelineMetrics.create(meter))


def test_a_stage_records_a_span_named_after_itself(
    telemetry: Telemetry, spans: InMemorySpanExporter
) -> None:
    with telemetry.stage(OBJECT_DETECTION, camera="hall"):
        pass

    (span,) = spans.get_finished_spans()
    assert span.name == "object_detection"
    assert span.attributes is not None
    assert span.attributes["camera"] == "hall"


def test_stages_nest_so_a_slow_frame_can_be_broken_down(
    telemetry: Telemetry, spans: InMemorySpanExporter
) -> None:
    """The reason for spans on top of histograms: seeing which stage ate the frame."""
    with telemetry.stage(OBJECT_DETECTION), telemetry.stage(FACE_DETECTION):
        pass

    finished = {span.name: span for span in spans.get_finished_spans()}
    assert finished["face_detection"].parent is not None
    assert finished["face_detection"].parent.span_id == finished["object_detection"].context.span_id


def test_a_stage_that_raises_still_records_its_timing(
    telemetry: Telemetry, spans: InMemorySpanExporter
) -> None:
    """A stage that blew up is exactly the one whose timing you want."""
    with pytest.raises(RuntimeError), telemetry.stage(FACE_DETECTION):
        raise RuntimeError("model failed to load")

    (span,) = spans.get_finished_spans()
    assert span.name == "face_detection"


def test_a_stage_does_not_swallow_the_error_it_timed(telemetry: Telemetry) -> None:
    """Instrumentation observes; it must never change control flow."""
    with pytest.raises(ValueError, match="boom"), telemetry.stage(FACE_DETECTION):
        raise ValueError("boom")


def test_logs_are_json_and_carry_the_trace_id(
    telemetry: Telemetry, capsys: pytest.CaptureFixture[str]
) -> None:
    """The link that makes a suspicious log line traceable to the frame that caused it."""
    configure_logging("INFO")
    logger = structlog.get_logger()

    with telemetry.stage(OBJECT_DETECTION):
        logger.info("matched", identity="jeremie", similarity=0.71)

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["event"] == "matched"
    assert payload["identity"] == "jeremie"
    assert len(payload["trace_id"]) == 32


def test_logs_outside_a_trace_simply_omit_the_trace_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")

    structlog.get_logger().info("starting up")

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "trace_id" not in payload


def test_telemetry_can_be_turned_off_entirely() -> None:
    """A NAS with no collector should pay nothing for telemetry it cannot ship."""
    configured = configure_telemetry(build_settings(telemetry_exporter=TelemetryExporter.NONE))

    with configured.stage(OBJECT_DETECTION):
        configured.metrics.frames_decoded.add(1, {"camera": "hall"})


def test_the_log_level_is_honoured(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING")

    structlog.get_logger().debug("noisy detail")

    assert capsys.readouterr().out.strip() == ""
    configure_logging("INFO")
    logging.getLogger().setLevel(logging.INFO)
