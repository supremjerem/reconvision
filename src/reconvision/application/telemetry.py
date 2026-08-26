"""Traces, metrics and structured logs.

Wired before the pipeline exists so every stage is instrumented as it is written
rather than retrofitted. The metrics below are the ones that answer the questions
this system actually raises in operation: is it keeping up, where is the time
going, and why did it decide that.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import structlog
from opentelemetry import metrics, trace
from opentelemetry.metrics import Counter, Histogram, Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Tracer
from structlog.typing import EventDict, WrappedLogger

from reconvision import __version__
from reconvision.application.config import Settings, TelemetryExporter

SERVICE_NAME = "reconvision"


class PipelineStage(str):
    """Stage names, used as a span name and as a metric attribute.

    A plain string subclass rather than an enum so the values can be used
    directly as OpenTelemetry attributes without conversion at every call site.
    """

    __slots__ = ()


MOTION_GATE = PipelineStage("motion_gate")
OBJECT_DETECTION = PipelineStage("object_detection")
FACE_DETECTION = PipelineStage("face_detection")
EMBEDDING = PipelineStage("embedding")
MATCHING = PipelineStage("matching")
TRACKING = PipelineStage("tracking")


@dataclass(frozen=True, slots=True)
class PipelineMetrics:
    """The instruments the pipeline reports through."""

    frames_decoded: Counter
    frames_analysed: Counter
    frames_dropped: Counter
    stage_duration: Histogram
    detections: Counter
    faces_rejected: Counter
    match_similarity: Histogram
    events_emitted: Counter
    notifications: Counter

    @classmethod
    def create(cls, meter: Meter) -> PipelineMetrics:
        return cls(
            frames_decoded=meter.create_counter(
                "reconvision.frames.decoded",
                unit="1",
                description="Frames pulled off a camera.",
            ),
            frames_analysed=meter.create_counter(
                "reconvision.frames.analysed",
                unit="1",
                description="Frames that passed the motion gate and were inspected.",
            ),
            frames_dropped=meter.create_counter(
                "reconvision.frames.dropped",
                unit="1",
                # The single most important health signal: sustained drops mean
                # inference is not keeping up with the cameras.
                description="Frames discarded because inference fell behind.",
            ),
            stage_duration=meter.create_histogram(
                "reconvision.stage.duration",
                unit="ms",
                description="Wall time per pipeline stage.",
            ),
            detections=meter.create_counter(
                "reconvision.detections",
                unit="1",
                description="Objects detected, by label.",
            ),
            faces_rejected=meter.create_counter(
                "reconvision.faces.rejected",
                unit="1",
                # Broken down by reason, so a camera that mostly sees faces too
                # small can be repositioned rather than guessed about.
                description="Faces declined by the quality gate, by reason.",
            ),
            match_similarity=meter.create_histogram(
                "reconvision.match.similarity",
                unit="1",
                # The distribution is what a threshold is chosen from. Recording it
                # in production means calibration reflects the real cameras.
                description="Cosine similarity of the best gallery candidate.",
            ),
            events_emitted=meter.create_counter(
                "reconvision.events",
                unit="1",
                description="Recognition events, by verdict.",
            ),
            notifications=meter.create_counter(
                "reconvision.notifications",
                unit="1",
                description="Notification deliveries, by channel and outcome.",
            ),
        )


@dataclass(frozen=True, slots=True)
class Telemetry:
    """The handle the rest of the application uses to report on itself."""

    tracer: Tracer
    metrics: PipelineMetrics

    @contextmanager
    def stage(self, stage: str, **attributes: str | int | float | bool) -> Iterator[None]:
        """Time one pipeline stage, recording a span and a duration histogram.

        Both, because they answer different questions: the histogram shows that
        the face stage is slow, the span shows which frame it was slow on.
        """
        started = perf_counter()
        with self.tracer.start_as_current_span(stage) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
            try:
                yield
            finally:
                elapsed_ms = (perf_counter() - started) * 1000.0
                self.metrics.stage_duration.record(elapsed_ms, {"stage": stage, **attributes})


def configure_logging(log_level: str = "INFO") -> None:
    """Structured JSON logs, with the trace id attached to every line.

    The trace id is the point: a log line about a suspicious match can be taken
    straight to the trace showing the frame, the stage timings and the scores.
    """
    logging.basicConfig(format="%(message)s", level=getattr(logging, log_level.upper(), 20))

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_trace_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), 20)
        ),
        cache_logger_on_first_use=True,
    )


def _add_trace_context(_logger: WrappedLogger, _method: str, event: EventDict) -> EventDict:
    """Attach the current trace and span ids, when there is a trace in progress."""
    span = trace.get_current_span()
    context = span.get_span_context()
    if context.is_valid:
        event["trace_id"] = format(context.trace_id, "032x")
        event["span_id"] = format(context.span_id, "016x")
    return event


def configure_telemetry(settings: Settings) -> Telemetry:
    """Install the trace and metric providers, and return the application handle."""
    configure_logging(settings.log_level)

    resource = Resource.create({"service.name": SERVICE_NAME, "service.version": __version__})
    tracer_provider = TracerProvider(resource=resource)
    readers: list[PeriodicExportingMetricReader] = []

    if settings.telemetry_exporter is TelemetryExporter.OTLP:
        # Imported lazily: the OTLP exporter pulls in protobuf and gRPC machinery
        # that a console-only development run has no reason to load.
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        endpoint = settings.otlp_endpoint.rstrip("/")
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        )
        readers.append(
            PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"))
        )
    elif settings.telemetry_exporter is TelemetryExporter.CONSOLE:
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        readers.append(
            PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=60_000)
        )

    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))

    meter = metrics.get_meter(SERVICE_NAME, __version__)
    return Telemetry(
        tracer=trace.get_tracer(SERVICE_NAME, __version__),
        metrics=PipelineMetrics.create(meter),
    )
