"""The front of the pipeline: turning a stream of frames into the few worth analysing.

Two filters, in order of cost. Sampling discards most frames outright; the motion
gate discards the rest of the still ones. What survives is handed to detection,
which is the expensive part. On a typical home camera this is the difference
between analysing 25 frames a second and analysing a handful when someone is
actually there.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from time import perf_counter

import structlog

from reconvision.adapters.video.motion import MotionGate
from reconvision.application.telemetry import MOTION_GATE, Telemetry
from reconvision.domain.models import Frame
from reconvision.domain.ports import FrameSource

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class IngestStats:
    """Throughput counters for one camera, reported by `reconvision run`."""

    camera_name: str
    decoded: int = 0
    sampled: int = 0
    analysed: int = 0
    started_at: float = field(default_factory=perf_counter)

    @property
    def elapsed_seconds(self) -> float:
        return max(1e-9, perf_counter() - self.started_at)

    @property
    def decoded_fps(self) -> float:
        """Frames arriving from the camera."""
        return self.decoded / self.elapsed_seconds

    @property
    def analysed_fps(self) -> float:
        """Frames that reached detection. The number that sets the CPU budget."""
        return self.analysed / self.elapsed_seconds

    @property
    def skipped_ratio(self) -> float:
        """Share of frames the two filters removed before any model ran."""
        return 1.0 - (self.analysed / self.decoded) if self.decoded else 0.0


class FrameIngestor:
    """Yields only the frames worth spending a detector on."""

    def __init__(
        self,
        source: FrameSource,
        telemetry: Telemetry,
        sample_every_n_frames: int = 3,
        motion_gate: MotionGate | None = None,
    ) -> None:
        if sample_every_n_frames < 1:
            message = f"Sampling must keep at least every frame, got {sample_every_n_frames}"
            raise ValueError(message)

        self._source = source
        self._telemetry = telemetry
        self._sample_every = sample_every_n_frames
        self._motion = motion_gate if motion_gate is not None else MotionGate()
        self.stats = IngestStats(camera_name=source.name)

    def analysable_frames(self) -> Iterator[Frame]:
        """Decode, sample, gate on motion, and yield what is left."""
        camera = {"camera": self._source.name}
        metrics = self._telemetry.metrics

        for index, frame in enumerate(self._source.frames()):
            self.stats.decoded += 1
            metrics.frames_decoded.add(1, camera)

            if index % self._sample_every != 0:
                continue
            self.stats.sampled += 1

            with self._telemetry.stage(MOTION_GATE, **camera):
                moved = self._motion.has_motion(frame)

            if not moved:
                continue

            self.stats.analysed += 1
            metrics.frames_analysed.add(1, camera)
            yield frame

    def log_throughput(self) -> None:
        """Report what the camera cost, which is what `run` prints on exit."""
        logger.info(
            "ingest_throughput",
            camera=self.stats.camera_name,
            decoded=self.stats.decoded,
            analysed=self.stats.analysed,
            decoded_fps=round(self.stats.decoded_fps, 1),
            analysed_fps=round(self.stats.analysed_fps, 1),
            skipped_ratio=round(self.stats.skipped_ratio, 3),
        )
