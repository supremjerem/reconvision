"""Frame sources: files for tests, a webcam for development, RTSP for real cameras.

Live sources and recorded ones need opposite behaviour under load. A file must
yield every frame or the test is not reproducible. A camera must yield the
*newest* frame and discard whatever piled up behind it, because a frame from four
seconds ago has no value for live recognition and processing it only pushes the
system further behind.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import structlog

from reconvision.domain.models import Frame

logger = structlog.get_logger(__name__)

# OpenCV's stubs describe read() as returning any numeric array. For the video
# backends used here it is always an 8-bit BGR image, so the frames are cast
# rather than converted: np.asarray would copy every frame for nothing.

#: How long to wait for a first frame before treating the source as broken.
_OPEN_TIMEOUT_SECONDS = 10.0

#: OpenCV reads its FFmpeg options from the environment at capture construction,
#: so the variable has to be set immediately before and cannot be passed in. The
#: lock keeps two camera threads from overwriting each other's settings.
_FFMPEG_OPTIONS_LOCK = threading.Lock()
_RTSP_SCHEMES = ("rtsp://", "rtsps://")


class VideoSourceError(RuntimeError):
    """A source could not be opened at all."""


@dataclass(frozen=True, slots=True)
class ReconnectPolicy:
    """How hard to try when a camera drops off the network.

    Home cameras reboot, Wi-Fi drops and PoE switches get power-cycled. None of
    those should end the process; all of them should stop hammering a camera that
    is genuinely gone.
    """

    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_factor: float = 2.0
    #: How long a single connection attempt may block. OpenCV's FFmpeg backend
    #: defaults to 30 seconds and that wait cannot be interrupted, which means a
    #: dead camera stalls shutdown long enough for a container runtime to kill
    #: the process instead of letting it stop cleanly.
    connect_timeout_seconds: float = 5.0
    #: None means retry forever, which is the right default for a camera that is
    #: expected to come back.
    max_attempts: int | None = None

    def delay_for(self, attempt: int) -> float:
        """Delay before the given retry attempt, capped."""
        delay = self.initial_delay_seconds * (self.backoff_factor ** max(0, attempt - 1))
        return min(delay, self.max_delay_seconds)


class FileFrameSource:
    """Replays a video file, every frame, in order.

    Used by the integration tests and by `reconvision run --source ./clip.mp4` to
    check a change against recorded footage. Never drops frames: a test that
    skipped frames under load would not be reproducible.
    """

    def __init__(self, path: Path | str, name: str | None = None) -> None:
        self._path = Path(path)
        self._name = name or self._path.stem
        self._capture: cv2.VideoCapture | None = None

    @property
    def name(self) -> str:
        return self._name

    def frames(self) -> Iterator[Frame]:
        if not self._path.exists():
            message = f"Video file not found: {self._path}"
            raise VideoSourceError(message)

        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            message = f"Could not decode video file: {self._path}"
            raise VideoSourceError(message)

        self._capture = capture
        try:
            while True:
                received, frame = capture.read()
                if not received:
                    return
                yield cast("Frame", frame)
        finally:
            self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class LiveFrameSource:
    """A camera or capture device, read newest-frame-first.

    A background thread drains the decoder continuously and keeps only the latest
    frame. That is what bounds latency: without it OpenCV buffers frames, and a
    consumer slower than the camera falls steadily further behind until it is
    recognising people who left the room a minute ago.
    """

    def __init__(
        self,
        source: str | int,
        name: str,
        reconnect: ReconnectPolicy | None = None,
    ) -> None:
        self._source = source
        self._name = name
        self._reconnect = reconnect or ReconnectPolicy()

        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._dropped = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def dropped_frames(self) -> int:
        """Frames decoded but never consumed, because a newer one arrived first.

        Reported as a metric: a steadily climbing count means inference cannot
        keep up with this camera and the sampling rate should come down.
        """
        return self._dropped

    def frames(self) -> Iterator[Frame]:
        self._stop.clear()
        self._reader = threading.Thread(
            target=self._read_continuously, name=f"reader-{self._name}", daemon=True
        )
        self._reader.start()

        try:
            while not self._stop.is_set():
                if not self._frame_ready.wait(timeout=_OPEN_TIMEOUT_SECONDS):
                    if self._stop.is_set():
                        return
                    logger.warning("camera_stalled", camera=self._name)
                    continue

                with self._lock:
                    frame = self._latest
                    self._latest = None
                    self._frame_ready.clear()

                if frame is not None:
                    yield frame
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        self._frame_ready.set()
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=5.0)
        self._reader = None

    def _read_continuously(self) -> None:
        """Decode in the background, keeping only the newest frame."""
        attempt = 0
        while not self._stop.is_set():
            capture = _open_capture(self._source, self._reconnect.connect_timeout_seconds)
            # Ask the decoder itself to hold as little as possible. Honoured by
            # some backends and ignored by others, which is why the newest-frame
            # slot below is the actual guarantee rather than an optimisation.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not capture.isOpened():
                capture.release()
                attempt += 1
                if not self._wait_before_retry(attempt):
                    return
                continue

            logger.info("camera_connected", camera=self._name, attempt=attempt)
            attempt = 0
            self._drain(capture)
            capture.release()

            if not self._stop.is_set():
                attempt += 1
                logger.warning("camera_disconnected", camera=self._name)
                if not self._wait_before_retry(attempt):
                    return

    def _drain(self, capture: cv2.VideoCapture) -> None:
        """Publish frames until the stream fails or the source is closed."""
        while not self._stop.is_set():
            received, frame = capture.read()
            if not received:
                return

            with self._lock:
                if self._latest is not None:
                    self._dropped += 1
                self._latest = cast("Frame", frame)
            self._frame_ready.set()

    def _wait_before_retry(self, attempt: int) -> bool:
        """Sleep before reconnecting. False means give up."""
        if self._reconnect.max_attempts is not None and attempt > self._reconnect.max_attempts:
            logger.error("camera_unreachable", camera=self._name, attempts=attempt)
            self._stop.set()
            self._frame_ready.set()
            return False

        delay = self._reconnect.delay_for(attempt)
        logger.info("camera_retry", camera=self._name, attempt=attempt, delay_seconds=delay)
        # Waiting on the stop event rather than sleeping means close() is
        # immediate instead of blocking for up to a minute of backoff.
        return not self._stop.wait(timeout=delay)


def _open_capture(source: str | int, timeout_seconds: float) -> cv2.VideoCapture:
    """Open a capture, bounding how long an unreachable camera can block.

    TCP transport is requested alongside the timeout: over home Wi-Fi, RTSP over
    UDP loses packets and delivers visibly corrupted frames, which the detector
    then reports as spurious motion.
    """
    if not (isinstance(source, str) and source.startswith(_RTSP_SCHEMES)):
        return cv2.VideoCapture(source)

    timeout_microseconds = int(timeout_seconds * 1_000_000)
    with _FFMPEG_OPTIONS_LOCK:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;tcp|timeout;{timeout_microseconds}"
        )
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)


def create_frame_source(
    spec: str,
    name: str | None = None,
    reconnect: ReconnectPolicy | None = None,
) -> FileFrameSource | LiveFrameSource:
    """Build the right source for a manifest entry or a `--source` argument.

    Accepts `webcam:0`, an `rtsp://` or `http://` URL, or a path to a video file.
    """
    if spec.startswith("webcam:"):
        index = spec.removeprefix("webcam:")
        if not index.isdigit():
            message = f"Webcam source must be webcam:<index>, got {spec!r}"
            raise VideoSourceError(message)
        return LiveFrameSource(int(index), name or f"webcam{index}", reconnect)

    if spec.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return LiveFrameSource(spec, name or "camera", reconnect)

    return FileFrameSource(spec, name)
