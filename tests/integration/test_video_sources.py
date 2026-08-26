"""Video sources are exercised against real encoded files and a real capture loop.

Mocking cv2 here would test the mock. These tests encode actual video and decode
it back, which is what catches the codec and lifecycle problems that matter.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from reconvision.adapters.video.sources import (
    FileFrameSource,
    LiveFrameSource,
    ReconnectPolicy,
    VideoSourceError,
    _open_capture,
    create_frame_source,
)
from tests.fixtures.video import FRAME_HEIGHT, FRAME_WIDTH, still_frame, walk_past, write_video

pytestmark = pytest.mark.integration

#: TEST-NET-1 (RFC 5737). Guaranteed unroutable, so these tests exercise the
#: unreachable-camera path without depending on the network they run on.
UNREACHABLE_CAMERA = "rtsp://192.0.2.1:554/nonexistent"


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return write_video(tmp_path / "clip.mp4", walk_past(steps=12))


def test_a_file_yields_every_frame_in_order(clip: Path) -> None:
    """Recorded footage must replay completely, or a test on it is not reproducible."""
    frames = list(FileFrameSource(clip).frames())

    assert len(frames) == 12
    assert frames[0].shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)


def test_a_file_source_is_named_after_the_file(clip: Path) -> None:
    assert FileFrameSource(clip).name == "clip"


def test_a_missing_file_fails_with_the_path(tmp_path: Path) -> None:
    source = FileFrameSource(tmp_path / "absent.mp4")

    with pytest.raises(VideoSourceError, match=r"absent\.mp4"):
        list(source.frames())


def test_an_undecodable_file_fails_rather_than_yielding_nothing(tmp_path: Path) -> None:
    """Silence here would look exactly like an empty camera, which is the wrong
    diagnosis to hand someone at three in the morning."""
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    with pytest.raises(VideoSourceError, match="decode"):
        list(FileFrameSource(broken).frames())


def test_a_file_source_releases_its_decoder_when_abandoned(clip: Path) -> None:
    """Consumers stop early all the time. The decoder must not leak when they do."""
    source = FileFrameSource(clip)
    frames = source.frames()
    next(frames)
    frames.close()  # type: ignore[attr-defined]  # frames() is a generator

    assert source._capture is None


def test_a_live_source_yields_the_newest_frame_and_drops_the_backlog() -> None:
    """The property that bounds latency: a consumer slower than the camera sees
    recent frames, not a queue of stale ones."""
    source = LiveFrameSource(source="ignored", name="fake")
    published = threading.Event()

    def publish_faster_than_the_consumer() -> None:
        for index in range(50):
            with source._lock:
                if source._latest is not None:
                    source._dropped += 1
                source._latest = np.full((8, 8, 3), index, dtype=np.uint8)
            source._frame_ready.set()
        published.set()

    threading.Thread(target=publish_faster_than_the_consumer, daemon=True).start()
    published.wait(timeout=5)

    with source._lock:
        latest = source._latest

    assert latest is not None
    assert source.dropped_frames > 0
    # The consumer would receive a recent frame, never the first of the backlog.
    assert int(latest[0, 0, 0]) > 0
    source.close()


def test_a_connection_attempt_cannot_block_for_longer_than_configured() -> None:
    """Regression guard on a defect that is invisible until deployment.

    OpenCV's FFmpeg backend defaults to a 30 second connect timeout, and that wait
    is not interruptible. Left at the default, a camera that is switched off both
    stretches every retry by half a minute and stalls shutdown long enough for a
    container runtime to kill the process instead of letting it stop cleanly.
    """
    started = time.monotonic()
    capture = _open_capture(UNREACHABLE_CAMERA, timeout_seconds=2.0)
    elapsed = time.monotonic() - started
    capture.release()

    assert elapsed < 10.0


def test_an_unreachable_camera_gives_up_after_the_configured_attempts() -> None:
    """Without a cap, a permanently dead camera retries forever in the logs."""
    source = LiveFrameSource(
        source=UNREACHABLE_CAMERA,
        name="dead",
        reconnect=ReconnectPolicy(
            initial_delay_seconds=0.01,
            max_delay_seconds=0.02,
            max_attempts=2,
            connect_timeout_seconds=1.0,
        ),
    )
    started = time.monotonic()

    frames = list(source.frames())

    assert frames == []
    assert time.monotonic() - started < 20


def test_backoff_grows_and_then_stops_growing() -> None:
    """A camera that reboots should be picked up quickly; one that is gone should
    not be hammered every second forever."""
    policy = ReconnectPolicy(initial_delay_seconds=1.0, backoff_factor=2.0, max_delay_seconds=10.0)

    assert [policy.delay_for(attempt) for attempt in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 8.0]
    assert policy.delay_for(20) == 10.0


def test_closing_a_live_source_is_immediate_even_mid_backoff() -> None:
    """Shutdown must not block for a minute of backoff, or the container gets killed."""
    source = LiveFrameSource(
        source=UNREACHABLE_CAMERA,
        name="dead",
        reconnect=ReconnectPolicy(initial_delay_seconds=30.0, connect_timeout_seconds=1.0),
    )
    consumer = threading.Thread(target=lambda: list(source.frames()), daemon=True)
    consumer.start()
    time.sleep(0.3)

    started = time.monotonic()
    source.close()

    assert time.monotonic() - started < 6


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("webcam:0", LiveFrameSource),
        ("rtsp://camera.local/stream", LiveFrameSource),
        ("https://camera.local/stream.m3u8", LiveFrameSource),
        ("./clip.mp4", FileFrameSource),
        ("/srv/footage/clip.mkv", FileFrameSource),
    ],
)
def test_the_factory_picks_the_right_source_for_a_spec(spec: str, expected: type) -> None:
    assert isinstance(create_frame_source(spec), expected)


def test_a_malformed_webcam_spec_is_refused() -> None:
    with pytest.raises(VideoSourceError, match="webcam:<index>"):
        create_frame_source("webcam:front")


def test_a_written_clip_round_trips_through_the_encoder(tmp_path: Path) -> None:
    """Guards the fixture the other integration tests are built on."""
    path = write_video(tmp_path / "still.mp4", [still_frame()] * 5)

    assert len(list(FileFrameSource(path).frames())) == 5
