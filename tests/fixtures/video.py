"""Synthetic footage, so the video tests exercise real decoding without shipping
recordings of anyone's home into a public repository."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from reconvision.domain.models import Frame

FRAME_WIDTH = 320
FRAME_HEIGHT = 240


def still_frame(brightness: int = 40) -> Frame:
    """A uniform frame standing in for an empty room."""
    return np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), brightness, dtype=np.uint8)


def frame_with_subject(x_position: int, size: int = 90, brightness: int = 40) -> Frame:
    """A frame with a bright rectangle standing in for someone walking through."""
    frame = still_frame(brightness)
    left = max(0, min(FRAME_WIDTH - size, x_position))
    top = (FRAME_HEIGHT - size) // 2
    frame[top : top + size, left : left + size] = 230
    return frame


def noisy_frame(rng: np.random.Generator, brightness: int = 40) -> Frame:
    """An empty room as a real sensor sees it: uniform, plus grain.

    The case the motion gate must not trip on, since a dark room produces this on
    every single frame.
    """
    noise = rng.integers(-6, 7, size=(FRAME_HEIGHT, FRAME_WIDTH, 3))
    return np.clip(noise + brightness, 0, 255).astype(np.uint8)


def write_video(path: Path, frames: list[Frame], fps: int = 25) -> Path:
    """Encode frames to a real video file the decoder has to read back."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    if not writer.isOpened():
        message = f"Could not open a video writer for {path}"
        raise RuntimeError(message)
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def walk_past(steps: int = 12) -> list[Frame]:
    """A subject crossing the frame from left to right."""
    stride = FRAME_WIDTH // max(1, steps)
    return [frame_with_subject(step * stride) for step in range(steps)]
