"""Storing the single best frame of each event.

Kept on disk rather than in the database: they are large binary blobs with their
own retention rules, and SQLite is the wrong place for either property.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import structlog

from reconvision.adapters.images import read_image
from reconvision.domain.models import Frame

logger = structlog.get_logger(__name__)

_JPEG_QUALITY = 85


class FileSnapshotStore:
    """Writes snapshots as JPEGs under a date-partitioned directory.

    Partitioned by day so that retention is a directory removal rather than a scan
    of a folder holding a year of images, and so a human can look at one day.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, frame: Frame, event_id: str) -> str:
        day = datetime.now().astimezone().strftime("%Y-%m-%d")
        folder = self._root / day
        folder.mkdir(parents=True, exist_ok=True)

        path = folder / f"{event_id}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        return f"{day}/{event_id}.jpg"

    def load(self, snapshot_id: str) -> Frame | None:
        path = self._resolve(snapshot_id)
        return read_image(path) if path is not None and path.exists() else None

    def path_for(self, snapshot_id: str) -> Path | None:
        """The file backing a snapshot, for the web screen to serve directly."""
        path = self._resolve(snapshot_id)
        return path if path is not None and path.exists() else None

    def purge_older_than(self, cutoff: datetime) -> int:
        """Remove whole days that have aged out, returning how many files went."""
        removed = 0
        for folder in sorted(self._root.iterdir()):
            if not folder.is_dir():
                continue
            try:
                day = datetime.strptime(folder.name, "%Y-%m-%d").astimezone()
            except ValueError:
                continue
            if day.date() >= cutoff.date():
                continue

            for file in folder.glob("*.jpg"):
                file.unlink()
                removed += 1
            folder.rmdir()

        if removed:
            logger.info("snapshots_purged", removed=removed, before=cutoff.date().isoformat())
        return removed

    def _resolve(self, snapshot_id: str) -> Path | None:
        """Turn a stored id into a path, refusing anything that escapes the root.

        Snapshot ids reach this from HTTP requests, so a crafted id must not be
        able to read arbitrary files off the machine.
        """
        candidate = (self._root / snapshot_id).resolve()
        root = self._root.resolve()
        return candidate if candidate.is_relative_to(root) else None
