"""Posting events to an arbitrary HTTP endpoint.

The escape hatch: Discord, Slack, n8n, a home-grown service. Anything this project
does not support natively can be reached by someone who wants it.
"""

from __future__ import annotations

import base64

import cv2
import httpx
import structlog

from reconvision.domain.events import RecognitionEvent
from reconvision.domain.models import Frame

logger = structlog.get_logger(__name__)

_REQUEST_TIMEOUT_SECONDS = 10.0
_SNAPSHOT_QUALITY = 80


class WebhookNotifier:
    """POSTs a JSON document describing the event."""

    def __init__(
        self,
        url: str,
        include_snapshot: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        if not url:
            message = "A webhook URL is required"
            raise ValueError(message)
        self._url = url
        self._include_snapshot = include_snapshot
        self._client = client or httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS)

    def notify(self, event: RecognitionEvent, snapshot: Frame | None = None) -> None:
        payload: dict[str, object] = {
            "event_id": event.event_id,
            "camera": event.camera_name,
            "verdict": event.verdict.value,
            "identity": event.identity_id,
            "animal": event.animal_label,
            "started_at": event.started_at.isoformat(),
            "ended_at": event.ended_at.isoformat(),
            "duration_seconds": round(event.duration_seconds, 1),
            "confidence": round(event.confidence, 3),
            "best_similarity": round(event.best_similarity, 3),
            "observations": event.observations,
            "noteworthy": event.is_noteworthy,
        }

        if self._include_snapshot and snapshot is not None:
            encoded = _encode(snapshot)
            if encoded is not None:
                # Base64 inside the JSON rather than multipart: every endpoint that
                # accepts JSON can read it, and the alternative is a second request
                # shape for every consumer to handle.
                payload["snapshot_jpeg_base64"] = base64.b64encode(encoded).decode("ascii")

        self._client.post(self._url, json=payload).raise_for_status()
        logger.debug("webhook_delivered", event_id=event.event_id)

    def close(self) -> None:
        self._client.close()


def _encode(snapshot: Frame) -> bytes | None:
    success, buffer = cv2.imencode(".jpg", snapshot, [cv2.IMWRITE_JPEG_QUALITY, _SNAPSHOT_QUALITY])
    return bytes(buffer) if success else None
