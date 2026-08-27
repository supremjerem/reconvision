"""Push notifications through ntfy.

The primary interface. A dashboard is something you have to remember to open; an
alert about an unknown person at three in the morning has to reach a phone.
"""

from __future__ import annotations

import base64

import cv2
import httpx
import structlog

from reconvision.domain.events import EventVerdict, RecognitionEvent
from reconvision.domain.models import Frame

logger = structlog.get_logger(__name__)

#: JPEG quality for the attached snapshot. Enough to recognise a face on a phone,
#: small enough not to stall on a slow uplink.
_SNAPSHOT_QUALITY = 80
_REQUEST_TIMEOUT_SECONDS = 10.0

#: ntfy priorities. An unknown person should break through a silenced phone; a
#: recognised household member should not.
_PRIORITY_BY_VERDICT = {
    EventVerdict.UNKNOWN_PERSON: "high",
    EventVerdict.UNIDENTIFIED: "default",
    EventVerdict.KNOWN_PERSON: "low",
    EventVerdict.ANIMAL: "low",
}

_TAG_BY_VERDICT = {
    EventVerdict.UNKNOWN_PERSON: "warning",
    EventVerdict.UNIDENTIFIED: "grey_question",
    EventVerdict.KNOWN_PERSON: "house",
    EventVerdict.ANIMAL: "paw_prints",
}


class NtfyNotifier:
    """Sends an event, with its snapshot, to an ntfy topic."""

    def __init__(
        self,
        topic: str,
        base_url: str = "https://ntfy.sh",
        client: httpx.Client | None = None,
    ) -> None:
        if not topic:
            message = "An ntfy topic is required"
            raise ValueError(message)
        self._topic = topic
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS)

    def notify(self, event: RecognitionEvent, snapshot: Frame | None = None) -> None:
        headers = {
            "Title": _encode_header(_title(event)),
            "Priority": _PRIORITY_BY_VERDICT.get(event.verdict, "default"),
            "Tags": _TAG_BY_VERDICT.get(event.verdict, "eyes"),
        }

        encoded = _encode(snapshot)
        if encoded is not None:
            # ntfy attaches the raw body as a file when Filename is set, which puts
            # the picture in the notification itself rather than behind a link.
            headers["Filename"] = f"{event.camera_name}.jpg"
            response = self._client.put(
                f"{self._base_url}/{self._topic}", content=encoded, headers=headers
            )
        else:
            response = self._client.post(
                f"{self._base_url}/{self._topic}",
                content=_body(event).encode("utf-8"),
                headers=headers,
            )

        response.raise_for_status()
        logger.debug("ntfy_delivered", event_id=event.event_id, topic=self._topic)

    def close(self) -> None:
        self._client.close()


def _title(event: RecognitionEvent) -> str:
    room = event.camera_name.replace("_", " ")
    if event.verdict is EventVerdict.KNOWN_PERSON:
        return f"{event.identity_id} - {room}"
    if event.verdict is EventVerdict.ANIMAL:
        return f"{event.animal_label or 'animal'} - {room}"
    if event.verdict is EventVerdict.UNIDENTIFIED:
        return f"Someone unidentified - {room}"
    return f"Unknown person - {room}"


def _encode_header(value: str) -> str:
    """Make a header value safe to send.

    HTTP headers are ASCII. An identity called Jeremie is fine and one called
    Jérémie is not, and the failure is total: the request never leaves, so every
    notification for that person is lost rather than degraded. Non-ASCII values
    are wrapped as an RFC 2047 encoded word, which ntfy decodes back for display.
    """
    if value.isascii():
        return value
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"=?UTF-8?B?{encoded}?="


def _body(event: RecognitionEvent) -> str:
    when = event.started_at.strftime("%H:%M:%S")
    return f"{when}, seen for {event.duration_seconds:.0f}s over {event.observations} frames."


def _encode(snapshot: Frame | None) -> bytes | None:
    if snapshot is None:
        return None
    success, buffer = cv2.imencode(".jpg", snapshot, [cv2.IMWRITE_JPEG_QUALITY, _SNAPSHOT_QUALITY])
    return bytes(buffer) if success else None
