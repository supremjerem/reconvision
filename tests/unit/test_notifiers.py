"""Notification is the product: a dashboard is something you have to remember to open.

The rule every notifier obeys is that delivery failure never stops the cameras.
Surveillance that halts because a phone was out of range is worse than a
notification that never arrives, so most of these tests are about failure.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pytest

from reconvision.adapters.notify.composite import CompositeNotifier
from reconvision.adapters.notify.ntfy import NtfyNotifier
from reconvision.adapters.notify.webhook import WebhookNotifier
from reconvision.application.telemetry import configure_logging
from reconvision.domain.events import EventVerdict, RecognitionEvent
from reconvision.domain.models import Frame, SubjectKind
from reconvision.domain.ports import Notifier
from tests.fakes import RecordingNotifier

START = datetime(2026, 8, 27, 3, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def logging_configured() -> None:
    """Exercise these notifiers with logging actually switched on.

    Their success paths log at debug level, and structlog reserves `event` as the
    message keyword: passing an event id under that name raises rather than logs.
    With logging left unconfigured the calls are no-ops and the collision hides
    until production, where it breaks every successful delivery.
    """
    configure_logging("DEBUG")


def event(**overrides: object) -> RecognitionEvent:
    defaults: dict[str, object] = {
        "camera_name": "living_room",
        "verdict": EventVerdict.UNKNOWN_PERSON,
        "started_at": START,
        "ended_at": START + timedelta(seconds=6),
        "subject_kind": SubjectKind.PERSON,
        "observations": 18,
    }
    defaults.update(overrides)
    return RecognitionEvent(**defaults)  # type: ignore[arg-type]


def snapshot() -> Frame:
    return np.full((120, 160, 3), 90, dtype=np.uint8)


class ExplodingNotifier:
    """A channel that fails every time, as an unreachable broker would."""

    def notify(self, event: RecognitionEvent, snapshot: Frame | None = None) -> None:
        message = "broker unreachable"
        raise ConnectionError(message)


# --- fan-out --------------------------------------------------------------------


def test_an_event_reaches_every_channel() -> None:
    first, second = RecordingNotifier(), RecordingNotifier()

    CompositeNotifier([first, second]).notify(event())

    assert len(first.delivered) == 1
    assert len(second.delivered) == 1


def test_one_failing_channel_does_not_deprive_the_others() -> None:
    """The reason for configuring more than one channel in the first place."""
    working = RecordingNotifier()

    CompositeNotifier([ExplodingNotifier(), working, ExplodingNotifier()]).notify(event())

    assert len(working.delivered) == 1


def test_a_failing_channel_never_stops_the_cameras() -> None:
    """A notifier raising must not propagate: the pipeline would stop watching."""
    CompositeNotifier([ExplodingNotifier()]).notify(event())


def test_the_composite_satisfies_the_notifier_port() -> None:
    notifier: Notifier = CompositeNotifier([])

    assert isinstance(notifier, Notifier)


def test_no_configured_channels_is_not_an_error() -> None:
    """A fresh install has none, and that is a valid way to run."""
    composite = CompositeNotifier([])

    composite.notify(event())

    assert composite.channel_count == 0


# --- ntfy -----------------------------------------------------------------------


def ntfy_with(status: int = 200) -> tuple[NtfyNotifier, list[httpx.Request]]:
    """An ntfy notifier whose requests are captured instead of sent."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status)

    client = httpx.Client(transport=httpx.MockTransport(record))
    return NtfyNotifier(topic="my-topic", client=client), seen


def test_an_unknown_person_is_sent_at_high_priority() -> None:
    """It should break through a silenced phone at three in the morning."""
    notifier, seen = ntfy_with()

    notifier.notify(event(verdict=EventVerdict.UNKNOWN_PERSON))

    assert seen[0].headers["Priority"] == "high"
    assert "Unknown person" in seen[0].headers["Title"]


def test_a_recognised_household_member_is_sent_quietly() -> None:
    """Otherwise the system cries wolf every time you walk to the kitchen."""
    notifier, seen = ntfy_with()

    notifier.notify(event(verdict=EventVerdict.KNOWN_PERSON, identity_id="jeremie"))

    assert seen[0].headers["Priority"] == "low"
    assert "jeremie" in seen[0].headers["Title"]


def test_the_camera_is_named_in_the_title() -> None:
    """Read on a lock screen, the room matters more than anything else."""
    notifier, seen = ntfy_with()

    notifier.notify(event())

    assert "living room" in seen[0].headers["Title"]


def test_the_snapshot_is_attached_to_the_notification() -> None:
    """A picture in the notification itself, rather than a link to open later."""
    notifier, seen = ntfy_with()

    notifier.notify(event(), snapshot())

    assert seen[0].method == "PUT"
    assert seen[0].headers["Filename"].endswith(".jpg")
    assert seen[0].content.startswith(b"\xff\xd8\xff")


def test_an_event_without_a_snapshot_still_sends() -> None:
    notifier, seen = ntfy_with()

    notifier.notify(event(), snapshot=None)

    assert seen[0].method == "POST"
    assert b"seen for" in seen[0].content


def test_a_rejected_delivery_is_raised_for_the_composite_to_absorb() -> None:
    """Each notifier reports its own failure; only the composite decides to
    swallow it, so a single-channel setup still logs the problem."""
    notifier, _ = ntfy_with(status=503)

    with pytest.raises(httpx.HTTPStatusError):
        notifier.notify(event())


def test_a_topic_is_required() -> None:
    with pytest.raises(ValueError, match="topic is required"):
        NtfyNotifier(topic="")


# --- webhook --------------------------------------------------------------------


def webhook_with(status: int = 200) -> tuple[WebhookNotifier, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status)

    client = httpx.Client(transport=httpx.MockTransport(record))
    return WebhookNotifier(url="https://example.test/hook", client=client), seen


def test_the_webhook_describes_the_event_as_json() -> None:
    notifier, seen = webhook_with()

    notifier.notify(event(verdict=EventVerdict.ANIMAL, animal_label="cat"))

    payload = json.loads(seen[0].content)
    assert payload["verdict"] == "animal"
    assert payload["animal"] == "cat"
    assert payload["camera"] == "living_room"


def test_the_webhook_carries_the_snapshot_inline() -> None:
    """Base64 inside the JSON: any endpoint accepting JSON can read it, without a
    second request shape for every consumer to handle."""
    notifier, seen = webhook_with()

    notifier.notify(event(), snapshot())

    payload = json.loads(seen[0].content)
    assert payload["snapshot_jpeg_base64"]


def test_the_snapshot_can_be_left_out() -> None:
    """Some endpoints reject large bodies, and an automation rarely needs the image."""
    client = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200)))
    notifier = WebhookNotifier("https://example.test/hook", include_snapshot=False, client=client)
    seen: list[dict[str, object]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200)

    notifier._client = httpx.Client(transport=httpx.MockTransport(capture))
    notifier.notify(event(), snapshot())

    assert "snapshot_jpeg_base64" not in seen[0]


def test_a_url_is_required() -> None:
    with pytest.raises(ValueError, match="URL is required"):
        WebhookNotifier(url="")
