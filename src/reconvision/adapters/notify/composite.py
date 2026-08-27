"""Delivering one event to every configured channel.

The rule every notifier here obeys: a delivery failure is logged, never raised. A
phone that is out of range, a broker that is restarting or a webhook that returns
500 must not stop the cameras being watched. Surveillance that halts because a
notification failed is worse than a notification that never arrives.
"""

from __future__ import annotations

from collections.abc import Sequence

import structlog

from reconvision.application.telemetry import Telemetry
from reconvision.domain.events import RecognitionEvent
from reconvision.domain.models import Frame
from reconvision.domain.ports import Notifier

logger = structlog.get_logger(__name__)


class CompositeNotifier:
    """Fans an event out to several channels.

    Each is attempted independently: one failing channel must not deprive the
    others of the event, which is the whole reason for configuring more than one.
    """

    def __init__(self, notifiers: Sequence[Notifier], telemetry: Telemetry | None = None) -> None:
        self._notifiers = list(notifiers)
        self._telemetry = telemetry

    @property
    def channel_count(self) -> int:
        return len(self._notifiers)

    def notify(self, event: RecognitionEvent, snapshot: Frame | None = None) -> None:
        for notifier in self._notifiers:
            channel = type(notifier).__name__
            try:
                notifier.notify(event, snapshot)
                self._record(channel, "delivered")
            except Exception:
                # Broad by intent. A notifier raising something unforeseen is a bug
                # in that notifier, and the correct response is to note it and keep
                # watching rather than to take the pipeline down with it.
                logger.exception("notification_failed", channel=channel, event_id=event.event_id)
                self._record(channel, "failed")

    def _record(self, channel: str, outcome: str) -> None:
        if self._telemetry is not None:
            self._telemetry.metrics.notifications.add(1, {"channel": channel, "outcome": outcome})
