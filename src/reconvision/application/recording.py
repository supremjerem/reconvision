"""What happens to an event once the pipeline has decided it.

Ordered so the most durable step comes first. The snapshot and the row are written
before anything is delivered, because a notification that arrives with no record
behind it leaves the user nothing to review, whereas a record with a failed
notification is merely quiet.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import structlog

from reconvision.application.pipeline import ObservedEvent
from reconvision.domain.events import RecognitionEvent
from reconvision.domain.ports import Clock, EventRepository, Notifier, SnapshotStore

logger = structlog.get_logger(__name__)


class EventRecorder:
    """Persists an event, then delivers it."""

    def __init__(
        self,
        events: EventRepository,
        snapshots: SnapshotStore,
        notifier: Notifier,
        clock: Clock,
        retention_days: int = 30,
    ) -> None:
        self._events = events
        self._snapshots = snapshots
        self._notifier = notifier
        self._clock = clock
        self._retention_days = retention_days

    def record(self, observed: ObservedEvent) -> RecognitionEvent:
        """Store the event and notify, returning the event as it was stored."""
        event = observed.event

        if observed.snapshot is not None:
            snapshot_id = self._snapshots.save(observed.snapshot, event.event_id)
            event = replace(event, snapshot_id=snapshot_id)

        self._events.save(event)

        if event.is_noteworthy:
            # Only what is worth interrupting someone for. A recognised household
            # member crossing their own hallway is recorded, not announced.
            self._notifier.notify(event, observed.snapshot)

        return event

    def purge_expired(self) -> tuple[int, int]:
        """Apply the retention policy, returning (events, snapshots) removed.

        Retention is enforced rather than offered: a camera on a hallway builds a
        record of everyone who lives there, and keeping that forever should not be
        the consequence of nobody having chosen otherwise.
        """
        cutoff = self._clock.now() - timedelta(days=self._retention_days)
        removed_events = self._events.delete_older_than(cutoff)
        removed_snapshots = self._snapshots.purge_older_than(cutoff)

        if removed_events or removed_snapshots:
            logger.info(
                "retention_applied",
                events=removed_events,
                snapshots=removed_snapshots,
                retention_days=self._retention_days,
            )
        return removed_events, removed_snapshots
