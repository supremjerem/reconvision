"""Durable storage for recognition events and the corrections made against them."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime

import structlog

from reconvision.domain.events import (
    EventFeedback,
    EventVerdict,
    FeedbackLabel,
    RecognitionEvent,
)
from reconvision.domain.models import SubjectKind

logger = structlog.get_logger(__name__)


class SqliteEvents:
    """Recognition events, persisted."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def save(self, event: RecognitionEvent) -> None:
        """Store an event, replacing any earlier version of it.

        Replacing rather than failing: an event is written when its subject leaves,
        and a restart mid-passage can produce the same id twice. Losing the second
        write would silently discard the more complete record.
        """
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO events ("
                "  event_id, camera_name, verdict, subject_kind, identity_id, animal_label,"
                "  started_at, ended_at, confidence, best_similarity, observations, snapshot_id"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.camera_name,
                    event.verdict.value,
                    event.subject_kind.value,
                    event.identity_id,
                    event.animal_label,
                    event.started_at.isoformat(),
                    event.ended_at.isoformat(),
                    event.confidence,
                    event.best_similarity,
                    event.observations,
                    event.snapshot_id,
                ),
            )

    def get(self, event_id: str) -> RecognitionEvent | None:
        row = self._connection.execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return _to_event(row) if row is not None else None

    def list_recent(
        self,
        limit: int = 50,
        camera_name: str | None = None,
        since: datetime | None = None,
    ) -> Sequence[RecognitionEvent]:
        clauses: list[str] = []
        parameters: list[object] = []
        if camera_name is not None:
            clauses.append("camera_name = ?")
            parameters.append(camera_name)
        if since is not None:
            clauses.append("started_at >= ?")
            parameters.append(since.isoformat())

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)

        rows = self._connection.execute(
            f"SELECT * FROM events {where} ORDER BY started_at DESC LIMIT ?",  # noqa: S608
            parameters,
        ).fetchall()
        return [_to_event(row) for row in rows]

    def save_feedback(self, feedback: EventFeedback) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO event_feedback "
                "(event_id, label, corrected_identity_id, submitted_at) VALUES (?, ?, ?, ?)",
                (
                    feedback.event_id,
                    feedback.label.value,
                    feedback.corrected_identity_id,
                    feedback.submitted_at.isoformat(),
                ),
            )

    def list_feedback(self) -> Sequence[EventFeedback]:
        rows = self._connection.execute(
            "SELECT event_id, label, corrected_identity_id, submitted_at "
            "FROM event_feedback ORDER BY submitted_at DESC"
        ).fetchall()
        return [
            EventFeedback(
                event_id=row["event_id"],
                label=FeedbackLabel(row["label"]),
                corrected_identity_id=row["corrected_identity_id"],
                submitted_at=datetime.fromisoformat(row["submitted_at"]),
            )
            for row in rows
        ]

    def delete_older_than(self, cutoff: datetime) -> int:
        """Drop events that have aged out, returning how many went.

        Retention is enforced rather than offered. A camera pointed at a hallway
        accumulates a record of everyone who lives there, and keeping it forever by
        default is not a decision to leave to inertia.
        """
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM events WHERE ended_at < ?", (cutoff.isoformat(),)
            )
        return cursor.rowcount

    def anonymise_identity(self, identity_id: str) -> int:
        """Strip a person's name from their past events, returning how many changed.

        Forgetting someone cannot simply null the name: an event that still claims
        to be a known person with nobody attached is a state the domain refuses to
        represent, and rightly so. The truthful transformation is that their past
        passages become an unknown person - the record of activity survives, the
        identification does not.

        Deleting the events instead would erase the history of what happened in the
        house, which is a different and larger decision than forgetting a face.
        """
        with self._connection:
            cursor = self._connection.execute(
                "UPDATE events SET identity_id = NULL, verdict = ? WHERE identity_id = ?",
                (EventVerdict.UNKNOWN_PERSON.value, identity_id),
            )
        return cursor.rowcount

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS n FROM events").fetchone()
        return int(row["n"])


def _to_event(row: sqlite3.Row) -> RecognitionEvent:
    return RecognitionEvent(
        event_id=row["event_id"],
        camera_name=row["camera_name"],
        verdict=EventVerdict(row["verdict"]),
        subject_kind=SubjectKind(row["subject_kind"]),
        identity_id=row["identity_id"],
        animal_label=row["animal_label"],
        started_at=datetime.fromisoformat(row["started_at"]),
        ended_at=datetime.fromisoformat(row["ended_at"]),
        confidence=row["confidence"],
        best_similarity=row["best_similarity"],
        observations=row["observations"],
        snapshot_id=row["snapshot_id"],
    )
