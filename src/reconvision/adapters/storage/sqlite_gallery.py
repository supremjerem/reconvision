"""Durable storage for enrolled identities and their embeddings.

SQLite in WAL mode, with embeddings held as raw float32 blobs. Matching loads
them all into one NumPy matrix at startup and never queries the database again:
at household scale that matrix is a few thousand vectors, where an exact
brute-force product beats any index. See ADR 0003.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import structlog

from reconvision.adapters.storage.migrations import apply_migrations
from reconvision.domain.models import Embedding, GalleryEntry, GalleryEntrySource, Identity

logger = structlog.get_logger(__name__)

#: Embeddings are stored as raw little-endian float32, the dtype ArcFace emits.
_EMBEDDING_DTYPE = np.float32


def connect(database_path: Path) -> sqlite3.Connection:
    """Open a database, creating and migrating it if necessary."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, check_same_thread=False)
    # WAL lets the reader that serves the web screens run while the pipeline
    # writes, which the default rollback journal would block.
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.row_factory = sqlite3.Row

    applied = apply_migrations(connection)
    if applied:
        logger.info("schema_migrated", database=str(database_path), migrations=applied)
    return connection


class SqliteGallery:
    """Enrolled identities and embeddings, persisted."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def list_identities(self) -> Sequence[Identity]:
        rows = self._connection.execute(
            "SELECT identity_id, display_name FROM identities ORDER BY display_name"
        ).fetchall()
        return [Identity(row["identity_id"], row["display_name"]) for row in rows]

    def add_identity(self, identity: Identity) -> None:
        """Insert an identity, or update the display name of an existing one.

        Re-enrolling adds photographs to a person rather than replacing them, so
        the identity row is upserted instead of conflicting.
        """
        with self._connection:
            self._connection.execute(
                "INSERT INTO identities (identity_id, display_name, created_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(identity_id) DO UPDATE SET display_name = excluded.display_name",
                (
                    identity.identity_id,
                    identity.display_name,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def remove_identity(self, identity_id: str) -> None:
        """Forget a person entirely.

        The cascade takes their embeddings with them: leaving biometric data
        behind after someone asked to be removed would be the wrong default in
        any system, and especially in one running in a home.
        """
        with self._connection:
            self._connection.execute("DELETE FROM identities WHERE identity_id = ?", (identity_id,))

    def load_entries(self) -> Sequence[GalleryEntry]:
        rows = self._connection.execute(
            "SELECT identity_id, embedding, source, captured_at FROM gallery_entries"
        ).fetchall()
        return [
            GalleryEntry(
                identity_id=row["identity_id"],
                embedding=_from_blob(row["embedding"]),
                source=GalleryEntrySource(row["source"]),
                captured_at=(
                    datetime.fromisoformat(row["captured_at"]) if row["captured_at"] else None
                ),
            )
            for row in rows
        ]

    def add_entry(self, entry: GalleryEntry) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO gallery_entries (identity_id, embedding, source, captured_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    entry.identity_id,
                    _to_blob(entry.embedding),
                    entry.source.value,
                    entry.captured_at.isoformat() if entry.captured_at else None,
                ),
            )

    def count_entries(self, identity_id: str | None = None) -> int:
        """How many embeddings are stored, in total or for one person."""
        if identity_id is None:
            row = self._connection.execute("SELECT COUNT(*) AS n FROM gallery_entries").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM gallery_entries WHERE identity_id = ?",
                (identity_id,),
            ).fetchone()
        return int(row["n"])

    def close(self) -> None:
        self._connection.close()


def _to_blob(embedding: Embedding) -> bytes:
    return np.ascontiguousarray(embedding, dtype=_EMBEDDING_DTYPE).tobytes()


def _from_blob(blob: bytes) -> Embedding:
    return np.frombuffer(blob, dtype=_EMBEDDING_DTYPE)
