"""Schema migrations, applied in order and recorded so they run once.

Deliberately a plain list of statements rather than a migration framework: the
schema is small, it only ever moves forward, and a NAS deployment should not need
an extra tool to start. Each entry is appended, never edited - editing one that
has already run means the schema differs between installs with no way to tell.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

#: (version, description, statements). Append only.
MIGRATIONS: Sequence[tuple[int, str, tuple[str, ...]]] = (
    (
        1,
        "identities and their face embeddings",
        (
            """
            CREATE TABLE identities (
                identity_id   TEXT PRIMARY KEY,
                display_name  TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE gallery_entries (
                entry_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_id  TEXT NOT NULL
                             REFERENCES identities(identity_id) ON DELETE CASCADE,
                embedding    BLOB NOT NULL,
                source       TEXT NOT NULL,
                captured_at  TEXT
            )
            """,
            "CREATE INDEX idx_gallery_identity ON gallery_entries(identity_id)",
        ),
    ),
)


def apply_migrations(connection: sqlite3.Connection) -> int:
    """Bring a database up to date, returning how many migrations ran."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "  version INTEGER PRIMARY KEY,"
        "  description TEXT NOT NULL,"
        "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )
    applied = {
        row[0] for row in connection.execute("SELECT version FROM schema_version").fetchall()
    }

    ran = 0
    for version, description, statements in MIGRATIONS:
        if version in applied:
            continue
        # One transaction per migration, so a failure half way leaves the database
        # at the previous version rather than in a state no migration describes.
        with connection:
            for statement in statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                (version, description),
            )
        ran += 1
    return ran
