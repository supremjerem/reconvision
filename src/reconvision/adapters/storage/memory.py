"""In-memory storage.

Useful in its own right, not only as a stand-in: `reconvision run` against a
recorded clip wants recognition without leaving a database behind, and the tests
want the repository contract without SQLite. The durable implementation arrives
alongside event persistence.
"""

from __future__ import annotations

from collections.abc import Sequence

from reconvision.domain.models import GalleryEntry, Identity


class InMemoryGallery:
    """Enrolled identities and embeddings, held for the life of the process."""

    def __init__(self, entries: Sequence[GalleryEntry] = ()) -> None:
        self._identities: dict[str, Identity] = {}
        self._entries: list[GalleryEntry] = list(entries)

    def list_identities(self) -> Sequence[Identity]:
        return list(self._identities.values())

    def add_identity(self, identity: Identity) -> None:
        self._identities[identity.identity_id] = identity

    def remove_identity(self, identity_id: str) -> None:
        self._identities.pop(identity_id, None)
        self._entries = [entry for entry in self._entries if entry.identity_id != identity_id]

    def load_entries(self) -> Sequence[GalleryEntry]:
        return list(self._entries)

    def add_entry(self, entry: GalleryEntry) -> None:
        self._entries.append(entry)
