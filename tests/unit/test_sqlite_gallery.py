"""Durable storage for identities and their biometric descriptors."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from reconvision.adapters.storage.migrations import MIGRATIONS, apply_migrations
from reconvision.adapters.storage.sqlite_gallery import SqliteGallery, connect
from reconvision.domain.models import GalleryEntry, GalleryEntrySource, Identity
from reconvision.domain.ports import GalleryRepository
from tests.unit.conftest import random_embedding

JEREMIE = Identity("jeremie", "Jeremie")
GUEST = Identity("guest", "Guest")


@pytest.fixture
def gallery(tmp_path: Path) -> SqliteGallery:
    return SqliteGallery(connect(tmp_path / "reconvision.db"))


def test_the_repository_satisfies_its_port(gallery: SqliteGallery) -> None:
    port: GalleryRepository = gallery

    assert isinstance(port, GalleryRepository)


def test_a_database_is_created_and_migrated_on_first_use(tmp_path: Path) -> None:
    """A fresh install should not need a setup step to have a schema."""
    database = tmp_path / "nested" / "reconvision.db"

    connect(database)

    assert database.exists()


def test_migrations_run_once(tmp_path: Path) -> None:
    connection = connect(tmp_path / "reconvision.db")

    assert apply_migrations(connection) == 0


def test_migration_versions_are_unique_and_ordered() -> None:
    """Appended, never edited: an edited migration leaves two installs with
    different schemas and no way to tell them apart."""
    versions = [version for version, _, _ in MIGRATIONS]

    assert versions == sorted(versions)
    assert len(versions) == len(set(versions))


def test_an_embedding_survives_the_round_trip(
    gallery: SqliteGallery, rng: np.random.Generator
) -> None:
    """Stored as a raw blob, so a dtype mistake would silently corrupt every
    descriptor rather than fail."""
    embedding = random_embedding(rng)
    gallery.add_identity(JEREMIE)

    gallery.add_entry(GalleryEntry("jeremie", embedding))

    (stored,) = gallery.load_entries()
    assert stored.embedding.dtype == np.float32
    assert np.allclose(stored.embedding, embedding)


def test_the_source_of_an_entry_is_preserved(
    gallery: SqliteGallery, rng: np.random.Generator
) -> None:
    gallery.add_identity(JEREMIE)
    gallery.add_entry(
        GalleryEntry("jeremie", random_embedding(rng), source=GalleryEntrySource.CORRECTED_EVENT)
    )

    (stored,) = gallery.load_entries()
    assert stored.source is GalleryEntrySource.CORRECTED_EVENT


def test_re_enrolling_adds_photos_rather_than_replacing_the_person(
    gallery: SqliteGallery, rng: np.random.Generator
) -> None:
    """Enrolling more photographs later should extend the gallery, not reset it."""
    gallery.add_identity(JEREMIE)
    gallery.add_entry(GalleryEntry("jeremie", random_embedding(rng)))
    gallery.add_identity(Identity("jeremie", "Jeremie R."))
    gallery.add_entry(GalleryEntry("jeremie", random_embedding(rng)))

    assert gallery.count_entries("jeremie") == 2
    assert gallery.list_identities() == [Identity("jeremie", "Jeremie R.")]


def test_removing_a_person_removes_their_biometric_data(
    gallery: SqliteGallery, rng: np.random.Generator
) -> None:
    """Leaving face descriptors behind after someone asked to be forgotten would
    be the wrong default anywhere, and unacceptable in a home."""
    gallery.add_identity(JEREMIE)
    gallery.add_identity(GUEST)
    gallery.add_entry(GalleryEntry("jeremie", random_embedding(rng)))
    gallery.add_entry(GalleryEntry("guest", random_embedding(rng)))

    gallery.remove_identity("guest")

    assert gallery.count_entries() == 1
    assert [identity.identity_id for identity in gallery.list_identities()] == ["jeremie"]


def test_an_entry_cannot_reference_an_unknown_person(
    gallery: SqliteGallery, rng: np.random.Generator
) -> None:
    """Foreign keys are enforced, so a typo cannot create an orphan gallery that
    matches against a person who does not exist."""
    with pytest.raises(sqlite3.IntegrityError):
        gallery.add_entry(GalleryEntry("never_enrolled", random_embedding(rng)))


def test_the_gallery_persists_across_connections(tmp_path: Path, rng: np.random.Generator) -> None:
    """Enrol in one command, recognise in another: the whole point of storing it."""
    database = tmp_path / "reconvision.db"
    first = SqliteGallery(connect(database))
    first.add_identity(JEREMIE)
    first.add_entry(GalleryEntry("jeremie", random_embedding(rng)))
    first.close()

    reopened = SqliteGallery(connect(database))

    assert reopened.count_entries("jeremie") == 1


def test_an_empty_gallery_reports_nothing(gallery: SqliteGallery) -> None:
    assert gallery.load_entries() == []
    assert gallery.list_identities() == []
    assert gallery.count_entries() == 0
