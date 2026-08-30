"""The public dataset used to make a false-accept rate measurable at all."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from reconvision.adapters.faces.lfw import (
    LFW_ROOT_NAME,
    DatasetUnavailableError,
    ensure_lfw,
    iter_people,
)


def build_dataset(root: Path, people: dict[str, int]) -> Path:
    """A directory shaped like LFW: one folder per person, jpgs inside."""
    root.mkdir(parents=True, exist_ok=True)
    for name, photo_count in people.items():
        folder = root / name
        folder.mkdir()
        for index in range(photo_count):
            (folder / f"{name}_{index:04d}.jpg").write_bytes(b"\xff\xd8\xff")
    return root


def test_only_people_with_several_photos_are_used(tmp_path: Path) -> None:
    """One photograph contributes no same-person pair, and that distribution is
    half of what calibration measures."""
    root = build_dataset(tmp_path / "lfw", {"Anna": 5, "Ben": 1, "Cleo": 3})

    names = [person.identity_id for person in iter_people(root)]

    assert "Ben" not in names
    assert set(names) == {"Anna", "Cleo"}


def test_the_most_photographed_people_come_first(tmp_path: Path) -> None:
    """A capped run should still get a well-populated same-person distribution."""
    root = build_dataset(tmp_path / "lfw", {"Anna": 2, "Ben": 9, "Cleo": 5})

    names = [person.identity_id for person in iter_people(root)]

    assert names == ["Ben", "Cleo", "Anna"]


def test_the_number_of_people_can_be_capped(tmp_path: Path) -> None:
    """Encoding 5 749 identities takes far longer than choosing a threshold needs."""
    root = build_dataset(tmp_path / "lfw", {"Anna": 5, "Ben": 4, "Cleo": 3})

    assert len(list(iter_people(root, limit=2))) == 2


def test_each_person_carries_their_photos(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "lfw", {"Anna": 4})

    (anna,) = list(iter_people(root))

    assert len(anna.photos) == 4
    assert all(path.suffix == ".jpg" for path in anna.photos)


def test_an_already_unpacked_dataset_is_reused(tmp_path: Path) -> None:
    """A 243 MB download must not repeat on every calibration run."""
    build_dataset(tmp_path / LFW_ROOT_NAME, {"Anna": 2})

    assert ensure_lfw(tmp_path) == tmp_path / LFW_ROOT_NAME


def test_an_unreachable_dataset_reports_rather_than_hangs(tmp_path: Path) -> None:
    """Calibration should degrade to enrolled identities only, not crash, when the
    machine is offline."""
    with pytest.raises(DatasetUnavailableError, match="Could not download"):
        ensure_lfw(tmp_path, url="https://192.0.2.1/lfw.tgz", timeout_seconds=1.0)


def test_an_archive_missing_the_expected_directory_is_reported(tmp_path: Path) -> None:
    """A truncated or redirected download often unpacks into something else
    entirely; saying so beats an empty measurement."""
    stray = tmp_path / "stray.txt"
    stray.write_text("wrong contents")
    archive = tmp_path / "lfw-funneled.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(stray, arcname="something_else/stray.txt")

    with pytest.raises(DatasetUnavailableError, match="did not contain"):
        ensure_lfw(tmp_path)
