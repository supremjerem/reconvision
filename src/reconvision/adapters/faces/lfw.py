"""Labeled Faces in the Wild, used as the impostor set for calibration.

Calibration needs many pairs of different people to measure a false-accept rate
at all: with only the two or three people in a household, a rate of one in a
thousand cannot be observed, let alone estimated. LFW supplies roughly 13 000
photographs of 5 749 people, which makes the figure meaningful.

Downloaded on demand into the data directory, never into the repository. It is a
public research dataset of public figures, but it is still 13 000 photographs of
real people, and it has no business in anyone's git history.
"""

from __future__ import annotations

import tarfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

LFW_URL = "https://ndownloader.figshare.com/files/5976015"
LFW_ARCHIVE_NAME = "lfw-funneled.tgz"
#: The archive unpacks to one directory per person.
LFW_ROOT_NAME = "lfw_funneled"

#: Applies to establishing the connection and to each read, not the whole transfer.
_DOWNLOAD_TIMEOUT_SECONDS = 30


class DatasetUnavailableError(RuntimeError):
    """The dataset could not be downloaded or unpacked."""


@dataclass(frozen=True, slots=True)
class PersonPhotos:
    """One identity and the photographs available for them."""

    identity_id: str
    photos: list[Path]


def ensure_lfw(
    data_dir: Path, url: str = LFW_URL, timeout_seconds: float = _DOWNLOAD_TIMEOUT_SECONDS
) -> Path:
    """Return the unpacked LFW directory, downloading it once if needed."""
    root = data_dir / LFW_ROOT_NAME
    if root.is_dir() and any(root.iterdir()):
        return root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / LFW_ARCHIVE_NAME

    if not archive.exists():
        logger.info("downloading_lfw", url=url, destination=str(archive))
        try:
            _download(url, archive, timeout_seconds)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            archive.unlink(missing_ok=True)
            message = f"Could not download LFW from {url}: {error}"
            raise DatasetUnavailableError(message) from error

    logger.info("unpacking_lfw", archive=str(archive))
    try:
        with tarfile.open(archive) as tar:
            # `data` filter rejects absolute paths, parent traversal and special
            # files. Extracting a downloaded archive without it is how a tarball
            # writes outside the directory it was meant to.
            tar.extractall(path=data_dir, filter="data")
    except (tarfile.TarError, OSError) as error:
        message = f"Could not unpack {archive}: {error}"
        raise DatasetUnavailableError(message) from error

    if not root.is_dir():
        message = f"{archive} did not contain {LFW_ROOT_NAME}"
        raise DatasetUnavailableError(message)
    return root


def iter_people(
    root: Path, minimum_photos: int = 2, limit: int | None = None
) -> Iterator[PersonPhotos]:
    """Yield people from the dataset, most-photographed first.

    Only people with at least two photographs are useful: a single photograph
    contributes no genuine pair, and the genuine distribution is half of what is
    being measured. Taking the most-photographed first means a capped run still
    gets a well-populated genuine distribution.
    """
    people = [
        PersonPhotos(identity_id=folder.name, photos=sorted(folder.glob("*.jpg")))
        for folder in sorted(root.iterdir())
        if folder.is_dir()
    ]
    eligible = [person for person in people if len(person.photos) >= minimum_photos]
    eligible.sort(key=lambda person: len(person.photos), reverse=True)

    yield from eligible[:limit] if limit is not None else eligible


def _download(url: str, destination: Path, timeout_seconds: float) -> None:
    """Stream a download to disk under a timeout.

    `urlretrieve` takes no timeout, so an unresponsive host blocks indefinitely -
    which for a 243 MB download over a home connection is indistinguishable from
    normal slowness until someone notices hours later. Streaming in chunks also
    keeps the archive off the heap.
    """
    partial = destination.with_suffix(destination.suffix + ".part")
    with (
        urllib.request.urlopen(url, timeout=timeout_seconds) as response,  # noqa: S310
        partial.open("wb") as handle,
    ):
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)

    # Renamed only once complete, so an interrupted download is never mistaken
    # for a usable archive on the next run.
    partial.replace(destination)
