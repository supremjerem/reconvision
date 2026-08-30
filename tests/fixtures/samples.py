"""Sample photographs for the model tests.

Images cannot live in the repository: it is public, and the privacy guard blocks
them precisely so that no capture of anyone's home is ever committed. They are
treated like the model weights instead - fetched on demand into the git-ignored
data directory, and the tests that need them skip when they are absent.

The sources are public assets used by their own projects' documentation.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

SAMPLE_URLS: dict[str, str] = {
    # Ultralytics' own documentation samples: a wide street scene with several
    # people whose faces are far too small to identify, and a close portrait pair.
    "people.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
    # PyTorch Hub's example image, used here as the animal case.
    "dog.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
}

_DOWNLOAD_TIMEOUT_SECONDS = 30


def sample_path(name: str, samples_dir: Path) -> Path | None:
    """Return a local sample, fetching it once if needed.

    None means the sample could not be obtained, which is a reason to skip a test
    rather than to fail it: an offline machine is not a broken code change.
    """
    if name not in SAMPLE_URLS:
        message = f"Unknown sample {name!r}; known samples: {sorted(SAMPLE_URLS)}"
        raise KeyError(message)

    destination = samples_dir / name
    if destination.exists():
        return destination

    samples_dir.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(  # noqa: S310  # fixed https URLs, not user input
            SAMPLE_URLS[name], timeout=_DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None

    # A blocked download often returns an error page rather than failing outright.
    if not payload.startswith(b"\xff\xd8\xff"):
        return None

    destination.write_bytes(payload)
    return destination
