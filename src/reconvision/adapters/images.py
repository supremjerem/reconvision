"""Reading image files.

One place holding the cast from OpenCV's loose return type to the domain's frame
type, and one place deciding that an unreadable file is a None rather than an
exception - a folder of enrolment photographs routinely contains a stray file,
and that is not an error worth stopping for.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2

from reconvision.domain.models import Frame


def read_image(path: Path) -> Frame | None:
    """Load an image, or None if it cannot be decoded."""
    image = cv2.imread(str(path))
    return None if image is None else cast("Frame", image)
