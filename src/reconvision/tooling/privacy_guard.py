#!/usr/bin/env python3
"""Pre-commit guard against leaking private data into this public repository.

Two failure modes matter here and neither produces an obvious error on its own:
committing a frame or a face embedding, and committing a camera URL that carries
credentials. Both are effectively irreversible once pushed, so they are blocked
before the commit rather than audited afterwards.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Media, databases and model weights never belong in the repository. Documentation
# assets are the sole exception: they are curated by hand and contain no capture.
BLOCKED_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".tiff",
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".ts",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".npy",
        ".npz",
        ".onnx",
        ".pt",
        ".pth",
    }
)
ALLOWED_PREFIXES = ("docs/", ".github/assets/")

# A stream URL is only dangerous once it carries a user:password pair.
CREDENTIALED_URL = re.compile(r"\b(?:rtsp|rtsps|http|https)://[^\s/@:]+:[^\s/@]+@", re.IGNORECASE)
# Example files exist to document the shape of a secret, so they are exempt.
CREDENTIAL_EXEMPT = ("cameras.example.yaml", ".env.example")


def _is_blocked_media(path: Path) -> bool:
    posix = path.as_posix()
    if posix.startswith(ALLOWED_PREFIXES):
        return False
    return path.suffix.lower() in BLOCKED_SUFFIXES


def _credential_lines(path: Path) -> list[tuple[int, str]]:
    if path.name in CREDENTIAL_EXEMPT:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return [
        (number, line.strip())
        for number, line in enumerate(text.splitlines(), start=1)
        if CREDENTIALED_URL.search(line)
    ]


def main(argv: list[str]) -> int:
    violations: list[str] = []

    for raw in argv:
        path = Path(raw)
        if not path.is_file():
            continue
        if _is_blocked_media(path):
            violations.append(
                f"{path}: media, database or model weights must stay in data/ (untracked)"
            )
        violations.extend(
            f"{path}:{number}: stream URL with embedded credentials -> move it to .env"
            for number, _ in _credential_lines(path)
        )

    if violations:
        print("Blocked: private data must not enter this public repository.", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
