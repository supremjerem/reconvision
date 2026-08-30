"""Converting model weights into the ONNX files the runtime consumes.

Development-time only. This module imports Ultralytics and therefore PyTorch,
roughly two gigabytes that the runtime container has no reason to carry, so
nothing under `adapters/` may import it. See ADR 0002.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

#: Input resolution the detector is exported at. 640 is what YOLO is trained on;
#: exporting at a smaller size trades away exactly the small-subject accuracy
#: that a wide-angle room camera needs.
DETECTOR_IMAGE_SIZE = 640

#: Filename recording what was exported and from what, so a stale or truncated
#: download is detected before it becomes a confusing inference error.
MANIFEST_NAME = "models.json"


@dataclass(frozen=True, slots=True)
class ExportedModel:
    """A model file the runtime can load."""

    name: str
    path: Path
    sha256: str
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def _digest(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a large model does not load into RAM."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_detector(
    models_dir: Path,
    weights: str = "yolo11s.pt",
    image_size: int = DETECTOR_IMAGE_SIZE,
) -> ExportedModel:
    """Download YOLO weights and convert them to ONNX.

    `yolo11s` rather than the nano variant: the detector only has to tell a person
    from a cat, but it also has to *find* them, and on a wide-angle room camera
    subjects are small. Missing the detection loses the event outright, whereas
    the extra milliseconds come out of a budget the face stage dominates anyway.
    """
    # Imported here rather than at module scope so that merely importing this
    # module does not pay for loading PyTorch.
    from ultralytics import YOLO

    models_dir.mkdir(parents=True, exist_ok=True)
    destination = models_dir / f"{Path(weights).stem}.onnx"

    model = YOLO(weights)
    exported = Path(model.export(format="onnx", imgsz=image_size, simplify=True, dynamic=False))

    if exported.resolve() != destination.resolve():
        shutil.move(str(exported), destination)

    # The downloaded .pt is only an export input and has no place in the data
    # directory that gets mounted into the container.
    downloaded_weights = Path(weights)
    if downloaded_weights.exists() and downloaded_weights.suffix == ".pt":
        downloaded_weights.unlink()

    return ExportedModel(
        name=destination.stem,
        path=destination,
        sha256=_digest(destination),
        size_bytes=destination.stat().st_size,
    )


def fetch_face_models(models_dir: Path, pack: str = "buffalo_l") -> ExportedModel:
    """Download the InsightFace pack, which already ships as ONNX.

    InsightFace defaults to caching under the user's home directory. Pointing it
    at the data directory instead keeps every model the system needs in the one
    volume that gets mounted into the container and backed up.
    """
    from insightface.app import FaceAnalysis

    models_dir.mkdir(parents=True, exist_ok=True)
    # Triggers the download as a side effect of preparing the model pack.
    FaceAnalysis(name=pack, root=str(models_dir), providers=["CPUExecutionProvider"])

    pack_dir = models_dir / "models" / pack
    if not pack_dir.exists():
        message = f"InsightFace did not produce {pack_dir}"
        raise RuntimeError(message)

    files = sorted(pack_dir.glob("*.onnx"))
    if not files:
        message = f"InsightFace pack {pack!r} contains no ONNX models"
        raise RuntimeError(message)

    # One digest over the whole pack: the individual files are only ever used
    # together, so their integrity is a single question.
    combined = hashlib.sha256()
    total_bytes = 0
    for file in files:
        combined.update(_digest(file).encode())
        total_bytes += file.stat().st_size

    return ExportedModel(
        name=pack,
        path=pack_dir,
        sha256=combined.hexdigest(),
        size_bytes=total_bytes,
    )


def write_manifest(models_dir: Path, models: list[ExportedModel]) -> Path:
    """Record what was exported, so the runtime can detect a stale download."""
    manifest = models_dir / MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                model.name: {
                    "path": str(model.path.relative_to(models_dir)),
                    "sha256": model.sha256,
                    "size_bytes": model.size_bytes,
                }
                for model in models
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
