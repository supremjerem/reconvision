"""Face detection, alignment and embedding through InsightFace.

The expensive stage, and the only one that produces an identity. It runs solely
on regions the object detector has already reported as people, which is what
keeps it affordable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import structlog
from insightface.app import FaceAnalysis

from reconvision.adapters.detection.onnx_yolo import select_providers
from reconvision.domain.models import BoundingBox, Face, FaceQuality, Frame

logger = structlog.get_logger(__name__)

#: Only what recognition needs. The pack also ships gender/age estimation and a
#: 2D landmark model, which cost memory and inference time for information this
#: system neither uses nor wants to be storing about the people in a home.
REQUIRED_MODULES = ("detection", "recognition", "landmark_3d_68")

#: Margin added around a person box before looking for a face inside it. The
#: detector's box can clip the top of the head, and a clipped crop costs the
#: alignment step the landmarks it needs.
_PERSON_BOX_MARGIN = 0.08


class InsightFaceAnalyzer:
    """Finds faces and turns them into comparable descriptors."""

    def __init__(
        self,
        models_dir: Path,
        pack: str = "buffalo_l",
        detection_size: int = 640,
        providers: Sequence[str] | None = None,
    ) -> None:
        pack_dir = models_dir / "models" / pack
        if not pack_dir.exists():
            message = f"Face models not found at {pack_dir}. Run `reconvision export-models` first."
            raise FileNotFoundError(message)

        self._app = FaceAnalysis(
            name=pack,
            root=str(models_dir),
            allowed_modules=list(REQUIRED_MODULES),
            providers=list(providers or select_providers()),
        )
        # ctx_id -1 selects CPU; the execution provider list above is what
        # actually decides where the work runs.
        self._app.prepare(ctx_id=-1, det_size=(detection_size, detection_size))

        logger.info("face_analyzer_loaded", pack=pack, detection_size=detection_size)

    def analyse(self, frame: Frame, region: BoundingBox | None = None) -> Sequence[Face]:
        """Find the faces in a frame, or within one person's box.

        Passing a region is the normal path: the object detector has already said
        where the person is, so re-scanning the whole frame would be paying twice
        for the same answer.
        """
        crop, offset_x, offset_y = self._crop_to(frame, region)
        if crop.size == 0:
            return []

        faces = [
            self._to_face(detected, crop, offset_x, offset_y) for detected in self._app.get(crop)
        ]

        if region is None:
            return faces

        # The margin exists to give the aligner context around a head the detector
        # clipped, not to widen the search. Without this filter the enlarged crop
        # picks up a bystander standing just behind, and their face is attributed
        # to this person's track - a wrong identity rather than a missing one.
        return [face for face in faces if region.contains_centre_of(face.box)]

    def _crop_to(self, frame: Frame, region: BoundingBox | None) -> tuple[Frame, float, float]:
        """Cut out the region of interest, returning it and its origin."""
        height, width = frame.shape[:2]
        if region is None:
            return frame, 0.0, 0.0

        margin_x = region.width * _PERSON_BOX_MARGIN
        margin_y = region.height * _PERSON_BOX_MARGIN
        left = int(max(0, region.left - margin_x))
        top = int(max(0, region.top - margin_y))
        right = int(min(width, region.right + margin_x))
        bottom = int(min(height, region.bottom + margin_y))

        if right <= left or bottom <= top:
            return frame[:0, :0], 0.0, 0.0

        return frame[top:bottom, left:right], float(left), float(top)

    def _to_face(self, detected: object, crop: Frame, offset_x: float, offset_y: float) -> Face:
        """Convert an InsightFace result into the domain's own representation."""
        bbox = np.asarray(detected.bbox, dtype=np.float32)  # type: ignore[attr-defined]
        box = BoundingBox(
            left=float(bbox[0]) + offset_x,
            top=float(bbox[1]) + offset_y,
            right=float(bbox[2]) + offset_x,
            bottom=float(bbox[3]) + offset_y,
        )

        pose = getattr(detected, "pose", None)
        # Pose is (pitch, yaw, roll). Without the landmark model there is no pose,
        # and reporting 0 would claim a perfectly frontal face; reporting the
        # limit instead makes the quality gate treat it as unverified.
        yaw = float(pose[1]) if pose is not None else 90.0

        return Face(
            box=box,
            embedding=np.asarray(detected.embedding, dtype=np.float32),  # type: ignore[attr-defined]
            quality=FaceQuality(
                pixel_height=round(box.height),
                sharpness=_sharpness(crop, bbox),
                yaw_degrees=yaw,
                detection_confidence=float(detected.det_score),  # type: ignore[attr-defined]
            ),
        )


def _sharpness(image: Frame, bbox: np.ndarray) -> float:
    """Variance of the Laplacian over the face region.

    A standard focus measure: the Laplacian responds to edges, and a blurred face
    has few of them. Computed on the face alone rather than the whole frame,
    because a sharp background behind a motion-blurred person would otherwise
    report the face as usable.
    """
    height, width = image.shape[:2]
    left = int(max(0, bbox[0]))
    top = int(max(0, bbox[1]))
    right = int(min(width, bbox[2]))
    bottom = int(min(height, bbox[3]))

    if right <= left or bottom <= top:
        return 0.0

    face = image[top:bottom, left:right]
    grey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())
