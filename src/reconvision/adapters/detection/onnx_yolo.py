"""YOLO object detection through ONNX Runtime.

The cheap gate in front of the face stage: it answers "is there a person here, or
is this the cat", and only the first answer costs anything further. Runs on
ONNX Runtime alone, with no PyTorch in the process. See ADR 0002.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import structlog

from reconvision.domain.models import (
    SUBJECT_KIND_BY_LABEL,
    BoundingBox,
    Detection,
    Frame,
)

logger = structlog.get_logger(__name__)

#: Grey used to pad a letterboxed frame, matching Ultralytics' own preprocessing.
_PADDING_VALUE = 114


@dataclass(frozen=True, slots=True)
class DetectorPolicy:
    """Thresholds applied to raw detector output."""

    #: Kept deliberately low. A missed person loses the event outright, while a
    #: spurious box only costs one wasted face-detection pass that finds nothing.
    confidence_threshold: float = 0.35
    #: Overlap above which two boxes are treated as the same subject.
    nms_iou_threshold: float = 0.45

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold < 1.0:
            message = f"Confidence must be in (0, 1), got {self.confidence_threshold}"
            raise ValueError(message)
        if not 0.0 < self.nms_iou_threshold < 1.0:
            message = f"NMS IoU must be in (0, 1), got {self.nms_iou_threshold}"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class _Letterbox:
    """How a frame was fitted into the square model input, so boxes can be undone."""

    scale: float
    pad_x: float
    pad_y: float

    def to_original(self, boxes: np.ndarray) -> np.ndarray:
        """Map xyxy boxes from model space back to frame coordinates."""
        boxes[:, [0, 2]] -= self.pad_x
        boxes[:, [1, 3]] -= self.pad_y
        return boxes / self.scale


def select_providers(prefer_coreml: bool = True) -> list[str]:
    """Pick the best available ONNX Runtime backend.

    CoreML on Apple silicon during development, plain CPU in the Linux container.
    The same exported model runs on both, which is the point of using ONNX at all.
    """
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if prefer_coreml and "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


class OnnxYoloDetector:
    """Detects people and animals in a frame."""

    def __init__(
        self,
        model_path: Path,
        policy: DetectorPolicy | None = None,
        providers: Sequence[str] | None = None,
        labels_of_interest: frozenset[str] | None = None,
    ) -> None:
        if not model_path.exists():
            message = (
                f"Detector model not found at {model_path}. Run `reconvision export-models` first."
            )
            raise FileNotFoundError(message)

        self._policy = policy or DetectorPolicy()
        # Only classes the pipeline acts on. Discarding the other 72 before NMS
        # keeps the sofa and the potted plant from ever becoming an event.
        self._labels_of_interest = labels_of_interest or frozenset(SUBJECT_KIND_BY_LABEL)

        self._session = ort.InferenceSession(
            str(model_path), providers=list(providers or select_providers())
        )
        self._input_name = self._session.get_inputs()[0].name
        _, _, self._input_height, self._input_width = self._session.get_inputs()[0].shape
        self._class_names = self._read_class_names()

        logger.info(
            "detector_loaded",
            model=model_path.name,
            provider=self._session.get_providers()[0],
            input_size=f"{self._input_width}x{self._input_height}",
            classes_of_interest=sorted(self._labels_of_interest),
        )

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    def detect(self, frame: Frame) -> Sequence[Detection]:
        """Locate the people and animals in a frame."""
        model_input, letterbox = self._preprocess(frame)
        raw = self._session.run(None, {self._input_name: model_input})[0]
        height, width = frame.shape[:2]
        return self._postprocess(raw, letterbox, width, height)

    def _read_class_names(self) -> dict[int, str]:
        """Read the label map the exporter embedded in the model.

        Reading it from the model rather than hardcoding COCO means a detector
        exported with a different label set cannot silently mislabel everything.
        """
        metadata = self._session.get_modelmeta().custom_metadata_map
        raw = metadata.get("names")
        if not raw:
            message = "Detector model carries no class names; re-export it."
            raise ValueError(message)

        parsed = ast.literal_eval(raw)
        return {int(index): str(name) for index, name in parsed.items()}

    def _preprocess(self, frame: Frame) -> tuple[np.ndarray, _Letterbox]:
        """Fit the frame into the square model input without distorting it.

        Stretching to a square instead would compress a standing person
        horizontally, which is precisely the shape the detector was trained on.
        """
        height, width = frame.shape[:2]
        scale = min(self._input_width / width, self._input_height / height)
        scaled_width, scaled_height = round(width * scale), round(height * scale)

        resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self._input_height, self._input_width, 3), _PADDING_VALUE, dtype=np.uint8)
        pad_x = (self._input_width - scaled_width) / 2
        pad_y = (self._input_height - scaled_height) / 2
        top, left = int(pad_y), int(pad_x)
        canvas[top : top + scaled_height, left : left + scaled_width] = resized

        # BGR to RGB, HWC to CHW, 0-255 to 0-1, with a batch dimension.
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(blob)[None], _Letterbox(scale, pad_x, pad_y)

    def _postprocess(
        self, raw: np.ndarray, letterbox: _Letterbox, width: int, height: int
    ) -> Sequence[Detection]:
        """Turn the (1, 84, 8400) output into detections in frame coordinates."""
        # 8400 candidate boxes, each 4 box values followed by 80 class scores.
        predictions = np.squeeze(raw, axis=0).T
        class_scores = predictions[:, 4:]

        wanted_columns = [
            index for index, name in self._class_names.items() if name in self._labels_of_interest
        ]
        if not wanted_columns:
            return []

        # Restrict to classes the pipeline acts on before scoring, so the other 72
        # never reach NMS.
        restricted = class_scores[:, wanted_columns]
        best_local = restricted.argmax(axis=1)
        confidences = restricted[np.arange(restricted.shape[0]), best_local]

        keep = confidences >= self._policy.confidence_threshold
        if not np.any(keep):
            return []

        boxes = self._to_corner_boxes(predictions[keep, :4])
        confidences = confidences[keep]
        labels = [self._class_names[wanted_columns[index]] for index in best_local[keep]]

        surviving = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            self._policy.confidence_threshold,
            self._policy.nms_iou_threshold,
        )
        if len(surviving) == 0:
            return []

        indices = np.asarray(surviving).flatten()
        corners = letterbox.to_original(self._to_xyxy(boxes[indices]))

        return [
            Detection(
                box=BoundingBox(
                    left=float(np.clip(corner[0], 0, width)),
                    top=float(np.clip(corner[1], 0, height)),
                    right=float(np.clip(corner[2], 0, width)),
                    bottom=float(np.clip(corner[3], 0, height)),
                ),
                label=labels[index],
                confidence=float(confidences[index]),
            )
            for corner, index in zip(corners, indices, strict=True)
        ]

    @staticmethod
    def _to_corner_boxes(centre_boxes: np.ndarray) -> np.ndarray:
        """Centre-form (cx, cy, w, h) to the (x, y, w, h) OpenCV's NMS expects."""
        boxes = centre_boxes.copy()
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        return boxes

    @staticmethod
    def _to_xyxy(corner_boxes: np.ndarray) -> np.ndarray:
        """(x, y, w, h) to (left, top, right, bottom)."""
        boxes = corner_boxes.copy()
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        return boxes
