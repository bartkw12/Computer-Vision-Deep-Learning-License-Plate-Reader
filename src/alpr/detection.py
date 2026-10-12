"""Object detection with Ultralytics YOLO models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from alpr.types import Box, Detection

#: COCO class indices of the vehicle types that carry license plates.
VEHICLE_CLASSES: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def resolve_device(device: str | None = None) -> str:
    """Pick the inference device.

    Args:
        device: An explicit device such as ``"cpu"`` or ``"cuda:0"``, or
            ``None`` / ``"auto"`` to use the first GPU when one is available.
    """
    if device not in (None, "auto"):
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class Detector(Protocol):
    """Anything that finds boxes in images."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect objects in one BGR image."""
        ...

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Detect objects in several BGR images; one list per image."""
        ...


class YoloDetector:
    """A YOLO model behind the :class:`Detector` interface.

    Args:
        weights: Path to a ``.pt`` file. Official Ultralytics weights such as
            ``yolo11n.pt`` are downloaded to this path if missing.
        conf: Minimum detection confidence.
        iou: IoU threshold for non-maximum suppression.
        imgsz: Inference image size (longest side, pixels).
        classes: Class indices to keep; ``None`` keeps all.
        agnostic_nms: Suppress overlapping boxes across classes. Needed for
            vehicles: without it a pickup detected as both ``car`` and
            ``truck`` survives as two boxes and becomes two tracks.
        device: Inference device, see :func:`resolve_device`.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        classes: Sequence[int] | None = None,
        agnostic_nms: bool = False,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.weights = Path(weights)
        self._model = YOLO(str(weights))
        self._predict_args = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "classes": list(classes) if classes is not None else None,
            "agnostic_nms": agnostic_nms,
            "device": resolve_device(device),
            "verbose": False,
        }

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect objects in one BGR image."""
        return self.detect_batch([image])[0]

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Detect objects in several BGR images; one list per image."""
        if not images:
            return []
        results = self._model.predict(list(images), **self._predict_args)
        return [
            [
                Detection(Box(*(float(v) for v in row[:4])), float(row[4]), int(row[5]))
                for row in result.boxes.data.cpu().numpy()
            ]
            for result in results
        ]
