"""OCR backends that turn a plate crop into text.

Backends are interchangeable behind :class:`PlateOCR` so they can be compared
on the same crops. Heavy dependencies are imported only when a backend is
built.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from alpr.types import PlateRead

#: Backend names accepted by :func:`build_ocr`.
BACKENDS = (
    "easyocr",
    "easyocr-allowlist",
    "easyocr-recognize",
    "easyocr-recognize-allowlist",
    "fast-plate-ocr",
)


class PlateOCR(Protocol):
    """Anything that reads text from a plate crop."""

    name: str
    preprocess_name: str

    def read(self, crop: np.ndarray) -> PlateRead | None:
        """Read a BGR plate crop; ``None`` if nothing was read."""
        ...


def build_ocr(
    backend: str = "easyocr",
    *,
    preprocess: str = "none",
    gpu: bool | None = None,
    model: str | None = None,
) -> PlateOCR:
    """Create an OCR backend by name.

    Args:
        backend: ``"easyocr"`` (locate text on the crop, then recognize),
            ``"easyocr-recognize"`` (recognize the whole crop as one line) or
            ``"fast-plate-ocr"`` (plate-specific recognizer). The EasyOCR
            names take an ``-allowlist`` suffix to restrict output to
            ``A-Z0-9``.
        preprocess: Preprocessing applied to each crop.
        gpu: Run on the GPU; ``None`` uses it when available.
        model: Model name for backends that offer several.

    Raises:
        ValueError: If the backend name is unknown.
    """
    if backend.startswith("easyocr") and backend in BACKENDS:
        from alpr.ocr.easyocr_backend import EasyOcrBackend

        return EasyOcrBackend(
            preprocess=preprocess,
            detect_text="recognize" not in backend,
            allowlist=backend.endswith("-allowlist"),
            gpu=gpu,
        )
    if backend == "fast-plate-ocr":
        from alpr.ocr.fast_plate_backend import DEFAULT_MODEL, FastPlateBackend

        return FastPlateBackend(model=model or DEFAULT_MODEL, preprocess=preprocess, gpu=gpu)
    raise ValueError(f"unknown OCR backend {backend!r}; choose from {list(BACKENDS)}")
