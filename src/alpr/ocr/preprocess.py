"""Image preprocessing applied to a plate crop before OCR.

Each step is a named function from a BGR or grayscale image to another image.
They are kept separate and selectable by name so that their effect on reading
accuracy can be measured rather than assumed.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

Preprocessor = Callable[[np.ndarray], np.ndarray]

#: Height that ``upscale`` brings small crops up to, in pixels.
UPSCALE_MIN_HEIGHT = 64


def _to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def identity(image: np.ndarray) -> np.ndarray:
    """Leave the crop as it is."""
    return image


def gray(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale."""
    return _to_gray(image)


def otsu_inv(image: np.ndarray) -> np.ndarray:
    """Binarize with Otsu's threshold and invert, giving white text on black.

    This is the preprocessing the first version of this project used. Otsu's
    method picks one global threshold from the histogram; it is not adaptive.
    """
    _, binary = cv2.threshold(_to_gray(image), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary


def clahe(image: np.ndarray) -> np.ndarray:
    """Equalize local contrast (CLAHE) on the grayscale crop."""
    equalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return equalizer.apply(_to_gray(image))


def upscale(image: np.ndarray) -> np.ndarray:
    """Enlarge crops shorter than ``UPSCALE_MIN_HEIGHT`` with bicubic interpolation."""
    height = image.shape[0]
    if height >= UPSCALE_MIN_HEIGHT:
        return image
    factor = UPSCALE_MIN_HEIGHT / height
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


PREPROCESSORS: dict[str, Preprocessor] = {
    "none": identity,
    "gray": gray,
    "otsu_inv": otsu_inv,
    "clahe": clahe,
    "upscale": upscale,
}


def build_preprocessor(spec: str) -> Preprocessor:
    """Build a preprocessing function from its name.

    Args:
        spec: A name from ``PREPROCESSORS``, or several joined with ``+`` to
            apply them left to right, e.g. ``"upscale+clahe"``.

    Raises:
        ValueError: If any step name is unknown.
    """
    names = [name.strip() for name in spec.split("+")]
    unknown = [name for name in names if name not in PREPROCESSORS]
    if unknown:
        raise ValueError(
            f"unknown preprocessing step(s) {unknown}; choose from {sorted(PREPROCESSORS)}"
        )
    steps = [PREPROCESSORS[name] for name in names]

    def apply(image: np.ndarray) -> np.ndarray:
        for step in steps:
            image = step(image)
        return image

    return apply
