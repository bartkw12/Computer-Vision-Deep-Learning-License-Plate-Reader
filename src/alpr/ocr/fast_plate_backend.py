"""OCR backend built on fast-plate-ocr, a recognizer trained on license plates.

Unlike a scene-text reader it takes the whole plate crop and outputs the serial
directly, with one probability per character slot. It needs the optional
``ocr`` extra: ``pip install -e ".[ocr]"``.
"""

from __future__ import annotations

import cv2
import numpy as np

from alpr.ocr.preprocess import build_preprocessor
from alpr.plate_format import normalize
from alpr.types import PlateRead

DEFAULT_MODEL = "cct-xs-v2-global-model"


class FastPlateBackend:
    """Reads plate crops with a fast-plate-ocr model.

    Args:
        model: Name of a model from the fast-plate-ocr hub.
        preprocess: Preprocessing applied to each crop before it is converted
            to the color mode the model expects.
        gpu: Run on the GPU. ``None`` lets ONNX Runtime decide.
    """

    def __init__(
        self, *, model: str = DEFAULT_MODEL, preprocess: str = "none", gpu: bool | None = None
    ) -> None:
        try:
            from fast_plate_ocr import LicensePlateRecognizer
        except ImportError as error:
            raise ImportError(
                'the fast-plate-ocr backend needs the optional extra: pip install -e ".[ocr]"'
            ) from error

        self.name = f"fast-plate-ocr:{model}"
        self.preprocess_name = preprocess
        self._preprocess = build_preprocessor(preprocess)
        device = "auto" if gpu is None else ("cuda" if gpu else "cpu")
        self._recognizer = LicensePlateRecognizer(model, device=device)
        self._wants_gray = self._recognizer.config.image_color_mode == "grayscale"

    def read(self, crop: np.ndarray) -> PlateRead | None:
        """Read the plate serial from a BGR crop.

        Returns:
            The normalized text and the mean probability of its characters, or
            ``None`` if the crop is empty or the model output no characters.
        """
        if crop.size == 0:
            return None
        image = self._preprocess(crop)
        if self._wants_gray:
            image = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        prediction = self._recognizer.run(image, return_confidence=True)[0]
        text = normalize(prediction.plate)
        if not text:
            return None
        # One probability per output slot; the serial occupies the leading slots.
        probabilities = np.asarray(prediction.char_probs, dtype=float)[: len(prediction.plate)]
        return PlateRead(text, float(probabilities.mean()))
