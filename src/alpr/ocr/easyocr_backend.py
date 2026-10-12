"""OCR backend built on EasyOCR, a general-purpose scene-text reader."""

from __future__ import annotations

import string
import warnings

import cv2
import numpy as np

from alpr.ocr.fragments import Fragment, merge_fragments
from alpr.ocr.preprocess import build_preprocessor
from alpr.types import PlateRead

_ALLOWLIST = string.ascii_uppercase + string.digits


class EasyOcrBackend:
    """Reads plate crops with EasyOCR.

    Args:
        preprocess: Preprocessing applied to each crop; see
            :func:`alpr.ocr.preprocess.build_preprocessor`.
        detect_text: If true, EasyOCR first locates text regions on the crop
            and the serial is assembled from them, which lets small print be
            filtered out. If false, the whole crop is recognized as one line,
            which is faster and does not depend on the text detector finding
            anything on a tiny crop.
        allowlist: Restrict the recognizer's output to ``A-Z0-9``. It sounds
            like an obvious win for plates but is off by default: the
            recognizer then has to turn separators (dashes, dots, stars) into
            letters or digits, and a correct lowercase read becomes its
            nearest uppercase look-alike instead of the same letter. Without
            it those characters come out as they are and normalization
            handles them.
        gpu: Run on the GPU. ``None`` uses it when available.
    """

    def __init__(
        self,
        *,
        preprocess: str = "none",
        detect_text: bool = True,
        allowlist: bool = False,
        gpu: bool | None = None,
    ) -> None:
        import easyocr
        import torch

        self.name = ("easyocr" if detect_text else "easyocr-recognize") + (
            "-allowlist" if allowlist else ""
        )
        self.preprocess_name = preprocess
        self._preprocess = build_preprocessor(preprocess)
        self._detect_text = detect_text
        self._allowlist = _ALLOWLIST if allowlist else None
        use_gpu = torch.cuda.is_available() if gpu is None else gpu
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # quantize=False: EasyOCR quantizes its recognizer on CPU only,
            # which changes its outputs. Disabling it keeps results the same
            # whichever device they were computed on.
            self._reader = easyocr.Reader(["en"], gpu=use_gpu, quantize=False, verbose=False)

    def read(self, crop: np.ndarray) -> PlateRead | None:
        """Read the plate serial from a BGR crop.

        Returns:
            The normalized text and its confidence, or ``None`` if the crop is
            empty or no text was found.
        """
        if crop.size == 0:
            return None
        image = self._preprocess(crop)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._detect_text:
                raw = self._reader.readtext(image, allowlist=self._allowlist)
            else:
                grayscale = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                height, width = grayscale.shape[:2]
                raw = self._reader.recognize(
                    grayscale,
                    horizontal_list=[[0, width, 0, height]],
                    free_list=[],
                    allowlist=self._allowlist,
                )
        fragments = []
        for corners, text, confidence in raw:
            xs = [float(point[0]) for point in corners]
            ys = [float(point[1]) for point in corners]
            fragments.append(
                Fragment(
                    text=text,
                    confidence=float(confidence),
                    x=min(xs),
                    y=(min(ys) + max(ys)) / 2.0,
                    height=max(ys) - min(ys),
                )
            )
        return merge_fragments(fragments)
