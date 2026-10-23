import cv2
import numpy as np
import pytest

from alpr.ocr import BACKENDS, build_ocr
from alpr.ocr.cache import CachedOCR
from alpr.ocr.fragments import Fragment, merge_fragments
from alpr.ocr.preprocess import PREPROCESSORS, UPSCALE_MIN_HEIGHT, build_preprocessor
from alpr.types import PlateRead


def rendered_plate(text="BXYZ 123", height=120, width=420):
    image = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(image, text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6, cv2.LINE_AA)
    return image


# --- preprocessing -----------------------------------------------------------


def test_none_leaves_the_crop_untouched():
    crop = rendered_plate()
    assert build_preprocessor("none")(crop) is crop


@pytest.mark.parametrize("name", ["gray", "clahe", "otsu_inv"])
def test_single_channel_steps_keep_the_size(name):
    crop = rendered_plate()
    out = build_preprocessor(name)(crop)
    assert out.shape == crop.shape[:2]
    assert out.dtype == np.uint8


def test_otsu_inv_gives_white_text_on_black():
    out = build_preprocessor("otsu_inv")(rendered_plate())
    assert set(np.unique(out)) == {0, 255}
    assert out[0, 0] == 0  # the white background became black
    assert (out == 255).mean() < 0.5  # text is the minority


def test_upscale_enlarges_small_crops_only():
    small = cv2.resize(rendered_plate(), (70, 20))
    enlarged = build_preprocessor("upscale")(small)
    assert enlarged.shape[0] == UPSCALE_MIN_HEIGHT
    assert enlarged.shape[1] / enlarged.shape[0] == pytest.approx(70 / 20, rel=0.02)
    big = rendered_plate()
    assert build_preprocessor("upscale")(big) is big


def test_steps_compose_left_to_right():
    small = cv2.resize(rendered_plate(), (70, 20))
    out = build_preprocessor("upscale+clahe")(small)
    assert out.ndim == 2 and out.shape[0] == UPSCALE_MIN_HEIGHT


@pytest.mark.parametrize("name", sorted(PREPROCESSORS))
def test_every_step_accepts_grayscale_input(name):
    gray = cv2.cvtColor(rendered_plate(), cv2.COLOR_BGR2GRAY)
    assert build_preprocessor(name)(gray).shape[:2] == gray.shape


def test_unknown_preprocessing_step():
    with pytest.raises(ValueError, match="unknown preprocessing"):
        build_preprocessor("gray+sharpen")


# --- fragment merging ---------------------------------------------------------


def fragment(text, x, y=50.0, height=40.0, confidence=0.9):
    return Fragment(text=text, confidence=confidence, x=x, y=y, height=height)


def test_single_fragment_is_normalized():
    assert merge_fragments([fragment("bxyz-123", 10)]) == PlateRead("BXYZ123", 0.9)


def test_split_serial_is_joined_left_to_right():
    """Regression: the original returned the first fragment that matched on its own.

    EasyOCR routinely reports ``ABCD`` and ``123`` separately; neither is a plate.
    """
    halves = [fragment("123", 200), fragment("BXYZ", 10)]
    assert merge_fragments(halves).text == "BXYZ123"
    assert merge_fragments(halves[::-1]).text == "BXYZ123"


def test_small_print_is_ignored():
    fragments = [
        fragment("ONTARIO", 60, y=10, height=12),
        fragment("BXYZ", 10, y=50, height=40),
        fragment("123", 200, y=50, height=38),
        fragment("YOURS TO DISCOVER", 30, y=90, height=10),
    ]
    assert merge_fragments(fragments).text == "BXYZ123"


def test_confidence_is_weighted_by_length():
    read = merge_fragments(
        [fragment("BXYZ", 10, confidence=1.0), fragment("123", 200, confidence=0.3)]
    )
    assert read.confidence == pytest.approx((4 * 1.0 + 3 * 0.3) / 7)


def test_two_row_plates_read_top_row_first():
    rows = [fragment("1234", 5, y=70), fragment("AB", 20, y=20)]
    assert merge_fragments(rows).text == "AB1234"


def test_nothing_readable():
    assert merge_fragments([]) is None
    assert merge_fragments([fragment("--", 10), fragment(" . ", 50)]) is None


# --- caching -------------------------------------------------------------------


class CountingBackend:
    name = "fake"
    preprocess_name = "none"

    def __init__(self):
        self.calls = 0

    def read(self, crop):
        self.calls += 1
        return None if crop.mean() == 0 else PlateRead("BXYZ123", 0.75)


def test_cache_calls_the_backend_once_per_distinct_crop(tmp_path):
    backend = CountingBackend()
    cache = CachedOCR(backend, tmp_path / "ocr.json")
    crop, blank = rendered_plate(), np.zeros((20, 60, 3), np.uint8)

    assert cache.read(crop) == cache.read(crop) == PlateRead("BXYZ123", 0.75)
    assert cache.read(blank) is None and cache.read(blank) is None  # misses are cached too
    assert backend.calls == 2
    assert (cache.hits, cache.misses) == (2, 2)


def test_cache_persists_between_sessions(tmp_path):
    path = tmp_path / "nested" / "ocr.json"
    with CachedOCR(CountingBackend(), path) as cache:
        cache.read(rendered_plate())
    backend = CountingBackend()
    assert CachedOCR(backend, path).read(rendered_plate()) == PlateRead("BXYZ123", 0.75)
    assert backend.calls == 0


def test_cache_key_depends_on_pixels_and_shape(tmp_path):
    backend = CountingBackend()
    cache = CachedOCR(backend, tmp_path / "ocr.json")
    pixels = np.arange(1, 121, dtype=np.uint8)
    cache.read(pixels.reshape(4, 10, 3))
    cache.read(pixels.reshape(10, 4, 3))  # same bytes, different image
    cache.read(pixels.reshape(4, 10, 3) + 1)
    assert backend.calls == 3


# --- real backends (need downloaded models) -----------------------------------


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown OCR backend"):
        build_ocr("tesseract")


@pytest.mark.models
@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_reads_a_rendered_plate(backend):
    ocr = build_ocr(backend, gpu=False)
    assert ocr.name.startswith(backend)
    read = ocr.read(rendered_plate())
    assert read is not None
    assert read.text == "BXYZ123"
    assert 0.0 < read.confidence <= 1.0
    assert ocr.read(np.empty((0, 0, 3), np.uint8)) is None


@pytest.mark.models
def test_easyocr_handles_every_preprocessing_variant():
    for name in ["gray", "otsu_inv", "clahe", "upscale"]:
        read = build_ocr("easyocr", preprocess=name, gpu=False).read(rendered_plate())
        assert read is not None and read.text == "BXYZ123", name
