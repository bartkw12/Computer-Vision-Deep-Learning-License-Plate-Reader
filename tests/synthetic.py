"""A synthetic traffic scene and pixel-driven stand-ins for the models.

The scene draws flat-colored vehicles with flat-colored plates. The stand-in
detectors find those rectangles by color and the stand-in OCR maps a plate's
color to its text. Everything between the models is therefore exercised for
real: cropping, tracking, association, gating, format fitting and voting.

Colors (BGR). The blue channel carries the vehicle's identity::

    background  (0, 0, 0)
    vehicle     (40 * k, 200, 0)
    plate       (40 * k, 0, 250)

The thresholds below leave enough margin to survive lossy video encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from alpr.types import Box, Detection, PlateRead

WIDTH, HEIGHT = 640, 360
ID_STEP = 40


@dataclass(frozen=True)
class Vehicle:
    """A vehicle that drives across the scene in a straight line."""

    key: int  # 1..6, encoded in the blue channel
    plate: str
    x0: float
    y: float
    vx: float
    first_frame: int
    last_frame: int
    width: int = 120
    height: int = 70

    def box(self, frame: int) -> Box | None:
        if not self.first_frame <= frame <= self.last_frame:
            return None
        x = self.x0 + self.vx * (frame - self.first_frame)
        return Box(x, self.y, x + self.width, self.y + self.height)

    def plate_box(self, frame: int) -> Box | None:
        body = self.box(frame)
        if body is None:
            return None
        cx = (body.x1 + body.x2) / 2
        return Box(cx - 18, body.y2 - 20, cx + 18, body.y2 - 6)


#: Three vehicles in separate lanes. Nothing is visible in frames 0-2 and
#: 41-47, which is the "gap in traffic" the original pipeline crashed on.
VEHICLES = (
    Vehicle(key=1, plate="BXYZ123", x0=20, y=30, vx=6, first_frame=3, last_frame=40),
    Vehicle(key=2, plate="CKLM456", x0=480, y=140, vx=-5, first_frame=8, last_frame=40),
    Vehicle(key=3, plate="DRST789", x0=60, y=250, vx=4, first_frame=48, last_frame=79),
)
N_FRAMES = 85


def _fill(image: np.ndarray, box: Box, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box.clipped(image.shape[1], image.shape[0]).to_int()
    image[y1:y2, x1:x2] = color


def render(frame: int, vehicles=VEHICLES, stray_plate: Box | None = None) -> np.ndarray:
    """Draw one frame of the scene."""
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for vehicle in vehicles:
        body = vehicle.box(frame)
        if body is not None:
            _fill(image, body, (ID_STEP * vehicle.key, 200, 0))
            _fill(image, vehicle.plate_box(frame), (ID_STEP * vehicle.key, 0, 250))
    if stray_plate is not None:
        _fill(image, stray_plate, (ID_STEP * 6, 0, 250))
    return image


def frames(n=N_FRAMES, **kwargs):
    """Yield ``(index, image)`` for the whole scene."""
    for index in range(n):
        yield index, render(index, **kwargs)


def _blobs(mask: np.ndarray, min_pixels: int) -> list[Box]:
    """Bounding boxes of the connected regions of ``mask``, left to right.

    Regions are found by connectivity rather than by exact color, so a few
    edge pixels smeared by video compression cannot merge two objects.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = [
        Box(float(x), float(y), float(x + w), float(y + h))
        for x, y, w, h, area in stats[1:count]
        if area >= min_pixels
    ]
    return sorted(boxes, key=lambda box: (box.x1, box.y1))


def _plate_mask(image: np.ndarray) -> np.ndarray:
    return (image[..., 2] > 180) & (image[..., 1] < 90)


class ColorDetector:
    """Finds the scene's vehicles or plates by their color."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        if image.size == 0:
            return []
        if self.kind == "plate":
            boxes = _blobs(_plate_mask(image), min_pixels=100)
        else:
            # A vehicle is its body plus the plate drawn on it: one region.
            body = (image[..., 1] > 130) & (image[..., 2] < 90)
            boxes = _blobs(body | _plate_mask(image), min_pixels=2000)
        return [Detection(box, 0.9, 0) for box in boxes]

    def detect_batch(self, images) -> list[list[Detection]]:
        return [self.detect(image) for image in images]


class ColorOCR:
    """Reads a plate by looking up its color.

    Args:
        plates: Mapping from vehicle key to plate text.
        corrupt_every: Every this-many-th read of a plate comes back with its
            last character wrong and a very high confidence, imitating a
            confident OCR mistake. ``0`` disables it.
        spacing: Insert a space in the middle, as real OCR output often has.
    """

    name = "color-ocr"
    preprocess_name = "none"

    def __init__(self, plates: dict[int, str], *, corrupt_every: int = 0, spacing: bool = True):
        self.plates = plates
        self.corrupt_every = corrupt_every
        self.spacing = spacing
        self.calls = 0
        self._reads_per_key: dict[int, int] = {}

    def read(self, crop: np.ndarray) -> PlateRead | None:
        self.calls += 1
        mask = _plate_mask(crop) if crop.size else np.zeros((0, 0), bool)
        if not mask.any():
            return None
        key = int(np.rint(np.median(crop[..., 0][mask]) / ID_STEP))
        text = self.plates.get(key)
        if text is None:
            return None
        count = self._reads_per_key[key] = self._reads_per_key.get(key, 0) + 1
        if self.corrupt_every and count % self.corrupt_every == 0:
            return PlateRead(text[:-1] + ("8" if text[-1] != "8" else "3"), 0.99)
        if self.spacing:
            text = f"{text[:4]} {text[4:]}"
        return PlateRead(text.replace(" ", ""), 0.8)


def stand_ins(**ocr_kwargs):
    """Build ``(vehicle_detector, plate_detector, ocr)`` for the default scene."""
    plates = {v.key: v.plate for v in VEHICLES}
    return ColorDetector("vehicle"), ColorDetector("plate"), ColorOCR(plates, **ocr_kwargs)
