"""Small immutable value types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in pixel coordinates.

    ``(x1, y1)`` is the top-left corner and ``(x2, y2)`` the bottom-right one.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Horizontal extent, never negative."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Vertical extent, never negative."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point ``(cx, cy)``."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(x1, y1, x2, y2)``."""
        return self.x1, self.y1, self.x2, self.y2

    def contains_point(self, x: float, y: float) -> bool:
        """Whether the point lies inside the box (edges included)."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def clipped(self, width: int, height: int) -> Box:
        """Return the part of the box that lies inside a ``width`` x ``height`` image."""
        return Box(
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
            min(max(self.x2, 0.0), float(width)),
            min(max(self.y2, 0.0), float(height)),
        )

    def shifted(self, dx: float, dy: float) -> Box:
        """Return the box translated by ``(dx, dy)``."""
        return Box(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def to_int(self) -> tuple[int, int, int, int]:
        """Return the corners rounded to whole pixels, for slicing and drawing."""
        return round(self.x1), round(self.y1), round(self.x2), round(self.y2)


@dataclass(frozen=True, slots=True)
class Detection:
    """One detector output: a box, its confidence, and its class index."""

    box: Box
    score: float
    class_id: int = 0


@dataclass(frozen=True, slots=True)
class Track:
    """A tracked object in one frame: a persistent ID and its current box."""

    track_id: int
    box: Box


@dataclass(frozen=True, slots=True)
class PlateRead:
    """Raw OCR output for one plate crop.

    ``text`` is normalized (upper-case letters and digits only) but not yet
    checked against any plate format.
    """

    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """Everything the pipeline knows about one track in one frame.

    One of these is logged for every active track in every frame, whether or
    not a plate was found, so nothing has to be reconstructed afterwards.

    Attributes:
        frame: Zero-based frame index.
        track_id: Tracker ID of the vehicle.
        vehicle_box: Vehicle box in this frame.
        plate_box: Plate box matched to the vehicle, if one was detected.
        plate_score: Detector confidence of ``plate_box``.
        ocr_text: Normalized OCR text before any format check.
        ocr_confidence: OCR confidence of ``ocr_text``.
        plate_text: ``ocr_text`` after fitting it to a plate format, or ``None``
            if OCR was skipped or the text did not fit.
        interpolated: True for rows filled in by gap interpolation.
    """

    frame: int
    track_id: int
    vehicle_box: Box
    plate_box: Box | None = None
    plate_score: float | None = None
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    plate_text: str | None = None
    interpolated: bool = False


@dataclass(frozen=True, slots=True)
class PlateResult:
    """Final answer for one track: the plate string agreed on across its frames.

    Attributes:
        track_id: Tracker ID of the vehicle.
        text: Consensus plate string.
        confidence: Share of the total vote weight won by ``text``, in ``[0, 1]``.
        n_reads: Number of per-frame reads that entered the vote.
        first_frame: First frame in which the track was seen.
        last_frame: Last frame in which the track was seen.
    """

    track_id: int
    text: str
    confidence: float
    n_reads: int
    first_frame: int
    last_frame: int
