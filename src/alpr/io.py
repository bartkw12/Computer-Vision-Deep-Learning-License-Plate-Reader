"""Reading and writing videos and result tables."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from alpr.types import Box, PlateResult, TrackObservation

logger = logging.getLogger(__name__)

OBSERVATION_COLUMNS = [
    "frame",
    "track_id",
    "vehicle_x1",
    "vehicle_y1",
    "vehicle_x2",
    "vehicle_y2",
    "plate_x1",
    "plate_y1",
    "plate_x2",
    "plate_y2",
    "plate_score",
    "ocr_text",
    "ocr_confidence",
    "plate_text",
    "interpolated",
]

PLATE_RESULT_COLUMNS = ["track_id", "text", "confidence", "n_reads", "first_frame", "last_frame"]


class VideoReader:
    """Iterates over the frames of a video file.

    Use as a context manager so the file is released::

        with VideoReader("clip.mp4") as video:
            for index, frame in video:
                ...

    Attributes:
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        frame_count: Frame count reported by the container; it can be off by
            a few frames, so do not rely on it for loop bounds.

    Raises:
        FileNotFoundError: If the path does not exist.
        OSError: If the file exists but cannot be decoded.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"video not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise OSError(f"could not open video: {self.path}")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                return
            yield index, frame
            index += 1

    def close(self) -> None:
        """Release the underlying file."""
        self._capture.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class VideoWriter:
    """Writes BGR frames to an H.264 MP4 that browsers and GitHub can play.

    OpenCV's own writer falls back to the ``mp4v`` codec, which most browsers
    refuse. This pipes frames to the ffmpeg binary bundled with
    ``imageio-ffmpeg`` instead, so no system ffmpeg is needed.

    Args:
        path: Output file; parent directories are created.
        fps: Frames per second.
        size: ``(width, height)`` of the frames that will be written.
        crf: H.264 quality, lower is better; 23 is ffmpeg's default.
    """

    def __init__(
        self, path: str | Path, fps: float, size: tuple[int, int], *, crf: int = 23
    ) -> None:
        import imageio_ffmpeg

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._size = size
        self._writer = imageio_ffmpeg.write_frames(
            str(self.path),
            size,
            fps=fps,
            codec="libx264",
            pix_fmt_in="bgr24",
            pix_fmt_out="yuv420p",
            macro_block_size=2,  # yuv420p needs even dimensions
            output_params=["-crf", str(crf), "-movflags", "+faststart"],
            ffmpeg_log_level="error",
        )
        self._writer.send(None)

    def write(self, frame: np.ndarray) -> None:
        """Append one BGR frame.

        Raises:
            ValueError: If the frame size differs from the one given at creation.
        """
        height, width = frame.shape[:2]
        if (width, height) != self._size:
            raise ValueError(f"frame is {width}x{height}, writer expects {self._size}")
        self._writer.send(np.ascontiguousarray(frame, dtype=np.uint8))

    def close(self) -> None:
        """Finish the file."""
        self._writer.close()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _number(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_observations(path: str | Path, observations: Iterable[TrackObservation]) -> None:
    """Write per-frame track observations as a flat CSV.

    Every box coordinate gets its own numeric column, so the file loads
    directly into pandas, a spreadsheet or ``numpy.genfromtxt``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(OBSERVATION_COLUMNS)
        for row in observations:
            plate = row.plate_box.as_tuple() if row.plate_box else (None,) * 4
            writer.writerow(
                [
                    row.frame,
                    row.track_id,
                    *(_number(v, 2) for v in row.vehicle_box.as_tuple()),
                    *(_number(v, 2) for v in plate),
                    _number(row.plate_score, 4),
                    row.ocr_text or "",
                    _number(row.ocr_confidence, 4),
                    row.plate_text or "",
                    int(row.interpolated),
                ]
            )


def read_observations(path: str | Path) -> list[TrackObservation]:
    """Read a CSV written by :func:`write_observations`."""

    def number(text: str) -> float | None:
        return float(text) if text else None

    observations = []
    with Path(path).open(newline="") as file:
        for row in csv.DictReader(file):
            has_plate = bool(row["plate_x1"])
            observations.append(
                TrackObservation(
                    frame=int(row["frame"]),
                    track_id=int(row["track_id"]),
                    vehicle_box=Box(
                        *(float(row[f"vehicle_{c}"]) for c in ("x1", "y1", "x2", "y2"))
                    ),
                    plate_box=(
                        Box(*(float(row[f"plate_{c}"]) for c in ("x1", "y1", "x2", "y2")))
                        if has_plate
                        else None
                    ),
                    plate_score=number(row["plate_score"]),
                    ocr_text=row["ocr_text"] or None,
                    ocr_confidence=number(row["ocr_confidence"]),
                    plate_text=row["plate_text"] or None,
                    interpolated=row["interpolated"] == "1",
                )
            )
    return observations


def write_plate_results(path: str | Path, results: Iterable[PlateResult]) -> None:
    """Write the final plate of each track as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(PLATE_RESULT_COLUMNS)
        for r in results:
            writer.writerow(
                [
                    r.track_id,
                    r.text,
                    _number(r.confidence, 4),
                    r.n_reads,
                    r.first_frame,
                    r.last_frame,
                ]
            )


def read_plate_results(path: str | Path) -> list[PlateResult]:
    """Read a CSV written by :func:`write_plate_results`."""
    with Path(path).open(newline="") as file:
        return [
            PlateResult(
                track_id=int(row["track_id"]),
                text=row["text"],
                confidence=float(row["confidence"]),
                n_reads=int(row["n_reads"]),
                first_frame=int(row["first_frame"]),
                last_frame=int(row["last_frame"]),
            )
            for row in csv.DictReader(file)
        ]
