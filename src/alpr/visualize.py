"""Drawing pipeline results onto video frames.

All sizes scale with the frame, so the overlay looks the same on a 720p clip
as on a 4K one, and everything drawn above a vehicle is clamped to the frame
instead of disappearing when the vehicle is near the top edge.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from alpr.geometry import crop
from alpr.io import VideoReader, VideoWriter
from alpr.postprocess import fill_gaps
from alpr.types import Box, PlateResult, TrackObservation

logger = logging.getLogger(__name__)

VEHICLE_COLOR = (0, 171, 255)  # BGR
PLATE_COLOR = (0, 0, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_corner_box(
    image: np.ndarray, box: Box, color: tuple[int, int, int], thickness: int
) -> None:
    """Draw the four corners of ``box`` as brackets, in place."""
    x1, y1, x2, y2 = box.clipped(image.shape[1], image.shape[0]).to_int()
    length = max(4, round(0.2 * min(x2 - x1, y2 - y1)))
    for x, dx in ((x1, length), (x2, -length)):
        for y, dy in ((y1, length), (y2, -length)):
            cv2.line(image, (x, y), (x + dx, y), color, thickness)
            cv2.line(image, (x, y), (x, y + dy), color, thickness)


def _place(image: np.ndarray, width: int, height: int, center_x: float, bottom: float):
    """Top-left corner for a ``width`` x ``height`` patch, kept inside the image.

    Returns ``None`` if the patch is larger than the image.
    """
    image_height, image_width = image.shape[:2]
    if width > image_width or height > image_height:
        return None
    x = min(max(round(center_x - width / 2), 0), image_width - width)
    y = min(max(round(bottom - height), 0), image_height - height)
    return x, y


def paste_above(image: np.ndarray, patch: np.ndarray, center_x: float, bottom: float) -> int:
    """Paste ``patch`` centered on ``center_x`` with its lower edge at ``bottom``.

    The patch is shifted as needed to stay inside the image.

    Returns:
        The y coordinate of the pasted patch's top edge, or ``bottom`` if the
        patch did not fit at all.
    """
    height, width = patch.shape[:2]
    position = _place(image, width, height, center_x, bottom)
    if position is None:
        return round(bottom)
    x, y = position
    image[y : y + height, x : x + width] = patch
    return y


def draw_label(
    image: np.ndarray, text: str, center_x: float, bottom: float, scale: float, thickness: int
) -> int:
    """Draw black text on a white banner, kept inside the image.

    Returns:
        The y coordinate of the banner's top edge.
    """
    (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    pad = max(2, round(0.35 * text_height))
    width, height = text_width + 2 * pad, text_height + baseline + 2 * pad
    position = _place(image, width, height, center_x, bottom)
    if position is None:
        return round(bottom)
    x, y = position
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), cv2.FILLED)
    cv2.putText(
        image,
        text,
        (x + pad, y + pad + text_height),
        _FONT,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return y


class FrameAnnotator:
    """Draws tracks, plate boxes and plate strings onto frames.

    Args:
        frame_height: Height of the video in pixels; all sizes derive from it.
        plates: Final plate per track ID. Tracks without one get a box only.
        insets: Optional enlarged plate crop per track ID, shown above the
            vehicle together with the plate string.
    """

    def __init__(
        self,
        frame_height: int,
        plates: Mapping[int, PlateResult],
        insets: Mapping[int, np.ndarray] | None = None,
    ) -> None:
        unit = frame_height / 1080
        self._plates = plates
        self._thickness = max(1, round(4 * unit))
        self._font_scale = max(0.4, 1.1 * unit)
        self._font_thickness = max(1, round(2.5 * unit))
        self._margin = max(2, round(8 * unit))
        inset_height = max(16, round(0.07 * frame_height))
        self._insets = {
            track_id: cv2.resize(
                image,
                (max(1, round(image.shape[1] * inset_height / image.shape[0])), inset_height),
                interpolation=cv2.INTER_CUBIC,
            )
            for track_id, image in (insets or {}).items()
            if image.size
        }

    def draw(self, frame: np.ndarray, observations: Sequence[TrackObservation]) -> np.ndarray:
        """Annotate ``frame`` in place and return it."""
        for row in observations:
            draw_corner_box(frame, row.vehicle_box, VEHICLE_COLOR, self._thickness)
            if row.plate_box is not None:
                x1, y1, x2, y2 = row.plate_box.to_int()
                cv2.rectangle(frame, (x1, y1), (x2, y2), PLATE_COLOR, max(1, self._thickness // 2))
            plate = self._plates.get(row.track_id)
            if plate is None:
                continue
            center_x = row.vehicle_box.center[0]
            bottom = row.vehicle_box.y1 - self._margin
            inset = self._insets.get(row.track_id)
            if inset is not None:
                bottom = paste_above(frame, inset, center_x, bottom)
            draw_label(frame, plate.text, center_x, bottom, self._font_scale, self._font_thickness)
        return frame


def select_best_views(observations: Iterable[TrackObservation]) -> dict[int, TrackObservation]:
    """Pick, per track, the frame in which its plate appears largest.

    Tracks whose plate was never detected fall back to the frame with the
    largest vehicle box.
    """
    best: dict[int, TrackObservation] = {}

    def quality(row: TrackObservation) -> tuple[float, float]:
        return (row.plate_box.area if row.plate_box else 0.0, row.vehicle_box.area)

    for row in observations:
        if row.interpolated:
            continue
        current = best.get(row.track_id)
        if current is None or quality(row) > quality(current):
            best[row.track_id] = row
    return best


def extract_crops(
    video_path: str | Path, views: Mapping[int, TrackObservation]
) -> dict[int, tuple[np.ndarray, np.ndarray | None]]:
    """Cut each track's vehicle and plate out of the frame chosen for it.

    The video is decoded once, front to back, stopping after the last frame
    that is needed.

    Returns:
        ``{track_id: (vehicle_crop, plate_crop)}``; ``plate_crop`` is ``None``
        for tracks without a detected plate.
    """
    wanted: dict[int, list[TrackObservation]] = defaultdict(list)
    for row in views.values():
        wanted[row.frame].append(row)
    crops: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    if not wanted:
        return crops
    last = max(wanted)
    with VideoReader(video_path) as video:
        for index, frame in video:
            for row in wanted.get(index, ()):
                plate = crop(frame, row.plate_box).copy() if row.plate_box else None
                crops[row.track_id] = (crop(frame, row.vehicle_box).copy(), plate)
            if index >= last:
                break
    return crops


def render_video(
    video_path: str | Path,
    observations: Sequence[TrackObservation],
    results: Sequence[PlateResult],
    output_path: str | Path,
    *,
    max_gap: int = 10,
    show_insets: bool = True,
    progress: bool = True,
) -> Path:
    """Write an annotated copy of the video.

    Args:
        video_path: The video the observations were computed from.
        observations: The pipeline's per-frame log.
        results: Final plate per track.
        output_path: Where to write the H.264 MP4.
        max_gap: Detector dropouts up to this many frames are bridged by
            interpolation so boxes do not flicker.
        show_insets: Show each track's clearest plate crop above the vehicle.
        progress: Show a progress bar.

    Returns:
        The output path.
    """
    insets = None
    if show_insets:
        views = select_best_views(observations)
        insets = {
            track_id: plate
            for track_id, (_, plate) in extract_crops(video_path, views).items()
            if plate is not None
        }

    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for row in fill_gaps(observations, max_gap):
        by_frame[row.frame].append(row)

    output_path = Path(output_path)
    with VideoReader(video_path) as video:
        annotator = FrameAnnotator(video.height, {r.track_id: r for r in results}, insets)
        with VideoWriter(output_path, video.fps, (video.width, video.height)) as writer:
            frames = tqdm(
                video, total=video.frame_count or None, desc="render", disable=not progress
            )
            for index, frame in frames:
                writer.write(annotator.draw(frame, by_frame.get(index, ())))
    logger.info("wrote %s", output_path)
    return output_path
