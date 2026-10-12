"""Filling short gaps in a track's boxes by linear interpolation.

The pipeline logs every track in every frame it is matched, so gaps only arise
where a detector missed for a few frames. Those are bridged here for smoother
rendering. Long gaps are left alone: drawing a box along a straight line for
several seconds would invent positions nobody observed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from alpr.types import Box, TrackObservation


def interpolate_gaps(
    frames: np.ndarray, boxes: np.ndarray, max_gap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate boxes over missing frames.

    Args:
        frames: Frame indices with a known box, shape ``(N,)``, in any order.
        boxes: The known boxes, shape ``(N, 4)``.
        max_gap: Longest run of missing frames that is filled.

    Returns:
        ``(frames, boxes, interpolated)``: all frames in ascending order, their
        boxes, and a boolean mask marking the rows that were filled in.

    Raises:
        ValueError: If a frame index appears more than once.
    """
    frames = np.asarray(frames, dtype=int).reshape(-1)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(np.unique(frames)) != len(frames):
        raise ValueError("frame indices must be unique within a track")
    order = np.argsort(frames)
    frames, boxes = frames[order], boxes[order]

    out_frames: list[np.ndarray] = []
    out_boxes: list[np.ndarray] = []
    out_filled: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        out_frames.append(frame[None])
        out_boxes.append(boxes[i][None])
        out_filled.append(np.zeros(1, dtype=bool))
        if i + 1 == len(frames):
            break
        missing = np.arange(frame + 1, frames[i + 1])
        if 0 < len(missing) <= max_gap:
            fraction = (missing - frame) / (frames[i + 1] - frame)
            out_frames.append(missing)
            out_boxes.append(boxes[i] + fraction[:, None] * (boxes[i + 1] - boxes[i]))
            out_filled.append(np.ones(len(missing), dtype=bool))

    if not out_frames:
        return frames, boxes, np.zeros(0, dtype=bool)
    return np.concatenate(out_frames), np.concatenate(out_boxes), np.concatenate(out_filled)


def fill_gaps(observations: Iterable[TrackObservation], max_gap: int) -> list[TrackObservation]:
    """Bridge short gaps in every track's vehicle and plate boxes.

    Vehicle boxes are interpolated between the frames a track was observed in;
    the new rows are flagged ``interpolated``. Plate boxes are interpolated
    separately between the frames a plate was actually detected, because a
    vehicle can be tracked while its plate goes undetected. An interpolated
    plate box carries no score and no text.

    Args:
        observations: Logged observations, in any order.
        max_gap: Longest run of missing frames that is filled.

    Returns:
        Original and interpolated observations, sorted by frame then track ID.
    """
    by_track: dict[int, list[TrackObservation]] = defaultdict(list)
    for observation in observations:
        by_track[observation.track_id].append(observation)

    result: list[TrackObservation] = []
    for track_id, rows in by_track.items():
        by_frame = {row.frame: row for row in rows}

        frames, boxes, filled = interpolate_gaps(
            np.array([r.frame for r in rows]),
            np.array([r.vehicle_box.as_tuple() for r in rows]),
            max_gap,
        )
        for frame, box, is_filled in zip(frames, boxes, filled, strict=True):
            if is_filled:
                by_frame[int(frame)] = TrackObservation(
                    frame=int(frame), track_id=track_id, vehicle_box=Box(*box), interpolated=True
                )

        with_plate = [r for r in rows if r.plate_box is not None]
        if len(with_plate) > 1:
            frames, boxes, filled = interpolate_gaps(
                np.array([r.frame for r in with_plate]),
                np.array([r.plate_box.as_tuple() for r in with_plate]),
                max_gap,
            )
            for frame, box, is_filled in zip(frames, boxes, filled, strict=True):
                row = by_frame.get(int(frame))
                if is_filled and row is not None:
                    by_frame[int(frame)] = replace(row, plate_box=Box(*box))

        result.extend(by_frame.values())
    return sorted(result, key=lambda row: (row.frame, row.track_id))
