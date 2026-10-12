"""Deciding which tracked vehicle a detected plate belongs to."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alpr.geometry import boxes_to_array, ioa_matrix
from alpr.types import Detection, Track


def assign_plates_to_tracks(
    plates: Sequence[Detection],
    tracks: Sequence[Track],
    *,
    min_overlap: float = 0.7,
    tie_margin: float = 0.05,
) -> dict[int, Detection]:
    """Match plate detections to vehicle tracks.

    A plate belongs to the vehicle that contains the largest share of it.
    Containment is measured as intersection over the *plate's* area, so a
    plate box that pokes a pixel or two outside the vehicle box still matches.

    When a plate lies almost equally inside several vehicle boxes, which
    happens when one vehicle partly hides another, the smallest box wins: it
    is the tightest explanation of where the plate is.

    Args:
        plates: Plate detections in the frame.
        tracks: Vehicle tracks in the frame.
        min_overlap: Smallest share of the plate that must lie inside the
            vehicle box.
        tie_margin: Overlaps within this much of the best one count as tied.

    Returns:
        Mapping from track ID to its plate. Each track gets at most one plate,
        the highest-scoring one if several match. Plates that match no track
        are left out.
    """
    if not plates or not tracks:
        return {}

    track_boxes = boxes_to_array([t.box for t in tracks])
    overlap = ioa_matrix(boxes_to_array([p.box for p in plates]), track_boxes)
    track_areas = np.array([t.box.area for t in tracks])

    assigned: dict[int, Detection] = {}
    for plate, row in zip(plates, overlap, strict=True):
        best = row.max()
        if best < min_overlap:
            continue
        tied = np.flatnonzero((row >= best - tie_margin) & (row >= min_overlap))
        track_id = tracks[tied[np.argmin(track_areas[tied])]].track_id
        current = assigned.get(track_id)
        if current is None or plate.score > current.score:
            assigned[track_id] = plate
    return assigned
