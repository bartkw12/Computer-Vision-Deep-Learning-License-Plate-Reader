"""Evaluating plate reading on a video against hand-typed ground truth.

Workflow:

1. :func:`write_label_sheet` turns a pipeline run into a CSV with one row per
   tracked vehicle and contact-sheet images showing each vehicle and its
   clearest plate crop. The system's own reads are deliberately left out so
   they cannot bias the person typing.
2. A person fills in the ``plate`` column: the serial as they read it, or
   ``?`` if it cannot be read even by eye.
3. :func:`evaluate_video` scores any run of the same video against the sheet.

Labels are anchored to a frame number and a pixel, not to a track ID. Track
IDs change whenever the detector or tracker settings change; a point on a
vehicle in a given frame does not, so one labeled sheet can score every
configuration.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from alpr.evaluation.metrics import score_readings, substitution_counts
from alpr.plate_format import PlateFormat, fit_plate, normalize
from alpr.types import PlateResult, TrackObservation
from alpr.visualize import extract_crops, select_best_views

logger = logging.getLogger(__name__)

LABEL_COLUMNS = ["label_id", "frame", "x", "y", "plate"]
UNREADABLE = "?"
_ROWS_PER_SHEET = 12


@dataclass(frozen=True, slots=True)
class Label:
    """One ground-truth entry.

    Attributes:
        label_id: Row number on the label sheet.
        frame: Frame the anchor point refers to.
        x: Anchor point, a pixel on the vehicle (its plate, when one was seen).
        y: See ``x``.
        plate: The normalized true serial, ``"?"`` if unreadable by eye, or
            ``""`` if the row has not been filled in.
    """

    label_id: int
    frame: int
    x: float
    y: float
    plate: str

    @property
    def readable(self) -> bool:
        """Whether the row carries a ground-truth serial."""
        return bool(self.plate) and self.plate != UNREADABLE


def _fit_into(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize ``image`` to fit a ``width`` x ``height`` cell, centered on black."""
    cell = np.zeros((height, width, 3), np.uint8)
    if image is None or image.size == 0:
        return cell
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(image, size, interpolation=interpolation)
    x, y = (width - size[0]) // 2, (height - size[1]) // 2
    cell[y : y + size[1], x : x + size[0]] = resized
    return cell


def _contact_sheet(rows: Sequence[tuple[int, np.ndarray, np.ndarray | None]]) -> np.ndarray:
    """Stack ``(label_id, vehicle_crop, plate_crop)`` rows into one image."""
    id_width, vehicle_size, plate_size = 90, (300, 180), (420, 180)
    strips = []
    for label_id, vehicle, plate in rows:
        id_cell = np.full((vehicle_size[1], id_width, 3), 32, np.uint8)
        cv2.putText(
            id_cell, str(label_id), (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2
        )
        strip = np.hstack(
            [id_cell, _fit_into(vehicle, *vehicle_size), _fit_into(plate, *plate_size)]
        )
        strip[-2:] = 90  # separator line
        strips.append(strip)
    return np.vstack(strips)


def write_label_sheet(
    observations: Sequence[TrackObservation],
    video_path: str | Path,
    csv_path: str | Path,
    *,
    min_frames: int = 5,
    overwrite: bool = False,
) -> int:
    """Create an empty label sheet for a run.

    Writes ``csv_path`` and, next to it, ``<name>_sheet_NN.jpg`` contact sheets
    whose row numbers match the CSV's ``label_id`` column.

    Args:
        observations: The run's per-frame log.
        video_path: The video the run was computed from.
        csv_path: Where to write the CSV.
        min_frames: Tracks seen in fewer frames are left out as flicker.
        overwrite: Replace an existing CSV. Off by default so that typed
            labels are never lost by accident.

    Returns:
        The number of rows written.

    Raises:
        FileExistsError: If the CSV exists and ``overwrite`` is false.
    """
    csv_path = Path(csv_path)
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"{csv_path} already exists; pass overwrite=True to replace it")

    frames_seen: dict[int, int] = defaultdict(int)
    for row in observations:
        frames_seen[row.track_id] += not row.interpolated
    views = {
        track_id: view
        for track_id, view in select_best_views(observations).items()
        if frames_seen[track_id] >= min_frames
    }
    crops = extract_crops(video_path, views)

    ordered = sorted(views.values(), key=lambda view: (view.frame, view.track_id))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(LABEL_COLUMNS)
        for label_id, view in enumerate(ordered, start=1):
            x, y = (view.plate_box or view.vehicle_box).center
            writer.writerow([label_id, view.frame, f"{x:.1f}", f"{y:.1f}", ""])
            sheet_rows.append((label_id, *crops[view.track_id]))

    for page, start in enumerate(range(0, len(sheet_rows), _ROWS_PER_SHEET), start=1):
        sheet = _contact_sheet(sheet_rows[start : start + _ROWS_PER_SHEET])
        cv2.imwrite(str(csv_path.with_name(f"{csv_path.stem}_sheet_{page:02d}.jpg")), sheet)
    logger.info("wrote %d label rows to %s", len(ordered), csv_path)
    return len(ordered)


def read_labels(csv_path: str | Path) -> list[Label]:
    """Read a label sheet, normalizing the typed serials."""
    labels = []
    with Path(csv_path).open(newline="") as file:
        for row in csv.DictReader(file):
            typed = (row.get("plate") or "").strip()
            labels.append(
                Label(
                    label_id=int(row["label_id"]),
                    frame=int(row["frame"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    plate=UNREADABLE if typed == UNREADABLE else normalize(typed),
                )
            )
    return labels


def match_labels_to_tracks(
    labels: Sequence[Label], observations: Sequence[TrackObservation], *, window: int = 5
) -> dict[int, int | None]:
    """Find the track each label's anchor point falls on.

    The label's own frame is tried first, then frames up to ``window`` away,
    nearest first, for runs in which the vehicle was not tracked in exactly
    that frame. When several vehicle boxes contain the point, the smallest
    one is taken.

    Returns:
        ``{label_id: track_id}``, with ``None`` for labels on no track.
    """
    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for row in observations:
        if not row.interpolated:
            by_frame[row.frame].append(row)

    offsets = [0] + [sign * d for d in range(1, window + 1) for sign in (1, -1)]
    matches: dict[int, int | None] = {}
    for label in labels:
        matches[label.label_id] = None
        for offset in offsets:
            containing = [
                row
                for row in by_frame.get(label.frame + offset, ())
                if row.vehicle_box.contains_point(label.x, label.y)
            ]
            if containing:
                smallest = min(containing, key=lambda row: row.vehicle_box.area)
                matches[label.label_id] = smallest.track_id
                break
    return matches


def evaluate_video(
    observations: Sequence[TrackObservation],
    results: Sequence[PlateResult],
    labels: Sequence[Label],
    *,
    formats: Sequence[PlateFormat] = (),
) -> dict[str, Any]:
    """Score a run against a filled-in label sheet.

    Args:
        observations: The run's per-frame log.
        results: The run's final plate per track.
        labels: Ground truth from :func:`read_labels`.
        formats: The plate formats the run assumed. Used only to report how
            many true serials fall outside them, which tests the assumption.

    Returns:
        A dictionary with two parts. ``"summary"`` holds aggregate numbers and
        contains no plate strings, so it can be published. ``"records"`` lists
        every label with its truth and prediction and should stay private when
        the video shows real vehicles.

        Summary fields: ``reading`` scores every readable label, counting an
        untracked or unread vehicle as wrong. ``precision`` is the share of
        emitted plates that were right. ``tracks_per_vehicle`` is the mean
        number of distinct track IDs covering the labels of one serial; 1.0
        means no vehicle was split across tracks.
    """
    readable = [label for label in labels if label.readable]
    track_of = match_labels_to_tracks(readable, observations)
    plate_of_track = {r.track_id: r.text for r in results}

    records = []
    for label in readable:
        track_id = track_of[label.label_id]
        records.append(
            {
                "label_id": label.label_id,
                "truth": label.plate,
                "track_id": track_id,
                "prediction": plate_of_track.get(track_id) if track_id is not None else None,
            }
        )

    truths = [r["truth"] for r in records]
    predictions = [r["prediction"] for r in records]
    reading = score_readings(predictions, truths)
    emitted = sum(p is not None for p in predictions)

    tracks_by_plate: dict[str, set[int]] = defaultdict(set)
    correct_plates = set()
    for record in records:
        if record["track_id"] is not None:
            tracks_by_plate[record["truth"]].add(record["track_id"])
        if record["prediction"] == record["truth"]:
            correct_plates.add(record["truth"])
    unique_plates = set(truths)
    fragmentation = [len(tracks) for tracks in tracks_by_plate.values()]

    labeled_tracks = {r["track_id"] for r in records if r["track_id"] is not None}
    confusions = substitution_counts(predictions, truths)
    outside_format = sum(
        fit_plate(truth, formats, max_substitutions=0) is None for truth in unique_plates
    )
    summary = {
        "labels": len(labels),
        "readable": len(readable),
        "unreadable": sum(label.plate == UNREADABLE for label in labels),
        "unlabeled": sum(not label.plate for label in labels),
        "reading": reading.as_dict(),
        "tracked": sum(r["track_id"] is not None for r in records),
        "emitted": emitted,
        "precision": reading.exact / emitted if emitted else 0.0,
        "unique_plates": len(unique_plates),
        "unique_plates_read_correctly": len(correct_plates),
        "tracks_per_vehicle": float(np.mean(fragmentation)) if fragmentation else 0.0,
        "tracks_in_run": len({row.track_id for row in observations}),
        "plates_emitted_on_unlabeled_tracks": sum(
            r.track_id not in labeled_tracks for r in results
        ),
        "true_plates_outside_assumed_format": outside_format if formats else None,
        "top_confusions": [
            {"truth": truth, "predicted": predicted, "count": count}
            for (truth, predicted), count in confusions.most_common(15)
        ],
    }
    return {"summary": summary, "records": records}
