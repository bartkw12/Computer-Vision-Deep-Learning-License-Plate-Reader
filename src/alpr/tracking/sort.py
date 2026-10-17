"""SORT: Simple Online and Realtime Tracking.

An implementation of Bewley et al., "Simple Online and Realtime Tracking"
(ICIP 2016). Each track is a constant-velocity Kalman filter; detections are
assigned to predicted track positions by solving a linear assignment problem
on intersection-over-union.

Two deliberate differences from the authors' reference code:

* **Confirmation is sticky.** The reference shows a track only while its
  current streak of consecutive matches is at least ``min_hits``, so a track
  that is re-matched after a missed frame disappears again for ``min_hits``
  frames. Here a track that has reached the streak once stays confirmed, which
  is what makes ``max_age > 1`` useful for bridging detector dropouts.
* **IDs are per tracker.** The reference counts IDs on the class, so they keep
  growing across videos processed in one Python session.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from alpr.geometry import iou_matrix
from alpr.tracking.kalman import KalmanBoxTracker

_EMPTY_TRACKS = np.empty((0, 5))


def associate(
    detections: np.ndarray, predictions: np.ndarray, iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int]]:
    """Match detections to predicted track boxes.

    Args:
        detections: Detected boxes, shape ``(N, 4)``.
        predictions: Predicted track boxes, shape ``(M, 4)``.
        iou_threshold: Pairs overlapping less than this are not matched.

    Returns:
        ``(matches, unmatched_detections)`` where ``matches`` holds
        ``(detection_index, track_index)`` pairs.
    """
    if len(detections) == 0 or len(predictions) == 0:
        return [], list(range(len(detections)))
    iou = iou_matrix(detections, predictions)
    rows, cols = linear_sum_assignment(iou, maximize=True)
    matches = [
        (int(r), int(c)) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= iou_threshold
    ]
    matched = {d for d, _ in matches}
    return matches, [d for d in range(len(detections)) if d not in matched]


class Sort:
    """Multi-object tracker.

    Args:
        max_age: Frames a track may go unmatched before it is dropped.
        min_hits: Consecutive matches a new track needs before it is reported.
            Tracks born in the first ``min_hits`` frames are reported at once.
        iou_threshold: Minimum overlap for a detection to continue a track.
    """

    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._trackers: list[KalmanBoxTracker] = []
        self._frame_count = 0
        self._next_id = 1

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Advance the tracker by one frame.

        Call this once per frame, including frames with no detections.

        Args:
            detections: Array of shape ``(N, 5)`` with rows
                ``[x1, y1, x2, y2, score]``. Anything empty is accepted for
                a frame without detections.

        Returns:
            Array of shape ``(M, 5)`` with rows ``[x1, y1, x2, y2, track_id]``,
            one per confirmed track matched in this frame.

        Raises:
            ValueError: If a non-empty input does not have five columns.
        """
        detections = np.asarray(detections, dtype=float)
        if detections.size == 0:
            detections = _EMPTY_TRACKS
        elif detections.ndim != 2 or detections.shape[1] != 5:
            raise ValueError(f"detections must have shape (N, 5), got {detections.shape}")
        self._frame_count += 1

        predictions = np.array([t.predict() for t in self._trackers]).reshape(-1, 4)
        finite = np.isfinite(predictions).all(axis=1)
        self._trackers = [t for t, ok in zip(self._trackers, finite, strict=True) if ok]
        predictions = predictions[finite]

        matches, unmatched = associate(detections[:, :4], predictions, self.iou_threshold)
        for detection_index, track_index in matches:
            tracker = self._trackers[track_index]
            tracker.update(detections[detection_index, :4])
            if tracker.hit_streak >= self.min_hits:
                tracker.confirmed = True
        for detection_index in unmatched:
            self._trackers.append(KalmanBoxTracker(detections[detection_index, :4], self._next_id))
            self._next_id += 1

        warming_up = self._frame_count <= self.min_hits
        reported = [
            [*t.box, t.track_id]
            for t in self._trackers
            if t.time_since_update == 0 and (t.confirmed or warming_up)
        ]
        self._trackers = [t for t in self._trackers if t.time_since_update <= self.max_age]
        return np.array(reported, dtype=float).reshape(-1, 5)
