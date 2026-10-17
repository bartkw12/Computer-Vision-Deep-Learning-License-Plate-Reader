"""Constant-velocity Kalman filter for a single bounding box.

The state follows the SORT paper (Bewley et al., 2016)::

    x = [cx, cy, s, r, vx, vy, vs]

``cx, cy`` is the box center, ``s`` its area and ``r`` its aspect ratio
(width / height). The center and area have velocities; the aspect ratio is
modeled as constant. Only ``[cx, cy, s, r]`` is observed.
"""

from __future__ import annotations

import numpy as np

_STATE_DIM = 7
_MEASUREMENT_DIM = 4

# State transition: position and area advance by their velocity each frame.
_F = np.eye(_STATE_DIM)
_F[0, 4] = _F[1, 5] = _F[2, 6] = 1.0

# Measurement model: the first four state variables are observed directly.
_H = np.eye(_MEASUREMENT_DIM, _STATE_DIM)

# Noise levels are those of the SORT reference implementation: the area and
# aspect ratio are measured less reliably than the center, and the velocities,
# which are never observed, start out almost unknown.
_R = np.diag([1.0, 1.0, 10.0, 10.0])
_Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])
_P0 = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

_I = np.eye(_STATE_DIM)


def box_to_measurement(box: np.ndarray) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` to ``[cx, cy, area, aspect_ratio]``."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return np.array(
        [
            box[0] + width / 2.0,
            box[1] + height / 2.0,
            width * height,
            width / max(height, 1e-6),
        ]
    )


def measurement_to_box(measurement: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, area, aspect_ratio]`` back to ``[x1, y1, x2, y2]``."""
    cx, cy, area, aspect = measurement[:4]
    width = np.sqrt(max(area * aspect, 0.0))
    height = area / width if width > 0 else 0.0
    return np.array([cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0])


class KalmanBoxTracker:
    """Tracks one object: its filter state plus the bookkeeping SORT needs.

    Attributes:
        track_id: Identifier assigned by the owning tracker.
        hit_streak: Consecutive frames in which the track was matched.
        time_since_update: Frames since the track was last matched.
        confirmed: Whether the track has ever reached the required streak.
    """

    def __init__(self, box: np.ndarray, track_id: int) -> None:
        self.track_id = track_id
        self.hit_streak = 0
        self.time_since_update = 0
        self.confirmed = False
        self._x = np.zeros(_STATE_DIM)
        self._x[:_MEASUREMENT_DIM] = box_to_measurement(np.asarray(box, dtype=float))
        self._P = _P0.copy()

    @property
    def box(self) -> np.ndarray:
        """Current box estimate as ``[x1, y1, x2, y2]``."""
        return measurement_to_box(self._x)

    def predict(self) -> np.ndarray:
        """Advance the state by one frame and return the predicted box."""
        if self._x[2] + self._x[6] <= 0:
            # Never let the predicted area go non-positive.
            self._x[6] = 0.0
        self._x = _F @ self._x
        self._P = _F @ self._P @ _F.T + _Q
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.box

    def update(self, box: np.ndarray) -> None:
        """Correct the state with a matched detection ``[x1, y1, x2, y2]``."""
        residual = box_to_measurement(np.asarray(box, dtype=float)) - _H @ self._x
        innovation_cov = _H @ self._P @ _H.T + _R
        gain = np.linalg.solve(innovation_cov, _H @ self._P).T
        self._x = self._x + gain @ residual
        # Joseph form keeps the covariance symmetric positive semi-definite.
        factor = _I - gain @ _H
        self._P = factor @ self._P @ factor.T + gain @ _R @ gain.T
        self.time_since_update = 0
        self.hit_streak += 1
