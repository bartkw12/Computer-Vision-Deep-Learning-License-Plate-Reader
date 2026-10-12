"""Box geometry on NumPy arrays.

Arrays of boxes have shape ``(N, 4)`` with columns ``x1, y1, x2, y2``. Every
function accepts ``N == 0`` and returns a correctly shaped empty result, so
callers never need a special case for frames with nothing in them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alpr.types import Box


def boxes_to_array(boxes: Sequence[Box]) -> np.ndarray:
    """Stack boxes into a float array of shape ``(N, 4)``."""
    if not boxes:
        return np.empty((0, 4), dtype=float)
    return np.array([b.as_tuple() for b in boxes], dtype=float)


def _areas(boxes: np.ndarray) -> np.ndarray:
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    return widths * heights


def intersection_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection areas between ``a`` ``(N, 4)`` and ``b`` ``(M, 4)``.

    Returns:
        Array of shape ``(N, M)``.
    """
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    extent = np.clip(right_bottom - left_top, 0.0, None)
    return extent[..., 0] * extent[..., 1]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection-over-union between ``a`` ``(N, 4)`` and ``b`` ``(M, 4)``.

    Returns:
        Array of shape ``(N, M)`` with values in ``[0, 1]``. Pairs whose union
        is empty get 0.
    """
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    inter = intersection_matrix(a, b)
    union = _areas(a)[:, None] + _areas(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection over the area of ``a``.

    Entry ``(i, j)`` is the fraction of box ``a[i]`` that lies inside ``b[j]``.
    Unlike IoU this does not shrink when ``b`` is much larger than ``a``, which
    is what matters when asking whether a small plate sits on a large vehicle.

    Returns:
        Array of shape ``(N, M)`` with values in ``[0, 1]``. Rows for
        zero-area boxes are 0.
    """
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    inter = intersection_matrix(a, b)
    area_a = _areas(a)[:, None]
    return np.divide(inter, area_a, out=np.zeros_like(inter), where=area_a > 0)


def crop_with_offset(image: np.ndarray, box: Box) -> tuple[np.ndarray, tuple[int, int]]:
    """Cut ``box`` out of ``image`` and report where the cut starts.

    Returns:
        ``(crop, (x, y))`` where ``(x, y)`` is the crop's top-left corner in
        image coordinates. Adding it to coordinates found inside the crop maps
        them back to the full image.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box.clipped(width, height).to_int()
    return image[y1:y2, x1:x2], (x1, y1)


def crop(image: np.ndarray, box: Box) -> np.ndarray:
    """Cut ``box`` out of ``image``, clipping it to the image bounds.

    Returns:
        A view into ``image``. It has zero size if the box lies outside the
        image or collapses to nothing after rounding; check ``.size``.
    """
    return crop_with_offset(image, box)[0]
