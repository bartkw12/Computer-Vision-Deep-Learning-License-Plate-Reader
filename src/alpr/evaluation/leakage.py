"""Finding near-duplicate images shared between a dataset's splits.

Datasets assembled from video often contain consecutive frames of the same
scene. When one frame lands in ``train`` and the next in ``test``, the test
score measures memory rather than generalization. This module looks for such
pairs with a perceptual hash: images that look alike hash to bit strings that
differ in few places, whatever their file names, sizes or compression.

The hash is the 64-bit difference hash (dHash): shrink the image to 9 x 8
gray pixels and record, for each pixel, whether it is brighter than its
right-hand neighbour. It survives rescaling, recompression and mild color or
brightness changes; it does not survive crops, flips or large shifts, so a
clean result bounds rather than rules out leakage.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

#: Split folder names, in the order Roboflow exports them.
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

#: Hamming distances (out of 64 bits) reported. 0 is a pixel-level copy; up
#: to about 4 is the same picture re-encoded; around 10, the same scene.
DEFAULT_THRESHOLDS = (0, 4, 10)

HASH_BYTES = 8
_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], np.uint8)


def dhash(image: np.ndarray) -> np.ndarray:
    """Return the 64-bit difference hash of a BGR or grayscale image.

    Returns:
        ``HASH_BYTES`` packed bits as a ``uint8`` array.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits(small[:, 1:] > small[:, :-1])


def split_images(dataset_dir: str | Path, split: str) -> list[Path]:
    """List a split's images, in YOLO (``split/images``) or flat layout."""
    folder = Path(dataset_dir) / split
    if (folder / "images").is_dir():
        folder = folder / "images"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def hash_images(paths: Sequence[Path], *, progress: bool = True) -> np.ndarray:
    """Hash every image; unreadable files get an all-zero hash and a warning.

    Returns:
        An ``(n, HASH_BYTES)`` ``uint8`` array, one row per path.
    """
    hashes = np.zeros((len(paths), HASH_BYTES), np.uint8)
    for row, path in enumerate(tqdm(paths, desc="hashing", disable=not progress)):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("could not read %s", path)
            continue
        hashes[row] = dhash(image)
    return hashes


def nearest(
    queries: np.ndarray, references: np.ndarray, *, chunk: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Find each query hash's closest reference hash.

    Args:
        queries: ``(n, HASH_BYTES)`` hashes.
        references: ``(m, HASH_BYTES)`` hashes, ``m >= 1``.
        chunk: Queries compared at once; bounds memory to ``chunk * m * 8``
            bytes.

    Returns:
        ``(distances, indices)``: for each query the Hamming distance to its
        nearest reference and that reference's row.
    """
    distances = np.empty(len(queries), np.int64)
    indices = np.empty(len(queries), np.int64)
    for start in range(0, len(queries), chunk):
        block = queries[start : start + chunk]
        bits = _POPCOUNT[block[:, None, :] ^ references[None, :, :]]
        differing = bits.sum(axis=2, dtype=np.int64)
        indices[start : start + chunk] = differing.argmin(axis=1)
        distances[start : start + chunk] = differing.min(axis=1)
    return distances, indices


def compare_splits(
    query_paths: Sequence[Path],
    query_hashes: np.ndarray,
    reference_paths: Sequence[Path],
    reference_hashes: np.ndarray,
    *,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
    examples: int = 10,
    root: Path | None = None,
) -> dict[str, Any]:
    """Count query images that have a near-duplicate among the references.

    Returns:
        The number of queries, for each threshold how many of them (and what
        share) lie within that distance of some reference, the distribution of
        nearest distances, and the ``examples`` closest pairs.
    """
    if len(query_paths) == 0 or len(reference_paths) == 0:
        return {"queries": len(query_paths), "references": len(reference_paths), "within": {}}
    distances, indices = nearest(query_hashes, reference_hashes)

    def name(path: Path) -> str:
        return str(path.relative_to(root)) if root else str(path)

    order = np.argsort(distances, kind="stable")[:examples]
    return {
        "queries": len(query_paths),
        "references": len(reference_paths),
        "within": {
            str(t): {
                "count": int((distances <= t).sum()),
                "share": float((distances <= t).mean()),
            }
            for t in thresholds
        },
        "nearest_distance_histogram": {
            str(d): int(n) for d, n in zip(*np.unique(distances, return_counts=True), strict=True)
        },
        "closest_pairs": [
            {
                "query": name(query_paths[i]),
                "reference": name(reference_paths[indices[i]]),
                "distance": int(distances[i]),
            }
            for i in order
        ],
    }


def leakage_report(
    dataset_dir: str | Path,
    *,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
    examples: int = 10,
    progress: bool = True,
) -> dict[str, Any]:
    """Check every later split against every earlier one.

    ``valid`` is compared with ``train``, and ``test`` with both. Each image of
    the later split is matched to its nearest image in the earlier one.

    Returns:
        Split sizes, and a :func:`compare_splits` result per split pair, keyed
        ``"<later>_vs_<earlier>"``.
    """
    root = Path(dataset_dir)
    paths = {split: split_images(root, split) for split in SPLITS}
    hashes = {split: hash_images(paths[split], progress=progress) for split in SPLITS}
    report: dict[str, Any] = {
        "hash": "dHash, 64 bits",
        "thresholds": list(thresholds),
        "split_sizes": {split: len(paths[split]) for split in SPLITS},
        "pairs": {},
    }
    for later_index, later in enumerate(SPLITS):
        for earlier in SPLITS[:later_index]:
            report["pairs"][f"{later}_vs_{earlier}"] = compare_splits(
                paths[later],
                hashes[later],
                paths[earlier],
                hashes[earlier],
                thresholds=thresholds,
                examples=examples,
                root=root,
            )
    return report
