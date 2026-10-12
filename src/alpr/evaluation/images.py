"""End-to-end evaluation on still images with labeled plates.

Uses the OpenALPR benchmark (https://github.com/openalpr/benchmarks), whose
``endtoend`` folder holds one image and one label file per plate. A label
file is a single tab-separated line::

    <image file>  <x>  <y>  <width>  <height>  <plate text>

Two questions are kept apart:

* **Recognition alone**: OCR on the labeled plate box. This isolates the
  reader from the detector and needs no trained plate detector.
* **End to end**: detect the plate, then read the detected crop. A plate that
  is not detected counts as a wrong read.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from alpr.detection import Detector
from alpr.evaluation.metrics import score_readings, substitution_counts
from alpr.geometry import boxes_to_array, crop, iou_matrix
from alpr.ocr import PlateOCR
from alpr.plate_format import normalize
from alpr.types import Box

logger = logging.getLogger(__name__)

REGIONS = ("us", "eu", "br")

#: Plate-height bins (pixels) for the accuracy-versus-size breakdown.
HEIGHT_BINS = ((0, 20), (20, 30), (30, 45), (45, 70), (70, 10_000))


@dataclass(frozen=True)
class BenchmarkImage:
    """One labeled plate.

    Attributes:
        path: Image file.
        region: Benchmark subset the image belongs to.
        box: Labeled plate box.
        text: Normalized ground-truth serial.
    """

    path: Path
    region: str
    box: Box
    text: str


def load_openalpr(root: str | Path, regions: Sequence[str] = REGIONS) -> list[BenchmarkImage]:
    """Load the OpenALPR end-to-end benchmark.

    Args:
        root: The benchmark's ``endtoend`` directory, or its parent.
        regions: Subsets to load, from ``"us"``, ``"eu"`` and ``"br"``.

    Raises:
        FileNotFoundError: If a region folder is missing.
    """
    root = Path(root)
    if (root / "endtoend").is_dir():
        root = root / "endtoend"
    images = []
    for region in regions:
        folder = root / region
        if not folder.is_dir():
            raise FileNotFoundError(
                f"{folder} not found; run scripts/download_data.py openalpr first"
            )
        for label_file in sorted(folder.glob("*.txt")):
            fields = label_file.read_text().strip().split("\t")
            if len(fields) != 6:
                logger.warning("skipping malformed label %s", label_file)
                continue
            name, x, y, width, height, text = fields
            x, y, width, height = (float(v) for v in (x, y, width, height))
            image_path = folder / name
            if not image_path.exists():
                logger.warning("skipping %s: image %s is missing", label_file.name, name)
                continue
            images.append(
                BenchmarkImage(
                    image_path, region, Box(x, y, x + width, y + height), normalize(text)
                )
            )
    return images


def _height_bin(height: float) -> str:
    for low, high in HEIGHT_BINS:
        if low <= height < high:
            return f"{low}-{high}px" if high < 10_000 else f"{low}px+"
    return "unknown"


def _summarize(records: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    """Score ``records`` overall, per region and per plate-height bin."""

    def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return score_readings(
            [r[prediction_key] for r in rows], [r["truth"] for r in rows]
        ).as_dict()

    by_region: dict[str, list] = defaultdict(list)
    by_height: dict[str, list] = defaultdict(list)
    for record in records:
        by_region[record["region"]].append(record)
        by_height[_height_bin(record["plate_height"])].append(record)
    bin_names = [_height_bin(low) for low, _ in HEIGHT_BINS]
    confusions = substitution_counts(
        [r[prediction_key] for r in records], [r["truth"] for r in records]
    )
    return {
        "overall": score(records),
        "by_region": {region: score(rows) for region, rows in sorted(by_region.items())},
        "by_plate_height": {name: score(by_height[name]) for name in bin_names if by_height[name]},
        "top_confusions": [
            {"truth": truth, "predicted": predicted, "count": count}
            for (truth, predicted), count in confusions.most_common(15)
        ],
    }


def evaluate_recognition(
    images: Sequence[BenchmarkImage], ocr: PlateOCR, *, progress: bool = True
) -> dict[str, Any]:
    """Read every labeled plate box and score the reads.

    No plate format is applied: the benchmark mixes many jurisdictions, so
    the raw normalized OCR text is compared with the label.

    Returns:
        Scores overall, per region and per plate-height bin, the commonest
        character confusions, mean OCR latency, and one record per image.
    """
    records = []
    seconds = 0.0
    for item in tqdm(images, desc=f"ocr {ocr.name}", disable=not progress):
        image = cv2.imread(str(item.path))
        if image is None:
            logger.warning("could not read %s", item.path)
            continue
        plate_crop = crop(image, item.box)
        start = time.perf_counter()
        read = ocr.read(plate_crop)
        seconds += time.perf_counter() - start
        records.append(
            {
                "image": item.path.name,
                "region": item.region,
                "truth": item.text,
                "prediction": read.text if read else None,
                "confidence": read.confidence if read else None,
                "plate_height": item.box.height,
            }
        )
    return {
        "ocr": ocr.name,
        "preprocess": ocr.preprocess_name,
        "ms_per_plate": 1000 * seconds / max(len(records), 1),
        **_summarize(records, "prediction"),
        "records": records,
    }


def evaluate_end_to_end(
    images: Sequence[BenchmarkImage],
    plate_detector: Detector,
    ocr: PlateOCR,
    *,
    iou_threshold: float = 0.5,
    progress: bool = True,
) -> dict[str, Any]:
    """Detect each plate, read the detection, and score both steps.

    A labeled plate counts as detected if some detection overlaps it with at
    least ``iou_threshold`` IoU; that detection's crop is then read. Only
    recall is reported for detection, not precision: each image labels a
    single plate, so other detections may be real, unlabeled plates.

    Returns:
        Detection recall and mean IoU, reading scores (undetected plates count
        as wrong), and one record per image.
    """
    records = []
    for item in tqdm(images, desc="end to end", disable=not progress):
        image = cv2.imread(str(item.path))
        if image is None:
            logger.warning("could not read %s", item.path)
            continue
        detections = plate_detector.detect(image)
        overlaps = iou_matrix(
            boxes_to_array([item.box]), boxes_to_array([d.box for d in detections])
        )[0]
        best = int(np.argmax(overlaps)) if len(overlaps) else None
        detected = best is not None and overlaps[best] >= iou_threshold
        read = ocr.read(crop(image, detections[best].box)) if detected else None
        records.append(
            {
                "image": item.path.name,
                "region": item.region,
                "truth": item.text,
                "detected": bool(detected),
                "best_iou": float(overlaps[best]) if best is not None else 0.0,
                "prediction": read.text if read else None,
                "plate_height": item.box.height,
            }
        )

    by_region: dict[str, list] = defaultdict(list)
    for record in records:
        by_region[record["region"]].append(record)

    def detection(rows: list[dict[str, Any]]) -> dict[str, Any]:
        found = sum(r["detected"] for r in rows)
        return {
            "n": len(rows),
            "detected": found,
            "recall": found / len(rows) if rows else 0.0,
            "mean_iou_when_detected": (
                float(np.mean([r["best_iou"] for r in rows if r["detected"]])) if found else 0.0
            ),
        }

    return {
        "ocr": ocr.name,
        "preprocess": ocr.preprocess_name,
        "iou_threshold": iou_threshold,
        "detection": {
            "overall": detection(records),
            "by_region": {region: detection(rows) for region, rows in sorted(by_region.items())},
        },
        "reading": _summarize(records, "prediction"),
        "records": records,
    }
