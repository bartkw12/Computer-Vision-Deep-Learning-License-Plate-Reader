"""Look for near-duplicate images shared between the plate dataset's splits.

Usage::

    python scripts/check_leakage.py                 # data/plates
    python scripts/check_leakage.py DATA_DIR --sheet outputs/leakage_pairs.jpg

A test split that contains near-copies of training images overstates how well
the detector generalizes. This compares every ``valid`` and ``test`` image
with its nearest neighbour in the earlier splits (see
``alpr.evaluation.leakage``), writes the counts to
``results/dataset_leakage.json``, and saves the closest pairs side by side so
they can be checked by eye: a small hash distance is a hint, not a verdict.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from alpr.evaluation.leakage import DEFAULT_THRESHOLDS, leakage_report
from alpr.provenance import environment_info
from alpr.runs import write_json

logger = logging.getLogger("leakage")

_CELL = (320, 240)


def _cell(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        return np.zeros((_CELL[1], _CELL[0], 3), np.uint8)
    return cv2.resize(image, _CELL, interpolation=cv2.INTER_AREA)


def write_pair_sheet(report: dict, root: Path, out: Path, per_pair: int = 4) -> None:
    """Save the closest pairs of every split comparison as one image."""
    strips = []
    for name, comparison in report["pairs"].items():
        for pair in comparison.get("closest_pairs", [])[:per_pair]:
            strip = np.hstack([_cell(root / pair["query"]), _cell(root / pair["reference"])])
            caption = f"{name}  d={pair['distance']}  {pair['query']}"
            cv2.putText(strip, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(strip, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            strips.append(strip)
    if not strips:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(strips))


def main() -> None:
    """Hash the splits, compare them, and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, nargs="?", default=Path("data/plates"))
    parser.add_argument("--out", type=Path, default=Path("results/dataset_leakage.json"))
    parser.add_argument("--sheet", type=Path, default=Path("outputs/leakage_pairs.jpg"))
    parser.add_argument(
        "--thresholds", type=int, nargs="+", default=list(DEFAULT_THRESHOLDS), metavar="BITS"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress bars")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = leakage_report(args.data, thresholds=args.thresholds, progress=not args.quiet)
    if not any(report["split_sizes"].values()):
        parser.error(f"no images found under {args.data}/{{train,valid,test}}")
    report["dataset"] = str(args.data)
    report["environment"] = environment_info()
    write_json(args.out, report)

    for name, comparison in report["pairs"].items():
        within = ", ".join(
            f"<={bits}: {row['count']} ({100 * row['share']:.1f}%)"
            for bits, row in comparison["within"].items()
        )
        logger.info("%-15s %5d images  %s", name, comparison["queries"], within)
    write_pair_sheet(report, args.data, args.sheet)
    logger.info("report -> %s, closest pairs -> %s", args.out, args.sheet)


if __name__ == "__main__":
    main()
