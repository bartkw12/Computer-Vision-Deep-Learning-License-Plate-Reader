r"""Recover what the original plate detector training run did and measured.

Usage::

    python scripts/recover_training_run.py \
        --run runs/original/train3 --weights weights/plate_yolo11n_original.pt

The first version of this project reported metrics from a training run whose
code was never committed. Its Ultralytics run folder still records the
arguments (``args.yaml``) and per-epoch validation metrics (``results.csv``),
and the checkpoint records when and with which Ultralytics release it was
made. This collects them into ``results/detector_training_original.json``
and lists where ``configs/train_plate_detector.yaml`` differs from the run,
so the config can be made to reproduce it.

Every metric in the output is on the **validation** split, the one used to
pick the best checkpoint. Held-out numbers come from ``alpr val --split test``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from alpr.provenance import environment_info
from alpr.runs import write_json
from alpr.training import config_differences, summarize_training_run

logger = logging.getLogger("recover")


def main() -> None:
    """Summarize the run and compare it with the training config."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, default=Path("runs/original/train3"))
    parser.add_argument("--weights", type=Path, help="the run's best.pt (trusted: fully unpickled)")
    parser.add_argument("--config", type=Path, default=Path("configs/train_plate_detector.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/detector_training_original.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    summary = summarize_training_run(args.run, args.weights)
    config = yaml.safe_load(args.config.read_text()) or {}
    summary["config"] = str(args.config)
    summary["config_differences"] = config_differences(summary["args"], config)
    summary["environment"] = environment_info()
    write_json(args.out, summary)

    final = summary["final_epoch"]
    logger.info(
        "%d epochs; final epoch (val): P %.3f  R %.3f  mAP50 %.3f  mAP50-95 %.3f",
        summary["epochs_run"],
        final.get("precision", float("nan")),
        final.get("recall", float("nan")),
        final.get("mAP50", float("nan")),
        final.get("mAP50-95", float("nan")),
    )
    for row in summary["config_differences"]:
        logger.info(
            "config differs: %-14s run=%r  config=%r", row["key"], row["run"], row["config"]
        )
    logger.info("summary -> %s", args.out)


if __name__ == "__main__":
    main()
