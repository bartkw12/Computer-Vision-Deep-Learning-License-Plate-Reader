"""Training and validating the license plate detector with Ultralytics."""

from __future__ import annotations

import csv
import logging
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from alpr.config import parse_overrides
from alpr.detection import resolve_device
from alpr.provenance import environment_info

logger = logging.getLogger(__name__)

#: ``results.csv`` columns of an Ultralytics detection run, and the names used
#: for them in results files. All are measured on the validation split.
VALIDATION_COLUMNS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "val/box_loss": "box_loss",
    "val/cls_loss": "cls_loss",
    "val/dfl_loss": "dfl_loss",
}

#: Training arguments compared between a recorded run and a config even when
#: the config leaves them out, because their defaults can change between
#: Ultralytics releases.
KEY_TRAIN_ARGS = (
    "epochs",
    "batch",
    "imgsz",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "patience",
    "seed",
    "deterministic",
)

#: Config entries that name files or output folders rather than settings.
_LOCATION_ARGS = frozenset({"model", "data", "project", "name", "exist_ok"})


def load_train_args(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Read training arguments from YAML and apply ``key=value`` overrides.

    The file holds a ``model`` entry (the weights to start from) plus any
    arguments accepted by Ultralytics' ``YOLO.train``.

    Raises:
        ValueError: If the file has no ``model`` or ``data`` entry.
    """
    args = yaml.safe_load(Path(path).read_text()) or {}
    args.update(parse_overrides(overrides))
    for required in ("model", "data"):
        if not args.get(required):
            raise ValueError(f"{path}: training config needs a {required!r} entry")
    return args


def train(args: dict[str, Any], *, export_to: str | Path | None = None) -> Path:
    """Fine-tune a YOLO model.

    Args:
        args: Training arguments as returned by :func:`load_train_args`.
        export_to: If given, the best checkpoint is copied here afterwards.

    Returns:
        Path of the best checkpoint inside the Ultralytics run directory.
    """
    from ultralytics import YOLO

    args = dict(args)
    model = YOLO(str(args.pop("model")))
    args["data"] = str(Path(args["data"]).resolve())
    if "project" in args:
        # An absolute project path keeps Ultralytics from relocating the run.
        args["project"] = str(Path(args["project"]).resolve())
    args.setdefault("device", resolve_device())
    model.train(**args)

    best = Path(model.trainer.best)
    logger.info("best checkpoint: %s", best)
    if export_to is not None:
        export_to = Path(export_to)
        export_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, export_to)
        logger.info("copied to %s", export_to)
    return best


def validate(
    weights: str | Path,
    data: str | Path,
    *,
    split: str = "test",
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    plots_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a detector on one split of a YOLO dataset.

    Args:
        weights: Model checkpoint.
        data: Dataset YAML.
        split: ``"train"``, ``"val"`` or ``"test"``. Report ``"test"``: the
            validation split is used during training to choose the best
            checkpoint, so numbers on it are optimistic.
        imgsz: Inference image size.
        batch: Batch size.
        device: Inference device; ``None`` picks a GPU when available.
        plots_dir: Directory for Ultralytics' curves and confusion matrix. A
            temporary run directory is used if omitted.

    Returns:
        Precision, recall, mAP50 and mAP50-95 with counts, per-image speed and
        provenance, ready to be written as JSON.
    """
    from ultralytics import YOLO

    device = resolve_device(device)
    model = YOLO(str(weights))
    options: dict[str, Any] = {
        "data": str(Path(data).resolve()),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "plots": True,
        "verbose": False,
    }
    if plots_dir is not None:
        plots_dir = Path(plots_dir).resolve()
        options.update(project=str(plots_dir.parent), name=plots_dir.name, exist_ok=True)
    metrics = model.val(**options)

    validator = model.validator
    return {
        "weights": str(weights),
        "data": str(data),
        "split": split,
        "imgsz": imgsz,
        "images": int(validator.seen),
        "instances": int(sum(validator.metrics.nt_per_class)),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
        "parameters": int(sum(p.numel() for p in model.model.parameters())),
        "plots_dir": str(metrics.save_dir),
        "environment": environment_info(device),
    }


def read_results_csv(path: str | Path) -> list[dict[str, float]]:
    """Read an Ultralytics ``results.csv``, one dictionary per epoch.

    Older releases pad the column names with spaces; they are stripped.
    """
    with Path(path).open(newline="") as handle:
        return [
            {key.strip(): float(value) for key, value in row.items() if key and value.strip()}
            for row in csv.DictReader(handle)
        ]


def _validation_metrics(row: Mapping[str, float]) -> dict[str, float]:
    metrics = {name: row[column] for column, name in VALIDATION_COLUMNS.items() if column in row}
    if "epoch" in row:
        metrics = {"epoch": int(row["epoch"]), **metrics}
    return metrics


def _matching_epoch(rows: Sequence[Mapping[str, float]], metrics: Mapping[str, Any]) -> int | None:
    """Find the epoch whose validation metrics are ``metrics`` (a checkpoint's)."""
    shared = [column for column in VALIDATION_COLUMNS if column in metrics]
    if not shared:
        return None
    for row in rows:
        if all(
            column in row and math.isclose(row[column], float(metrics[column]), abs_tol=1e-5)
            for column in shared
        ):
            return int(row["epoch"]) if "epoch" in row else None
    return None


def config_differences(
    run_args: Mapping[str, Any],
    config: Mapping[str, Any],
    keys: Iterable[str] = KEY_TRAIN_ARGS,
) -> list[dict[str, Any]]:
    """List the settings in which a training config differs from a recorded run.

    Every setting in ``config`` is compared, plus each of ``keys`` that the run
    recorded. Entries that name files or folders are skipped.

    Returns:
        ``{"key", "run", "config"}`` per difference; ``config`` is ``None``
        when the config leaves the setting to the Ultralytics default.
    """
    compared = (set(config) | {key for key in keys if key in run_args}) - _LOCATION_ARGS
    return [
        {"key": key, "run": run_args.get(key), "config": config.get(key)}
        for key in sorted(compared)
        if run_args.get(key) != config.get(key)
    ]


def summarize_training_run(
    run_dir: str | Path, weights: str | Path | None = None
) -> dict[str, Any]:
    """Recover what an earlier Ultralytics training run did and measured.

    Args:
        run_dir: The run folder, holding ``args.yaml`` and ``results.csv``.
        weights: Optionally the run's best checkpoint. It is loaded with full
            unpickling, so pass only a checkpoint you trust (your own).

    Returns:
        The run's arguments, epoch count and training time, validation
        metrics at the final epoch, at the epoch with the highest mAP50-95,
        and (given ``weights``) at the epoch the checkpoint was saved from,
        plus the checkpoint's date and Ultralytics version.
    """
    run_dir = Path(run_dir)
    run_args = yaml.safe_load((run_dir / "args.yaml").read_text()) or {}
    rows = read_results_csv(run_dir / "results.csv")
    if not rows:
        raise ValueError(f"{run_dir / 'results.csv'} has no epochs")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "args": run_args,
        "epochs_run": len(rows),
        "metrics_split": "val",
        "final_epoch": _validation_metrics(rows[-1]),
        "highest_mAP50-95_epoch": _validation_metrics(
            max(rows, key=lambda row: row.get("metrics/mAP50-95(B)", -1.0))
        ),
    }
    if "time" in rows[-1]:
        summary["training_hours"] = rows[-1]["time"] / 3600
    if weights is not None:
        import torch

        checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
        saved_metrics = checkpoint.get("train_metrics") or {}
        summary["checkpoint"] = {
            "weights": str(weights),
            "date": checkpoint.get("date"),
            "ultralytics_version": checkpoint.get("version"),
            "best_fitness": checkpoint.get("best_fitness"),
            "epoch": _matching_epoch(rows, saved_metrics),
            "metrics": {
                name: float(saved_metrics[column])
                for column, name in VALIDATION_COLUMNS.items()
                if column in saved_metrics
            },
            "args_differ_from_args_yaml": config_differences(
                run_args, checkpoint.get("train_args") or {}, keys=run_args
            ),
        }
    return summary
