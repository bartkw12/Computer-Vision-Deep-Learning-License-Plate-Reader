"""Training and validating the license plate detector with Ultralytics."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from alpr.config import parse_overrides
from alpr.detection import resolve_device
from alpr.provenance import environment_info

logger = logging.getLogger(__name__)


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
