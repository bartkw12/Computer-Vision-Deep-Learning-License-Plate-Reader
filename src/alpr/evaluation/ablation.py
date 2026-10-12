"""Ablation experiments on a labeled video.

Two kinds of experiment:

* :func:`replay_grid` varies the plate format rules and the aggregation
  strategy over one logged run. No model runs again, because the log keeps the
  raw OCR text of every frame.
* :func:`pipeline_grid` re-runs the whole pipeline once per configuration, for
  settings that change what the models see or how often they run: tracker
  lifetime, where plates are searched for, OCR backend and preprocessing, and
  how often OCR is invoked.

Every row is scored against the same label sheet, which is possible because
labels are anchored to frames and pixels rather than to track IDs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpr.config import PipelineConfig, load_config, parse_overrides
from alpr.evaluation.video import Label, evaluate_video
from alpr.pipeline import PlateReaderPipeline, read_tracks
from alpr.plate_format import get_formats
from alpr.runs import process_video
from alpr.types import TrackObservation

logger = logging.getLogger(__name__)

AGGREGATION_STRATEGIES = ("max_conf", "string_vote", "char_vote")


@dataclass(frozen=True)
class Variant:
    """One configuration to evaluate.

    Attributes:
        group: The question this variant helps answer, e.g. ``"tracker"``.
        name: Short label for tables.
        overrides: ``key=value`` settings applied on top of the base config.
    """

    group: str
    name: str
    overrides: tuple[str, ...] = ()


def format_modes(base_format: str) -> list[tuple[str, str, int]]:
    """Format-correction settings to compare: ``(label, format set, max substitutions)``.

    ``off`` uses no format knowledge. ``regex`` is what the first version of
    this project effectively did: an exact structural match with no knowledge
    of unused letters and no correction. The remaining rows use ``base_format``
    with increasing tolerance for look-alike characters.
    """
    modes = [("off", "any", 0)]
    if base_format == "ontario":
        modes.append(("regex", "4L3D", 0))
    modes += [
        ("strict", base_format, 0),
        ("lookalike-1", base_format, 1),
        ("lookalike-2", base_format, 2),
    ]
    return modes


def _score(
    observations: Sequence[TrackObservation],
    results: Sequence,
    labels: Sequence[Label],
    base_format: str,
) -> dict[str, Any]:
    summary = evaluate_video(observations, results, labels, formats=get_formats(base_format))[
        "summary"
    ]
    return {
        **summary["reading"],
        "emitted": summary["emitted"],
        "precision": summary["precision"],
        "tracked": summary["tracked"],
        "tracks_per_vehicle": summary["tracks_per_vehicle"],
        "tracks_in_run": summary["tracks_in_run"],
    }


def replay_grid(
    observations: Sequence[TrackObservation], labels: Sequence[Label], base_format: str
) -> list[dict[str, Any]]:
    """Score every format mode with every aggregation strategy on one log.

    Args:
        observations: A run's per-frame log. The run must have read every
            plate it saw (``ocr.every=1``), or the comparison is starved.
        labels: Ground truth for the video.
        base_format: The plate format set the video's plates follow.

    Returns:
        One row per combination, with reading scores and precision.
    """
    rows = []
    for label, format_name, max_substitutions in format_modes(base_format):
        for strategy in AGGREGATION_STRATEGIES:
            results = read_tracks(
                observations,
                formats=get_formats(format_name),
                max_substitutions=max_substitutions,
                strategy=strategy,
            )
            rows.append(
                {
                    "format_mode": label,
                    "format": format_name,
                    "max_substitutions": max_substitutions,
                    "aggregation": strategy,
                    **_score(observations, results, labels, base_format),
                }
            )
    return rows


def default_variants() -> list[Variant]:
    """The configurations compared by :func:`pipeline_grid` unless told otherwise."""
    variants = [Variant("baseline", "defaults")]
    variants += [
        Variant("tracker", f"max_age={age}", (f"tracker.max_age={age}",)) for age in (1, 5, 15, 60)
    ]
    variants.append(Variant("plate_search", "per-vehicle crops", ("plate_search=vehicle",)))
    variants += [
        Variant("ocr_backend", backend, (f"ocr.backend={backend}",))
        for backend in ("easyocr", "easyocr-allowlist", "easyocr-recognize", "fast-plate-ocr")
    ]
    variants += [
        Variant("ocr_preprocess", step, (f"ocr.preprocess={step}",))
        for step in ("gray", "otsu_inv", "clahe", "upscale")
    ]
    variants += [Variant("ocr_cost", f"every={n}", (f"ocr.every={n}",)) for n in (2, 3, 5, 10)]
    variants += [
        Variant("ocr_cost", f"min_plate_height={h}", (f"ocr.min_plate_height={h}",))
        for h in (12, 16, 20)
    ]
    return variants


def pipeline_grid(
    video_path: str | Path,
    labels: Sequence[Label],
    variants: Sequence[Variant],
    pipeline_factory: Callable[[PipelineConfig], PlateReaderPipeline],
    *,
    base_config: str | Path | None = None,
    base_overrides: Sequence[str] = (),
    max_frames: int | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Run the pipeline once per variant and score each run.

    Args:
        video_path: The labeled video.
        labels: Ground truth for the video.
        variants: Configurations to evaluate.
        pipeline_factory: Builds a pipeline from a configuration; normally
            :func:`alpr.pipeline.build_pipeline`.
        base_config: YAML file every variant starts from.
        base_overrides: Settings applied to every variant before its own.
        max_frames: Process only this many frames of the video.
        progress: Show progress bars.

    Returns:
        One row per variant: its settings, reading scores, precision, track
        counts, throughput, per-stage latency and OCR call count. A variant
        identical to the base configuration is run like any other, so its row
        doubles as a check on run-to-run consistency.
    """
    rows = []
    for variant in variants:
        overrides = [*base_overrides, *variant.overrides]
        config = load_config(base_config, overrides)
        logger.info("variant %s / %s", variant.group, variant.name)
        pipeline = pipeline_factory(config)
        process_video(pipeline, video_path, max_frames=max_frames, progress=progress)
        timing = pipeline.timer.summary(pipeline.frames_processed)
        rows.append(
            {
                "group": variant.group,
                "variant": variant.name,
                "overrides": parse_overrides(variant.overrides),
                **_score(
                    pipeline.observations, pipeline.results(), labels, config.plate_format.name
                ),
                "fps": timing["fps"],
                "ms_per_frame": timing["ms_per_frame"],
                "stage_ms_per_frame": {
                    name: stage["ms_per_frame"] for name, stage in timing["stages"].items()
                },
                "ocr_calls": pipeline.counters["ocr_calls"],
            }
        )
    return rows
