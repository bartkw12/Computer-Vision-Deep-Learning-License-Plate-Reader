"""Run the ablation experiments and write their results under ``results/``.

Usage::

    python scripts/run_ablations.py stills
    python scripts/run_ablations.py video-replay RUN_DIR --labels LABELS.csv
    python scripts/run_ablations.py video-grid VIDEO --labels LABELS.csv

``stills`` works on the labeled plate boxes of the OpenALPR benchmark and
needs no trained plate detector. It answers:

* Does preprocessing the crop help OCR, and which kind?
* How does a general scene-text reader compare with a plate-specific one?
* Does format-aware correction help? (On the Brazilian subset, whose 114
  plates all share one format.)

``video-replay`` varies the plate format rules and the aggregation strategy
over one logged run of a labeled video, without running any model again.

``video-grid`` re-runs the pipeline on a labeled video once per configuration:
tracker lifetime, where plates are searched for, OCR backend and
preprocessing, and how often OCR runs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from alpr.detection import resolve_device
from alpr.evaluation.ablation import default_variants, format_modes, pipeline_grid, replay_grid
from alpr.evaluation.images import evaluate_recognition, load_openalpr
from alpr.evaluation.metrics import score_readings
from alpr.evaluation.video import read_labels
from alpr.ocr import build_ocr
from alpr.pipeline import build_pipeline
from alpr.plate_format import fit_plate, get_formats
from alpr.provenance import environment_info
from alpr.runs import load_run, write_json

logger = logging.getLogger("ablations")

#: (backend, preprocessing, model) combinations compared on still images.
#: ``otsu_inv`` is the preprocessing the first version of this project used.
_PREPROCESSING = ("none", "gray", "otsu_inv", "clahe", "upscale", "upscale+clahe")
STILL_VARIANTS: list[tuple[str, str, str | None]] = [
    *(
        (backend, step, None)
        for backend in ("easyocr", "easyocr-allowlist")
        for step in _PREPROCESSING
    ),
    ("easyocr-recognize", "none", None),
    ("easyocr-recognize", "upscale", None),
    ("easyocr-recognize-allowlist", "none", None),
    ("easyocr-recognize-allowlist", "upscale", None),
    ("fast-plate-ocr", "none", "cct-xs-v2-global-model"),
    ("fast-plate-ocr", "none", "cct-s-v2-global-model"),
]

#: The single format shared by every plate of the benchmark's "br" subset.
BRAZIL_FORMAT = "3L4D"


def _format_rows(records: list[dict[str, Any]], format_name: str) -> list[dict[str, Any]]:
    """Score one variant's reads under each format-correction mode."""
    truths = [r["truth"] for r in records]
    rows = []
    for label, mode_format, max_substitutions in format_modes(format_name):
        if label == "off":
            # No format knowledge at all: score the raw read, whatever its length.
            predictions = [record["prediction"] for record in records]
        else:
            formats = get_formats(mode_format)
            predictions = []
            for record in records:
                fit = (
                    fit_plate(record["prediction"], formats, max_substitutions=max_substitutions)
                    if record["prediction"]
                    else None
                )
                predictions.append(fit.text if fit else None)
        score = score_readings(predictions, truths)
        rows.append(
            {"mode": label, "emitted": sum(p is not None for p in predictions), **score.as_dict()}
        )
    return rows


def run_stills(args: argparse.Namespace) -> None:
    """Compare OCR variants on the benchmark's labeled plate boxes."""
    device = resolve_device(args.device)
    images = load_openalpr(args.data)
    if args.limit:
        images = images[:: max(1, len(images) // args.limit)]
    warmup = np.full((48, 144, 3), 255, np.uint8)

    variants, all_records = [], {}
    for backend, preprocess, model in STILL_VARIANTS:
        ocr = build_ocr(backend, preprocess=preprocess, model=model, gpu=device.startswith("cuda"))
        ocr.read(warmup)  # keep one-time setup out of the timing
        report = evaluate_recognition(images, ocr, progress=not args.quiet)
        records = report.pop("records")
        all_records[f"{report['ocr']}|{preprocess}"] = records
        brazil = [r for r in records if r["region"] == "br"]
        report["format_correction_br"] = _format_rows(brazil, BRAZIL_FORMAT) if brazil else []
        variants.append(report)
        overall = report["overall"]
        logger.info(
            "%-42s %-14s exact %5.1f%%  CER %.3f  %6.1f ms/plate",
            report["ocr"],
            preprocess,
            100 * overall["accuracy"],
            overall["cer"],
            report["ms_per_plate"],
        )

    write_json(
        args.out,
        {
            "experiment": "OCR variants on labeled plate boxes (recognition only)",
            "dataset": "OpenALPR benchmark, endtoend/{us,eu,br}",
            "plates": len(images),
            "variants": variants,
            "environment": environment_info(device),
        },
    )
    # Per-image predictions, for error analysis. Not meant to be committed.
    write_json(args.records, all_records)
    logger.info("summary -> %s, per-image records -> %s", args.out, args.records)


def run_video_replay(args: argparse.Namespace) -> None:
    """Vary format rules and aggregation strategy over one logged run."""
    run = load_run(args.run_dir)
    base_format = run.meta["config"]["plate_format"]["name"]
    rows = replay_grid(run.observations, read_labels(args.labels), base_format)
    for row in rows:
        logger.info(
            "%-12s %-12s exact %d/%d  precision %5.1f%%",
            row["format_mode"],
            row["aggregation"],
            row["exact"],
            row["n"],
            100 * row["precision"],
        )
    write_json(
        args.out,
        {
            "experiment": "format correction x aggregation, replayed from one run's log",
            "base_config": run.meta["config"],
            "video": {k: v for k, v in run.meta["video"].items() if k != "path"},
            "rows": rows,
            "environment": environment_info(),
        },
    )
    logger.info("-> %s", args.out)


def run_video_grid(args: argparse.Namespace) -> None:
    """Re-run the pipeline on a labeled video once per configuration."""
    variants = default_variants()
    if args.groups:
        variants = [v for v in variants if v.group in {"baseline", *args.groups}]
    rows = pipeline_grid(
        args.video,
        read_labels(args.labels),
        variants,
        build_pipeline,
        base_config=args.config,
        base_overrides=args.set,
        max_frames=args.max_frames,
        progress=not args.quiet,
    )
    for row in rows:
        logger.info(
            "%-14s %-22s exact %d/%d  tracks/vehicle %.2f  %5.1f FPS",
            row["group"],
            row["variant"],
            row["exact"],
            row["n"],
            row["tracks_per_vehicle"],
            row["fps"],
        )
    device = resolve_device(next((s.split("=")[1] for s in args.set if "device=" in s), None))
    write_json(
        args.out,
        {
            "experiment": "one pipeline run per configuration on a labeled video",
            "base_overrides": args.set,
            "rows": rows,
            "environment": environment_info(device),
        },
    )
    logger.info("-> %s", args.out)


def main() -> None:
    """Parse arguments and run the requested experiment."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress bars")
    experiments = parser.add_subparsers(dest="experiment", required=True)

    stills = experiments.add_parser("stills", help="OCR variants on OpenALPR plate boxes")
    stills.add_argument("--data", type=Path, default=Path("data/openalpr"))
    stills.add_argument("--device", help="cpu, cuda:0, ... (default: GPU if available)")
    stills.add_argument("--limit", type=int, help="evaluate on about this many plates")
    stills.add_argument("--out", type=Path, default=Path("results/ablation_ocr_stills.json"))
    stills.add_argument(
        "--records", type=Path, default=Path("outputs/ablation_ocr_stills_records.json")
    )
    stills.set_defaults(handler=run_stills)

    replay = experiments.add_parser("video-replay", help="format and aggregation over a run's log")
    replay.add_argument("run_dir", type=Path)
    replay.add_argument("--labels", type=Path, required=True)
    replay.add_argument("--out", type=Path, default=Path("results/ablation_video_replay.json"))
    replay.set_defaults(handler=run_video_replay)

    grid = experiments.add_parser("video-grid", help="one pipeline run per configuration")
    grid.add_argument("video", type=Path)
    grid.add_argument("--labels", type=Path, required=True)
    grid.add_argument("--config", type=Path, help="base pipeline YAML")
    grid.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    grid.add_argument("--groups", nargs="+", help="run only these variant groups")
    grid.add_argument("--max-frames", type=int)
    grid.add_argument("--out", type=Path, default=Path("results/ablation_video_grid.json"))
    grid.set_defaults(handler=run_video_grid)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.handler(args)


if __name__ == "__main__":
    main()
