"""Command-line interface: ``alpr <command>``.

Commands::

    run          read the plates in a video
    render       draw a run's results onto its video
    train        fine-tune the plate detector
    val          measure the plate detector on a dataset split
    eval-images  measure plate reading on the OpenALPR still-image benchmark
    label-sheet  create a ground-truth sheet for a video run
    eval-video   score a video run against a filled-in label sheet
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from alpr import __version__
from alpr.config import load_config

logger = logging.getLogger("alpr")


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML file with pipeline settings")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one setting, e.g. --set tracker.max_age=10 (repeatable)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpr", description="License plate recognition for video."
    )
    parser.add_argument("--version", action="version", version=f"alpr {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress bars or info logs")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    run = commands.add_parser("run", help="read the plates in a video")
    run.add_argument("video", type=Path)
    run.add_argument("--out", type=Path, help="run directory (default: outputs/<video name>)")
    run.add_argument("--max-frames", type=int, help="stop after this many frames")
    run.add_argument("--render", action="store_true", help="also write annotated.mp4")
    _add_config_arguments(run)
    run.set_defaults(handler=_run)

    render = commands.add_parser("render", help="draw a run's results onto its video")
    render.add_argument("run_dir", type=Path)
    render.add_argument("--video", type=Path, help="video file, if it has moved since the run")
    render.add_argument("--out", type=Path, help="output file (default: <run_dir>/annotated.mp4)")
    render.add_argument("--no-insets", action="store_true", help="do not show plate crops")
    render.set_defaults(handler=_render)

    train = commands.add_parser("train", help="fine-tune the plate detector")
    train.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_plate_detector.yaml"),
        help="training YAML",
    )
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument(
        "--export", type=Path, default=Path("weights/plate_yolo11n.pt"), help="copy best.pt here"
    )
    train.set_defaults(handler=_train)

    val = commands.add_parser("val", help="measure the plate detector on a dataset split")
    val.add_argument("--weights", type=Path, default=Path("weights/plate_yolo11n.pt"))
    val.add_argument("--data", type=Path, default=Path("data/plates/data.yaml"))
    val.add_argument("--split", default="test", choices=["train", "val", "test"])
    val.add_argument("--imgsz", type=int, default=640)
    val.add_argument("--batch", type=int, default=16)
    val.add_argument("--device", help="cpu, cuda:0, ... (default: GPU if available)")
    val.add_argument("--plots", type=Path, help="directory for curves and confusion matrix")
    val.add_argument("--out", type=Path, required=True, help="results JSON to write")
    val.set_defaults(handler=_val)

    images = commands.add_parser("eval-images", help="measure plate reading on OpenALPR stills")
    images.add_argument("--data", type=Path, default=Path("data/openalpr"))
    images.add_argument("--regions", nargs="+", default=["us", "eu", "br"])
    images.add_argument("--ocr", default="easyocr", help="OCR backend")
    images.add_argument("--preprocess", default="none", help="crop preprocessing")
    images.add_argument(
        "--plate-weights", type=Path, help="plate detector; adds the end-to-end evaluation"
    )
    images.add_argument("--device", help="cpu, cuda:0, ... (default: GPU if available)")
    images.add_argument("--limit", type=int, help="use only the first N images per region")
    images.add_argument("--out", type=Path, required=True, help="results JSON to write")
    images.set_defaults(handler=_eval_images)

    sheet = commands.add_parser("label-sheet", help="create a ground-truth sheet for a run")
    sheet.add_argument("run_dir", type=Path)
    sheet.add_argument("--out", type=Path, required=True, help="CSV to create")
    sheet.add_argument("--video", type=Path, help="video file, if it has moved since the run")
    sheet.add_argument("--min-frames", type=int, default=5, help="skip tracks shorter than this")
    sheet.add_argument("--overwrite", action="store_true", help="replace an existing sheet")
    sheet.set_defaults(handler=_label_sheet)

    video = commands.add_parser("eval-video", help="score a run against a filled-in label sheet")
    video.add_argument("run_dir", type=Path)
    video.add_argument("--labels", type=Path, required=True, help="filled-in label sheet CSV")
    video.add_argument("--out", type=Path, required=True, help="summary JSON (no plate strings)")
    video.set_defaults(handler=_eval_video)

    return parser


def _print_plates(results) -> None:
    if not results:
        print("No plates read.")
        return
    print(f"{'track':>5}  {'plate':<10} {'conf':>5} {'reads':>5}  frames")
    for r in results:
        print(
            f"{r.track_id:>5}  {r.text:<10} {r.confidence:>5.2f} {r.n_reads:>5}  "
            f"{r.first_frame}-{r.last_frame}"
        )


def _run(args: argparse.Namespace) -> int:
    from alpr.pipeline import build_pipeline
    from alpr.runs import process_video, save_run
    from alpr.visualize import render_video

    config = load_config(args.config, args.set)
    pipeline = build_pipeline(config)
    video = process_video(pipeline, args.video, max_frames=args.max_frames, progress=not args.quiet)
    out_dir = save_run(args.out or Path("outputs") / args.video.stem, pipeline, video)
    results = pipeline.results()
    _print_plates(results)
    timing = pipeline.timer.summary(pipeline.frames_processed)
    print(f"{timing['frames']} frames at {timing['fps']:.1f} FPS -> {out_dir}")
    if args.render:
        render_video(
            args.video,
            pipeline.observations,
            results,
            out_dir / "annotated.mp4",
            progress=not args.quiet,
        )
    return 0


def _render(args: argparse.Namespace) -> int:
    from alpr.runs import load_run
    from alpr.visualize import render_video

    run = load_run(args.run_dir)
    output = render_video(
        args.video or run.video_path,
        run.observations,
        run.results,
        args.out or args.run_dir / "annotated.mp4",
        show_insets=not args.no_insets,
        progress=not args.quiet,
    )
    print(output)
    return 0


def _train(args: argparse.Namespace) -> int:
    from alpr.training import load_train_args, train

    best = train(load_train_args(args.config, args.set), export_to=args.export)
    print(f"best checkpoint: {best}\nexported to: {args.export}")
    return 0


def _val(args: argparse.Namespace) -> int:
    from alpr.runs import write_json
    from alpr.training import validate

    metrics = validate(
        args.weights,
        args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots_dir=args.plots,
    )
    write_json(args.out, metrics)
    print(
        f"{args.split}: {metrics['images']} images, {metrics['instances']} plates | "
        f"P {metrics['precision']:.3f}  R {metrics['recall']:.3f}  "
        f"mAP50 {metrics['mAP50']:.3f}  mAP50-95 {metrics['mAP50-95']:.3f} -> {args.out}"
    )
    return 0


def _eval_images(args: argparse.Namespace) -> int:
    from alpr.detection import YoloDetector, resolve_device
    from alpr.evaluation.images import evaluate_end_to_end, evaluate_recognition, load_openalpr
    from alpr.ocr import build_ocr
    from alpr.provenance import environment_info
    from alpr.runs import write_json

    device = resolve_device(args.device)
    images = load_openalpr(args.data, args.regions)
    if args.limit:
        images = [
            image
            for region in args.regions
            for image in [i for i in images if i.region == region][: args.limit]
        ]
    ocr = build_ocr(args.ocr, preprocess=args.preprocess, gpu=device.startswith("cuda"))
    report = {
        "dataset": "OpenALPR benchmark, endtoend",
        "regions": args.regions,
        "recognition": evaluate_recognition(images, ocr, progress=not args.quiet),
    }
    score = report["recognition"]["overall"]
    print(
        f"recognition on labeled boxes: {score['exact']}/{score['n']} exact "
        f"({score['accuracy']:.1%}), CER {score['cer']:.3f}"
    )
    if args.plate_weights:
        detector = YoloDetector(args.plate_weights, device=device)
        report["plate_weights"] = str(args.plate_weights)
        report["end_to_end"] = evaluate_end_to_end(images, detector, ocr, progress=not args.quiet)
        detection = report["end_to_end"]["detection"]["overall"]
        score = report["end_to_end"]["reading"]["overall"]
        print(
            f"end to end: plate recall {detection['recall']:.1%}, "
            f"{score['exact']}/{score['n']} exact ({score['accuracy']:.1%}), CER {score['cer']:.3f}"
        )
    report["environment"] = environment_info(device)
    write_json(args.out, report)
    print(f"-> {args.out}")
    return 0


def _label_sheet(args: argparse.Namespace) -> int:
    from alpr.evaluation.video import write_label_sheet
    from alpr.runs import load_run

    run = load_run(args.run_dir)
    count = write_label_sheet(
        run.observations,
        args.video or run.video_path,
        args.out,
        min_frames=args.min_frames,
        overwrite=args.overwrite,
    )
    print(
        f"{count} vehicles to label in {args.out}\n"
        f"Open the {args.out.stem}_sheet_*.jpg images next to it, and type each plate into the\n"
        f"'plate' column. Use ? where the plate cannot be read by eye."
    )
    return 0


def _eval_video(args: argparse.Namespace) -> int:
    from alpr.evaluation.video import evaluate_video, read_labels
    from alpr.plate_format import get_formats
    from alpr.provenance import environment_info
    from alpr.runs import load_run, write_json

    run = load_run(args.run_dir)
    labels = read_labels(args.labels)
    formats = get_formats(run.meta["config"]["plate_format"]["name"])
    evaluation = evaluate_video(run.observations, run.results, labels, formats=formats)
    summary = evaluation["summary"]
    if summary["unlabeled"]:
        logger.warning("%d rows of %s are still empty", summary["unlabeled"], args.labels)

    write_json(
        args.out,
        {
            "summary": summary,
            "config": run.meta["config"],
            "video": {k: v for k, v in run.meta["video"].items() if k != "path"},
            "environment": environment_info(),
        },
    )
    # Per-vehicle records contain real plate strings: keep them with the run.
    write_json(args.run_dir / "eval_records.json", evaluation["records"])
    reading = summary["reading"]
    print(
        f"{reading['exact']}/{reading['n']} labeled vehicles read correctly "
        f"({reading['accuracy']:.1%}, 95% CI {reading['ci_low']:.1%}-{reading['ci_high']:.1%}), "
        f"CER {reading['cer']:.3f}, precision {summary['precision']:.1%} -> {args.out}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s", force=True
    )
    # The model libraries emit deprecation chatter that is not actionable here.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        return args.handler(args)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
