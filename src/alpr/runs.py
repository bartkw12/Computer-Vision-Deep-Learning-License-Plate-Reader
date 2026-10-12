"""Running the pipeline over a video file and storing the outcome.

A run directory holds::

    tracks.csv    one row per track per frame (see alpr.io.OBSERVATION_COLUMNS)
    plates.csv    the final plate of each track
    timing.json   per-stage latency
    run.json      video properties, configuration, counters and provenance
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from tqdm import tqdm

from alpr.detection import resolve_device
from alpr.io import (
    VideoReader,
    read_observations,
    read_plate_results,
    write_observations,
    write_plate_results,
)
from alpr.pipeline import PlateReaderPipeline
from alpr.provenance import environment_info
from alpr.types import PlateResult, TrackObservation

logger = logging.getLogger(__name__)

TRACKS_FILE = "tracks.csv"
PLATES_FILE = "plates.csv"
TIMING_FILE = "timing.json"
RUN_FILE = "run.json"


def write_json(path: str | Path, data: Any) -> None:
    """Write ``data`` as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def process_video(
    pipeline: PlateReaderPipeline,
    video_path: str | Path,
    *,
    max_frames: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Feed a video through the pipeline.

    Args:
        pipeline: A freshly built pipeline.
        video_path: The video to read.
        max_frames: Stop after this many frames; ``None`` reads everything.
        progress: Show a progress bar.

    Returns:
        Properties of the video: path, fps, width, height.
    """
    with VideoReader(video_path) as video:
        total = video.frame_count or None
        if max_frames is not None:
            total = min(total, max_frames) if total else max_frames
        frames = islice(video, max_frames)
        pipeline.run(tqdm(frames, total=total, desc="frames", disable=not progress))
        return {
            "path": str(video_path),
            "fps": video.fps,
            "width": video.width,
            "height": video.height,
        }


def save_run(out_dir: str | Path, pipeline: PlateReaderPipeline, video: dict[str, Any]) -> Path:
    """Write a pipeline's log, results, timing and provenance to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_observations(out_dir / TRACKS_FILE, pipeline.observations)
    write_plate_results(out_dir / PLATES_FILE, pipeline.results())
    write_json(out_dir / TIMING_FILE, pipeline.timer.summary(pipeline.frames_processed))
    write_json(
        out_dir / RUN_FILE,
        {
            "video": {**video, "frames_processed": pipeline.frames_processed},
            "config": pipeline.config.to_dict(),
            "counters": dict(pipeline.counters),
            "environment": environment_info(resolve_device(pipeline.config.device)),
        },
    )
    logger.info("saved run to %s", out_dir)
    return out_dir


@dataclass(frozen=True)
class Run:
    """The contents of a run directory."""

    directory: Path
    observations: list[TrackObservation]
    results: list[PlateResult]
    meta: dict[str, Any]

    @property
    def video_path(self) -> Path:
        """The video the run was computed from."""
        return Path(self.meta["video"]["path"])


def load_run(run_dir: str | Path) -> Run:
    """Read a run directory written by :func:`save_run`.

    Raises:
        FileNotFoundError: If the directory is not a run directory.
    """
    run_dir = Path(run_dir)
    if not (run_dir / TRACKS_FILE).exists():
        raise FileNotFoundError(f"{run_dir} is not a run directory (no {TRACKS_FILE})")
    return Run(
        directory=run_dir,
        observations=read_observations(run_dir / TRACKS_FILE),
        results=read_plate_results(run_dir / PLATES_FILE),
        meta=json.loads((run_dir / RUN_FILE).read_text()),
    )
