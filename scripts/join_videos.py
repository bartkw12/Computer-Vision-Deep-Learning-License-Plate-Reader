r"""Join video clips into one evaluation video, separated by black frames.

Usage::

    python scripts/join_videos.py data/videos/ECE736_final1.mp4 \
        data/videos/ECE736_final2.mp4 --out data/videos/toronto_eval.mp4

One video is easier to label, score and ablate than several. The black gap
between clips is longer than any tracker lifetime the ablations try, so no
track survives from one clip into the next and joins two different vehicles.
The frame ranges of the clips are written next to the output as JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from alpr.io import VideoReader, VideoWriter

logger = logging.getLogger("join")


def join_videos(clips: list[Path], out: Path, gap_seconds: float, crf: int) -> list[dict]:
    """Write ``clips`` one after another into ``out``.

    Raises:
        ValueError: If the clips differ in frame size or frame rate.

    Returns:
        One ``{"clip", "first_frame", "last_frame"}`` entry per clip.
    """
    with VideoReader(clips[0]) as first:
        fps, size = first.fps, (first.width, first.height)
    black = np.zeros((size[1], size[0], 3), np.uint8)
    segments = []
    frame = 0
    with VideoWriter(out, fps=fps, size=size, crf=crf) as writer:
        for number, clip in enumerate(clips):
            with VideoReader(clip) as video:
                if (video.width, video.height) != size or abs(video.fps - fps) > 0.01:
                    raise ValueError(
                        f"{clip} is {video.width}x{video.height} at {video.fps:.2f} fps; "
                        f"{clips[0]} is {size[0]}x{size[1]} at {fps:.2f} fps"
                    )
                if number:
                    for _ in range(round(gap_seconds * fps)):
                        writer.write(black)
                        frame += 1
                start = frame
                for _, image in video:
                    writer.write(image)
                    frame += 1
            segments.append({"clip": clip.name, "first_frame": start, "last_frame": frame - 1})
    return segments


def main() -> None:
    """Join the clips and record where each one lies."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gap", type=float, default=2.0, help="seconds of black between clips")
    parser.add_argument("--crf", type=int, default=16, help="H.264 quality; lower is better")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    segments = join_videos(args.clips, args.out, args.gap, args.crf)
    index = args.out.with_suffix(".clips.json")
    index.write_text(json.dumps(segments, indent=2) + "\n")
    for segment in segments:
        logger.info(
            "%s: frames %d-%d", segment["clip"], segment["first_frame"], segment["last_frame"]
        )
    logger.info("-> %s, %s", args.out, index)


if __name__ == "__main__":
    main()
