import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from alpr.io import VideoReader, VideoWriter

SCRIPT = Path(__file__).parents[1] / "scripts" / "join_videos.py"
_spec = importlib.util.spec_from_file_location("join_videos", SCRIPT)
join = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = join
_spec.loader.exec_module(join)


def clip(path, frames, value, size=(64, 48), fps=10):
    with VideoWriter(path, fps=fps, size=size) as writer:
        for _ in range(frames):
            writer.write(np.full((size[1], size[0], 3), value, np.uint8))
    return path


def test_clips_are_joined_with_a_black_gap(tmp_path):
    clips = [clip(tmp_path / "a.mp4", 5, 200), clip(tmp_path / "b.mp4", 3, 120)]
    out = tmp_path / "joined.mp4"
    segments = join.join_videos(clips, out, gap_seconds=0.4, crf=10)
    assert segments == [
        {"clip": "a.mp4", "first_frame": 0, "last_frame": 4},
        {"clip": "b.mp4", "first_frame": 9, "last_frame": 11},
    ]
    with VideoReader(out) as video:
        brightness = [round(float(frame.mean())) for _, frame in video]
    assert len(brightness) == 12
    assert all(abs(b - 200) < 8 for b in brightness[:5])
    assert all(b < 8 for b in brightness[5:9])
    assert all(abs(b - 120) < 8 for b in brightness[9:])


def test_mismatched_clips_are_refused(tmp_path):
    clips = [clip(tmp_path / "a.mp4", 2, 0), clip(tmp_path / "b.mp4", 2, 0, size=(32, 24))]
    with pytest.raises(ValueError, match="32x24"):
        join.join_videos(clips, tmp_path / "joined.mp4", gap_seconds=1, crf=10)
