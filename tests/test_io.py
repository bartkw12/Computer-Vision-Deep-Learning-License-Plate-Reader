import csv

import numpy as np
import pytest

import synthetic
from alpr.io import (
    OBSERVATION_COLUMNS,
    VideoReader,
    VideoWriter,
    read_observations,
    read_plate_results,
    write_observations,
    write_plate_results,
)
from alpr.types import Box, PlateResult, TrackObservation

OBSERVATIONS = [
    TrackObservation(
        frame=0,
        track_id=1,
        vehicle_box=Box(10.25, 20.5, 110.75, 80.0),
        plate_box=Box(50.0, 60.0, 80.0, 72.0),
        plate_score=0.9123,
        ocr_text="BXYZ1Z3",
        ocr_confidence=0.6543,
        plate_text="BXYZ123",
    ),
    TrackObservation(frame=0, track_id=2, vehicle_box=Box(300, 20, 400, 80)),
    TrackObservation(
        frame=1,
        track_id=1,
        vehicle_box=Box(14, 20, 114, 80),
        plate_box=Box(54, 60, 84, 72),
        interpolated=True,
    ),
]


def test_observations_round_trip(tmp_path):
    path = tmp_path / "nested" / "tracks.csv"
    write_observations(path, OBSERVATIONS)
    assert read_observations(path) == OBSERVATIONS


def test_observation_csv_is_flat_and_numeric(tmp_path):
    """Regression: boxes used to be written as ``"[x y x y]"`` strings.

    Reading them back took a chain of ``str.replace`` calls and ``ast.literal_eval``.
    """
    path = tmp_path / "tracks.csv"
    write_observations(path, OBSERVATIONS)
    with path.open(newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == OBSERVATION_COLUMNS
    first = dict(zip(rows[0], rows[1], strict=True))
    assert [float(first[f"vehicle_{c}"]) for c in ("x1", "y1", "x2", "y2")] == [
        10.25, 20.5, 110.75, 80.0,
    ]  # fmt: skip
    assert not any("[" in cell for row in rows for cell in row)
    # A track without a plate leaves the plate columns empty rather than writing zeros.
    second = dict(zip(rows[0], rows[2], strict=True))
    assert second["plate_x1"] == second["plate_score"] == second["plate_text"] == ""


def test_plate_results_round_trip(tmp_path):
    results = [
        PlateResult(1, "BXYZ123", 0.8125, 31, 3, 40),
        PlateResult(7, "CKLM456", 1.0, 2, 8, 9),
    ]
    path = tmp_path / "plates.csv"
    write_plate_results(path, results)
    assert read_plate_results(path) == results


def test_empty_tables_round_trip(tmp_path):
    write_observations(tmp_path / "t.csv", [])
    write_plate_results(tmp_path / "p.csv", [])
    assert read_observations(tmp_path / "t.csv") == []
    assert read_plate_results(tmp_path / "p.csv") == []


def write_scene(path, n=20):
    with VideoWriter(path, fps=25, size=(synthetic.WIDTH, synthetic.HEIGHT)) as writer:
        for _, frame in synthetic.frames(n):
            writer.write(frame)
    return path


def test_video_round_trip(tmp_path):
    path = write_scene(tmp_path / "out" / "scene.mp4")
    with VideoReader(path) as video:
        assert (video.width, video.height) == (synthetic.WIDTH, synthetic.HEIGHT)
        assert video.fps == pytest.approx(25)
        frames = list(video)
    assert [index for index, _ in frames] == list(range(20))
    # Lossy codec: content is close to what was written, not identical.
    original = synthetic.render(10).astype(int)
    assert np.abs(frames[10][1].astype(int) - original).mean() < 6


def test_written_video_is_h264(tmp_path):
    """Regression: OpenCV's ``mp4v`` output does not play in browsers or on GitHub."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(write_scene(tmp_path / "scene.mp4")))
    meta = next(reader)
    reader.close()
    assert meta["codec"].startswith("h264")
    assert meta["pix_fmt"].startswith("yuv420p")


def test_video_can_be_iterated_only_once_per_reader(tmp_path):
    path = write_scene(tmp_path / "scene.mp4", n=5)
    with VideoReader(path) as video:
        assert len(list(video)) == 5
        assert list(video) == []


def test_missing_and_unreadable_videos(tmp_path):
    with pytest.raises(FileNotFoundError, match="video not found"):
        VideoReader(tmp_path / "nope.mp4")
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"this is not a video")
    with pytest.raises(OSError, match="could not open video"):
        VideoReader(garbage)


def test_writer_rejects_frames_of_the_wrong_size(tmp_path):
    with VideoWriter(tmp_path / "v.mp4", fps=10, size=(64, 48)) as writer:
        writer.write(np.zeros((48, 64, 3), np.uint8))
        with pytest.raises(ValueError, match="writer expects"):
            writer.write(np.zeros((64, 48, 3), np.uint8))
