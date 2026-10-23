import numpy as np

import synthetic
from alpr.config import load_config
from alpr.io import VideoReader, VideoWriter
from alpr.pipeline import PlateReaderPipeline
from alpr.types import Box, PlateResult, TrackObservation
from alpr.visualize import (
    FrameAnnotator,
    draw_corner_box,
    draw_label,
    extract_crops,
    paste_above,
    render_video,
    select_best_views,
)


def blank(height=360, width=640):
    return np.zeros((height, width, 3), np.uint8)


def test_corner_box_draws_corners_only():
    image = blank()
    draw_corner_box(image, Box(100, 100, 300, 200), (0, 171, 255), thickness=2)
    assert image[100, 100].any() and image[200, 300].any()
    assert not image[100, 200].any(), "the middle of an edge stays clear"
    assert not image[150, 200].any(), "the interior stays clear"


def test_corner_box_partly_outside_the_frame_does_not_fail():
    image = blank()
    draw_corner_box(image, Box(-50, -20, 80, 60), (255, 255, 255), thickness=2)
    draw_corner_box(image, Box(600, 300, 900, 500), (255, 255, 255), thickness=2)
    assert image.any()


def test_patch_is_kept_inside_the_frame_at_the_top_edge():
    """Regression: the original wrapped this in ``try/except: pass``.

    For a vehicle near the top of the frame the slice went negative, the
    assignment failed, and the overlay silently disappeared.
    """
    image = blank()
    patch = np.full((40, 100, 3), 200, np.uint8)
    top = paste_above(image, patch, center_x=320, bottom=10)  # would start at y = -30
    assert top == 0
    assert (image[0:40, 270:370] == 200).all()


def test_patch_is_kept_inside_the_frame_at_the_sides():
    image = blank()
    patch = np.full((20, 100, 3), 200, np.uint8)
    paste_above(image, patch, center_x=5, bottom=100)
    assert (image[80:100, 0:100] == 200).all()
    paste_above(image, patch, center_x=636, bottom=200)
    assert (image[180:200, 540:640] == 200).all()


def test_oversized_patch_is_skipped():
    image = blank(50, 50)
    assert paste_above(image, np.ones((80, 80, 3), np.uint8), center_x=25, bottom=40) == 40
    assert not image.any()


def test_label_is_drawn_inside_the_frame():
    image = blank()
    top = draw_label(image, "BXYZ123", center_x=320, bottom=5, scale=0.8, thickness=2)
    assert top == 0
    assert (image == 255).any(), "white banner"
    ys, _ = np.nonzero(image.any(axis=2))
    assert ys.min() >= 0 and ys.max() < 40


def test_annotator_labels_only_tracks_with_a_plate():
    plates = {1: PlateResult(1, "BXYZ123", 0.9, 10, 0, 20)}
    annotator = FrameAnnotator(360, plates, insets={1: np.full((14, 36, 3), 180, np.uint8)})
    rows = [
        TrackObservation(0, 1, Box(100, 150, 220, 220), plate_box=Box(142, 200, 178, 214)),
        TrackObservation(0, 2, Box(400, 150, 520, 220)),
    ]
    frame = annotator.draw(blank(), rows)
    assert (frame[60:150, 100:220] == 255).any(), "banner above the vehicle with a plate"
    assert not frame[60:140, 400:520].any(), "nothing above the vehicle without one"
    assert frame[150, 400].any(), "but its box is still drawn"


def test_annotator_handles_a_vehicle_at_the_top_edge():
    plates = {1: PlateResult(1, "BXYZ123", 0.9, 10, 0, 20)}
    annotator = FrameAnnotator(360, plates, insets={1: np.full((14, 36, 3), 180, np.uint8)})
    frame = annotator.draw(blank(), [TrackObservation(0, 1, Box(100, 4, 220, 74))])
    assert (frame[:60] == 255).any(), "the label is still visible"


def test_best_view_prefers_the_largest_plate():
    rows = [
        TrackObservation(0, 1, Box(0, 0, 100, 60), plate_box=Box(40, 40, 60, 48)),
        TrackObservation(1, 1, Box(0, 0, 120, 70), plate_box=Box(40, 40, 76, 54)),
        TrackObservation(2, 1, Box(0, 0, 200, 150)),  # bigger vehicle, no plate
        TrackObservation(3, 1, Box(0, 0, 300, 200), plate_box=Box(0, 0, 90, 40), interpolated=True),
        TrackObservation(0, 2, Box(0, 0, 50, 30)),
        TrackObservation(1, 2, Box(0, 0, 90, 60)),
    ]
    views = select_best_views(rows)
    assert views[1].frame == 1, "largest detected plate; interpolated rows do not count"
    assert views[2].frame == 1, "no plate at all: largest vehicle box"


def write_scene(path):
    with VideoWriter(path, fps=25, size=(synthetic.WIDTH, synthetic.HEIGHT)) as writer:
        for _, frame in synthetic.frames():
            writer.write(frame)
    return path


def run_on_file(path):
    config = load_config(overrides=["tracker.max_age=5"])
    pipeline = PlateReaderPipeline(*synthetic.stand_ins(), config)
    with VideoReader(path) as video:
        pipeline.run(video)
    return pipeline


def test_crops_are_cut_from_the_chosen_frames(tmp_path):
    path = write_scene(tmp_path / "scene.mp4")
    pipeline = run_on_file(path)
    crops = extract_crops(path, select_best_views(pipeline.observations))
    assert len(crops) == len(synthetic.VEHICLES)
    for vehicle_crop, plate_crop in crops.values():
        assert vehicle_crop.shape[0] > 50 and vehicle_crop.shape[1] > 90
        assert plate_crop is not None and plate_crop[..., 2].mean() > 150, "plates are red"


def test_render_writes_an_annotated_video_of_the_same_length(tmp_path):
    source = write_scene(tmp_path / "scene.mp4")
    pipeline = run_on_file(source)
    assert {r.text for r in pipeline.results()} == {v.plate for v in synthetic.VEHICLES}

    output = render_video(
        source,
        pipeline.observations,
        pipeline.results(),
        tmp_path / "annotated.mp4",
        progress=False,
    )
    with VideoReader(source) as before, VideoReader(output) as after:
        assert (after.width, after.height, after.fps) == (before.width, before.height, before.fps)
        original = dict(before)
        annotated = dict(after)
    assert len(annotated) == len(original) == synthetic.N_FRAMES
    assert np.abs(annotated[20].astype(int) - original[20].astype(int)).max() > 100
    assert (annotated[20] > 200).all(axis=2).any(), "white label banner present"
    assert np.abs(annotated[44].astype(int) - original[44].astype(int)).mean() < 2, (
        "frames with no vehicles are passed through"
    )
