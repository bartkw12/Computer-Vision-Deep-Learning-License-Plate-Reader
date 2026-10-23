import numpy as np
import pytest

from alpr.postprocess import fill_gaps, interpolate_gaps
from alpr.types import Box, TrackObservation


def test_gap_is_filled_linearly():
    frames, boxes, filled = interpolate_gaps(
        np.array([0, 4]), np.array([[0, 0, 10, 10], [40, 80, 50, 90]]), max_gap=10
    )
    np.testing.assert_array_equal(frames, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(boxes[2], [20, 40, 30, 50])
    np.testing.assert_allclose(boxes[1], [10, 20, 20, 30])
    np.testing.assert_array_equal(filled, [False, True, True, True, False])


def test_known_boxes_are_returned_untouched():
    known = np.array([[1.5, 2.5, 3.5, 4.5], [9.0, 9.0, 12.0, 12.0]])
    _, boxes, _ = interpolate_gaps(np.array([3, 6]), known, max_gap=5)
    np.testing.assert_array_equal(boxes[[0, -1]], known)


def test_long_gap_is_left_empty():
    frames, _, filled = interpolate_gaps(
        np.array([0, 2, 50]), np.repeat([[0.0], [2.0], [50.0]], 4, axis=1), max_gap=5
    )
    np.testing.assert_array_equal(frames, [0, 1, 2, 50])
    np.testing.assert_array_equal(filled, [False, True, False, False])


def test_max_gap_boundary():
    boxes = np.zeros((2, 4))
    assert len(interpolate_gaps(np.array([0, 4]), boxes, max_gap=3)[0]) == 5
    assert len(interpolate_gaps(np.array([0, 4]), boxes, max_gap=2)[0]) == 2
    assert len(interpolate_gaps(np.array([0, 4]), boxes, max_gap=0)[0]) == 2


def test_unsorted_input_is_sorted():
    frames, boxes, _ = interpolate_gaps(
        np.array([4, 0]), np.array([[40, 0, 50, 10], [0, 0, 10, 10]]), max_gap=10
    )
    np.testing.assert_array_equal(frames, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(boxes[:, 0], [0, 10, 20, 30, 40])


def test_no_gaps_and_empty_input():
    frames, boxes, filled = interpolate_gaps(np.array([5, 6, 7]), np.ones((3, 4)), max_gap=3)
    np.testing.assert_array_equal(frames, [5, 6, 7])
    assert not filled.any()
    frames, boxes, filled = interpolate_gaps(np.array([]), np.empty((0, 4)), max_gap=3)
    assert frames.shape == (0,) and boxes.shape == (0, 4) and filled.shape == (0,)


def test_duplicate_frames_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        interpolate_gaps(np.array([1, 1]), np.zeros((2, 4)), max_gap=3)


def observation(frame, track_id, x, plate_x=None, text=None):
    return TrackObservation(
        frame=frame,
        track_id=track_id,
        vehicle_box=Box(x, 0, x + 100, 60),
        plate_box=None if plate_x is None else Box(plate_x, 40, plate_x + 30, 50),
        plate_score=None if plate_x is None else 0.9,
        plate_text=text,
    )


def test_fill_gaps_returns_every_frame_with_both_boxes():
    """Regression for the original ``fill_missing_data.py``.

    It appended its output row outside the loop, so each car came out as a
    single row, and it never wrote the plate box at all.
    """
    rows = [
        observation(10, 1, x=0, plate_x=35, text="BXYZ123"),
        observation(14, 1, x=40, plate_x=75, text="BXYZ123"),
    ]
    filled = fill_gaps(rows, max_gap=10)
    assert [r.frame for r in filled] == [10, 11, 12, 13, 14]
    assert all(r.plate_box is not None for r in filled)
    middle = filled[2]
    assert middle.vehicle_box == Box(20, 0, 120, 60)
    assert middle.plate_box == Box(55, 40, 85, 50)


def test_interpolated_rows_are_flagged_and_carry_no_reading():
    rows = [observation(0, 1, x=0, plate_x=35, text="BXYZ123"), observation(2, 1, x=20, plate_x=55)]
    original_first, interpolated, original_last = fill_gaps(rows, max_gap=5)
    assert original_first == rows[0] and original_last == rows[1]
    assert interpolated.interpolated
    assert interpolated.plate_text is None and interpolated.plate_score is None


def test_plate_gap_is_bridged_while_the_vehicle_stays_tracked():
    rows = [
        observation(0, 1, x=0, plate_x=30),
        observation(1, 1, x=10),  # vehicle tracked, plate detector missed
        observation(2, 1, x=20, plate_x=50),
    ]
    filled = fill_gaps(rows, max_gap=5)
    assert not filled[1].interpolated
    assert filled[1].plate_box == Box(40, 40, 70, 50)
    assert filled[1].plate_score is None


def test_tracks_are_filled_independently_and_sorted():
    rows = [
        observation(3, 2, x=500),
        observation(0, 1, x=0),
        observation(2, 1, x=20),
        observation(0, 2, x=530),
    ]
    filled = fill_gaps(rows, max_gap=5)
    assert [(r.frame, r.track_id) for r in filled] == [
        (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2),
    ]  # fmt: skip
    assert next(r for r in filled if (r.frame, r.track_id) == (1, 2)).vehicle_box.x1 == 520


def test_fill_gaps_respects_max_gap():
    rows = [observation(0, 1, x=0), observation(30, 1, x=300)]
    assert fill_gaps(rows, max_gap=10) == rows
