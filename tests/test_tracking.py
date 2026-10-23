import numpy as np
import pytest

from alpr.tracking import Sort
from alpr.tracking.kalman import KalmanBoxTracker, box_to_measurement, measurement_to_box
from alpr.tracking.sort import associate


def moving_box(frame, *, x0=100.0, y0=200.0, vx=6.0, vy=0.0, w=80.0, h=50.0, score=0.9):
    """Detection row for an object that started at (x0, y0) and moves at constant velocity."""
    x, y = x0 + vx * frame, y0 + vy * frame
    return [x, y, x + w, y + h, score]


def ids(tracks):
    return sorted(int(i) for i in tracks[:, 4])


def test_measurement_round_trip():
    box = np.array([10.0, 20.0, 110.0, 60.0])
    np.testing.assert_allclose(measurement_to_box(box_to_measurement(box)), box)


def test_filter_learns_velocity_and_predicts_ahead():
    tracker = KalmanBoxTracker(np.array(moving_box(0)[:4]), track_id=1)
    for frame in range(1, 15):
        tracker.predict()
        tracker.update(np.array(moving_box(frame)[:4]))
    predicted = tracker.predict()
    np.testing.assert_allclose(predicted, moving_box(15)[:4], atol=1.0)


def test_single_object_keeps_one_id():
    tracker = Sort(max_age=1, min_hits=3)
    seen = []
    for frame in range(40):
        tracks = tracker.update(np.array([moving_box(frame)]))
        assert tracks.shape == (1, 5)
        seen.append(int(tracks[0, 4]))
        np.testing.assert_allclose(tracks[0, :4], moving_box(frame)[:4], atol=3.0)
    assert set(seen) == {1}


@pytest.mark.parametrize("empty", [[], np.asarray([]), np.empty((0, 5)), np.empty((0,))])
def test_frames_without_detections_do_not_crash(empty):
    """Regression: the original pipeline passed ``np.asarray([])`` (shape ``(0,)``).

    With a live track that raised IndexError inside the IoU computation.
    """
    tracker = Sort(max_age=3)
    for frame in range(5):
        tracker.update(np.array([moving_box(frame)]))
    for _ in range(6):
        tracks = tracker.update(empty)
        assert tracks.shape == (0, 5)


def test_tracker_starts_empty():
    assert Sort().update([]).shape == (0, 5)


def test_wrong_number_of_columns_is_rejected():
    with pytest.raises(ValueError, match=r"\(N, 5\)"):
        Sort().update(np.array([[0.0, 0.0, 10.0, 10.0]]))


def run_with_dropout(max_age, missing):
    tracker = Sort(max_age=max_age, min_hits=3)
    before = after = None
    for frame in range(30):
        detections = [] if frame in missing else [moving_box(frame)]
        tracks = tracker.update(np.array(detections))
        if frame == min(missing) - 1:
            before = ids(tracks)
        if frame == 29:
            after = ids(tracks)
    return before, after


def test_track_survives_a_dropout_within_max_age():
    before, after = run_with_dropout(max_age=5, missing={10, 11, 12})
    assert before == after == [1]


def test_track_is_replaced_after_a_dropout_longer_than_max_age():
    before, after = run_with_dropout(max_age=1, missing={10, 11, 12})
    assert before == [1]
    assert after == [2]


def test_rematched_track_is_reported_immediately():
    """A confirmed track must not vanish for ``min_hits`` frames after one miss."""
    tracker = Sort(max_age=5, min_hits=3)
    for frame in range(10):
        tracker.update(np.array([moving_box(frame)]))
    tracker.update([])
    assert ids(tracker.update(np.array([moving_box(11)]))) == [1]


def test_kalman_prediction_bridges_a_gap_that_overlap_alone_cannot():
    # 30 px/frame with an 80 px wide box: after 3 missed frames the object has
    # moved 120 px, so it no longer overlaps where it was last seen.
    fast = {"vx": 30.0}
    tracker = Sort(max_age=5, min_hits=3)
    for frame in range(12):
        tracker.update(np.array([moving_box(frame, **fast)]))
    for _ in range(3):
        tracker.update([])
    assert ids(tracker.update(np.array([moving_box(15, **fast)]))) == [1]


def test_two_objects_keep_distinct_ids():
    tracker = Sort(max_age=2, min_hits=3)
    for frame in range(30):
        detections = np.array(
            [moving_box(frame, y0=100.0), moving_box(frame, x0=900.0, y0=400.0, vx=-6.0)]
        )
        tracks = tracker.update(detections)
        assert ids(tracks) == [1, 2]
        by_id = {int(t[4]): t for t in tracks}
        # ID 1 stays with the object in the upper lane, whichever way it is listed.
        assert by_id[1][1] < 200 < by_id[2][1]


def test_detection_order_does_not_change_ids():
    tracker = Sort(min_hits=1)
    first, second = moving_box(0, y0=100.0), moving_box(0, y0=400.0)
    tracker.update(np.array([first, second]))
    for frame in range(1, 10):
        a, b = moving_box(frame, y0=100.0), moving_box(frame, y0=400.0)
        order = [a, b] if frame % 2 else [b, a]
        by_id = {int(t[4]): t for t in tracker.update(np.array(order))}
        assert by_id[1][1] < 200 < by_id[2][1]


def test_late_track_needs_min_hits_before_it_is_reported():
    tracker = Sort(max_age=1, min_hits=3)
    for frame in range(10):
        tracker.update(np.array([moving_box(frame)]))
    reported = []
    for frame in range(10, 16):
        late = moving_box(frame - 10, x0=600.0, y0=600.0)
        tracks = tracker.update(np.array([moving_box(frame), late]))
        reported.append(len(tracks))
    # Born at frame 10, then three consecutive matches (11, 12, 13) confirm it.
    assert reported == [1, 1, 1, 2, 2, 2]


def test_tracks_born_in_the_first_frames_are_reported_at_once():
    tracker = Sort(min_hits=3)
    assert ids(tracker.update(np.array([moving_box(0)]))) == [1]


def test_ids_restart_for_each_tracker_instance():
    for _ in range(2):
        tracker = Sort()
        assert ids(tracker.update(np.array([moving_box(0)]))) == [1]


def test_associate_respects_the_iou_threshold():
    detections = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]])
    predictions = np.array([[1.0, 1.0, 11.0, 11.0], [108.0, 108.0, 118.0, 118.0]])
    matches, unmatched = associate(detections, predictions, iou_threshold=0.3)
    assert matches == [(0, 0)]
    assert unmatched == [1]


def test_associate_solves_the_assignment_globally():
    # Greedy matching would give detection 0 its best track (0) and leave
    # detection 1 with nothing; the optimal assignment matches both.
    detections = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 2.0, 10.0, 12.0]])
    predictions = np.array([[0.0, 1.0, 10.0, 11.0], [0.0, -6.0, 10.0, 4.0]])
    matches, unmatched = associate(detections, predictions, iou_threshold=0.2)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert unmatched == []
