import numpy as np
import pytest

from alpr.geometry import boxes_to_array, crop, intersection_matrix, ioa_matrix, iou_matrix
from alpr.types import Box


def test_box_properties():
    box = Box(10, 20, 50, 40)
    assert (box.width, box.height, box.area) == (40, 20, 800)
    assert box.center == (30, 30)
    assert box.contains_point(10, 20) and not box.contains_point(9, 20)
    assert box.shifted(5, -5) == Box(15, 15, 55, 35)


def test_degenerate_box_has_zero_area():
    assert Box(10, 10, 5, 5).area == 0


def test_box_clipped_to_image():
    assert Box(-10, -5, 700, 500).clipped(640, 480) == Box(0, 0, 640, 480)


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 25 / 175, 0.0]])


def test_iou_is_symmetric():
    rng = np.random.default_rng(0)
    corners = rng.uniform(0, 100, size=(6, 2))
    boxes = np.hstack([corners, corners + rng.uniform(1, 50, size=(6, 2))])
    np.testing.assert_allclose(iou_matrix(boxes, boxes), iou_matrix(boxes, boxes).T)
    np.testing.assert_allclose(np.diag(iou_matrix(boxes, boxes)), 1.0)


def test_ioa_measures_containment_not_overlap_ratio():
    plate = np.array([[40, 80, 60, 90]])
    car = np.array([[0, 0, 100, 100]])
    assert ioa_matrix(plate, car)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(plate, car)[0, 0] == pytest.approx(0.02)
    # Asymmetric by design: only 2% of the car lies inside the plate.
    assert ioa_matrix(car, plate)[0, 0] == pytest.approx(0.02)


def test_ioa_partial_containment():
    plate = np.array([[90, 0, 110, 10]])  # half hangs off the car's right edge
    car = np.array([[0, 0, 100, 100]])
    assert ioa_matrix(plate, car)[0, 0] == pytest.approx(0.5)


@pytest.mark.parametrize("func", [iou_matrix, ioa_matrix, intersection_matrix])
def test_empty_inputs_give_correctly_shaped_empty_results(func):
    some = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
    none = np.empty((0, 4))
    assert func(none, some).shape == (0, 2)
    assert func(some, none).shape == (2, 0)
    assert func(none, none).shape == (0, 0)


def test_zero_area_boxes_do_not_divide_by_zero():
    point = np.array([[5, 5, 5, 5]])
    box = np.array([[0, 0, 10, 10]])
    assert iou_matrix(point, box)[0, 0] == 0
    assert ioa_matrix(point, box)[0, 0] == 0
    assert iou_matrix(point, point)[0, 0] == 0


def test_boxes_to_array():
    assert boxes_to_array([]).shape == (0, 4)
    np.testing.assert_array_equal(
        boxes_to_array([Box(1, 2, 3, 4), Box(5, 6, 7, 8)]), [[1, 2, 3, 4], [5, 6, 7, 8]]
    )


def test_crop_clips_to_image_and_can_be_empty():
    image = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    assert crop(image, Box(10, 20, 60, 50)).shape == (30, 50, 3)
    assert crop(image, Box(-20, -20, 30, 30)).shape == (30, 30, 3)
    assert crop(image, Box(150, 50, 400, 400)).shape == (50, 50, 3)
    assert crop(image, Box(300, 300, 400, 400)).size == 0
    assert crop(image, Box(10.2, 10.2, 10.4, 10.4)).size == 0
