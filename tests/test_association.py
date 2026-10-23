from alpr.association import assign_plates_to_tracks
from alpr.types import Box, Detection, Track


def plate(x1, y1, x2, y2, score=0.9):
    return Detection(Box(x1, y1, x2, y2), score)


CAR = Track(1, Box(100, 100, 300, 250))


def test_plate_inside_vehicle_is_assigned():
    inside = plate(180, 210, 230, 230)
    assert assign_plates_to_tracks([inside], [CAR]) == {1: inside}


def test_plate_poking_slightly_outside_still_matches():
    """Regression: the original demanded strict containment on all four sides.

    One pixel of box jitter at the bumper was enough to lose the plate.
    """
    jittered = plate(180, 235, 230, 252)  # 2 px below the vehicle box
    assert assign_plates_to_tracks([jittered], [CAR]) == {1: jittered}


def test_plate_matching_no_vehicle_is_dropped():
    """Regression: unmatched plates used to be pooled under a fake car ID of -1."""
    stray = plate(600, 600, 650, 620)
    assert assign_plates_to_tracks([stray], [CAR]) == {}


def test_plate_mostly_outside_is_not_matched():
    half_out = plate(280, 200, 330, 220)  # 40% inside
    assert assign_plates_to_tracks([half_out], [CAR]) == {}
    assert assign_plates_to_tracks([half_out], [CAR], min_overlap=0.3) == {1: half_out}


def test_smallest_box_wins_when_vehicles_overlap():
    """Regression: the original took whichever containing vehicle came first."""
    truck = Track(7, Box(50, 20, 400, 300))
    car = Track(8, Box(150, 150, 300, 280))
    on_car = plate(200, 250, 250, 270)  # fully inside both boxes
    assert assign_plates_to_tracks([on_car], [truck, car]) == {8: on_car}
    assert assign_plates_to_tracks([on_car], [car, truck]) == {8: on_car}


def test_clearly_better_overlap_beats_a_smaller_box():
    big = Track(1, Box(0, 0, 400, 300))
    small = Track(2, Box(215, 240, 300, 300))  # only covers 70% of the plate
    p = plate(200, 250, 250, 270)
    assert assign_plates_to_tracks([p], [big, small]) == {1: p}


def test_each_vehicle_keeps_only_its_best_plate():
    weak = plate(120, 120, 170, 140, score=0.4)
    strong = plate(180, 210, 230, 230, score=0.9)
    assert assign_plates_to_tracks([weak, strong], [CAR]) == {1: strong}
    assert assign_plates_to_tracks([strong, weak], [CAR]) == {1: strong}


def test_two_vehicles_two_plates():
    left, right = Track(1, Box(0, 0, 200, 200)), Track(2, Box(300, 0, 500, 200))
    plate_left, plate_right = plate(80, 160, 120, 180), plate(380, 160, 420, 180)
    assert assign_plates_to_tracks([plate_right, plate_left], [left, right]) == {
        1: plate_left,
        2: plate_right,
    }


def test_empty_inputs():
    assert assign_plates_to_tracks([], [CAR]) == {}
    assert assign_plates_to_tracks([plate(0, 0, 10, 10)], []) == {}
    assert assign_plates_to_tracks([], []) == {}
