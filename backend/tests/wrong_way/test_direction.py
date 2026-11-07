# backend/tests/wrong_way/test_direction.py

from backend.pipeline.wrong_way.contracts import Point, VehicleTrack
from backend.pipeline.wrong_way.direction import DirectionEstimator


def create_track(points):
    return VehicleTrack(
        track_id=1,
        vehicle_class="car",
        bbox=(10, 10, 50, 50),
        trajectory=points,
        smoothed_trajectory=points,
    )


def test_direction_estimation():
    estimator = DirectionEstimator(min_displacement=5)

    track = create_track([
        Point(10, 10),
        Point(20, 10),
        Point(30, 10),
    ])

    direction = estimator.estimate_direction(track)

    assert direction.x == 1.0
    assert direction.y == 0.0


def test_opposite_direction_detected():
    estimator = DirectionEstimator()

    track = create_track([
        Point(30, 10),
        Point(20, 10),
        Point(10, 10),
    ])

    expected_direction = Point(1, 0)

    assert estimator.is_wrong_way(track, expected_direction) is True


def test_same_direction_not_detected():
    estimator = DirectionEstimator()

    track = create_track([
        Point(10, 10),
        Point(20, 10),
        Point(30, 10),
    ])

    expected_direction = Point(1, 0)

    assert estimator.is_wrong_way(track, expected_direction) is False


def test_insufficient_displacement():
    estimator = DirectionEstimator(min_displacement=10)

    track = create_track([
        Point(10, 10),
        Point(12, 10),
    ])

    direction = estimator.estimate_direction(track)

    assert direction is None


def test_zero_lane_direction():
    estimator = DirectionEstimator()

    track = create_track([
        Point(10, 10),
        Point(20, 10),
    ])

    assert estimator.is_wrong_way(track, Point(0, 0)) is False