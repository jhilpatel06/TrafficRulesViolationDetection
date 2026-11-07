# backend/tests/wrong_way/test_violation.py
from backend.pipeline.wrong_way.contracts import Lane, Point, VehicleTrack
from backend.pipeline.wrong_way.violation import WrongWayVerifier


def create_lane():
    return Lane(
        lane_id="lane_1",
        polygon=[],
        direction_start=Point(0, 0),
        direction_end=Point(100, 0),
    )


def create_track():
    return VehicleTrack(
        track_id=1,
        vehicle_class="car",
        bbox=(10, 10, 50, 50),
        trajectory=[
            Point(100, 10),
            Point(90, 10),
            Point(80, 10),
        ],
        smoothed_trajectory=[
            Point(100, 10),
            Point(90, 10),
            Point(80, 10),
        ],
    )


def test_violation_requires_persistence():
    verifier = WrongWayVerifier(persistence_frames=3)

    track = create_track()
    lane = create_lane()

    assert verifier.update(track, lane, 1) is None
    assert verifier.update(track, lane, 2) is None

    event = verifier.update(track, lane, 3)

    assert event is not None
    assert event.track_id == 1
    assert event.lane_id == "lane_1"


def test_only_one_event_per_track():
    verifier = WrongWayVerifier(persistence_frames=1)

    track = create_track()
    lane = create_lane()

    event = verifier.update(track, lane, 1)

    assert event is not None
    assert verifier.update(track, lane, 2) is None


def test_non_wrong_way_movement_resets_counter():
    verifier = WrongWayVerifier(persistence_frames=3)

    track = create_track()
    lane = create_lane()

    verifier.update(track, lane, 1)

    track.trajectory = [
        Point(10, 10),
        Point(20, 10),
        Point(30, 10),
    ]
    track.smoothed_trajectory = track.trajectory

    assert verifier.update(track, lane, 2) is None
    assert verifier.violation_counts[1] == 0


def test_remove_track():
    verifier = WrongWayVerifier(persistence_frames=1)

    track = create_track()
    lane = create_lane()

    verifier.update(track, lane, 1)
    verifier.remove(track.track_id)

    assert track.track_id not in verifier.violation_counts
    assert track.track_id not in verifier.confirmed_tracks