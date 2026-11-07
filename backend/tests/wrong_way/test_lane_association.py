# backend/tests/wrong_way/test_lane_association.py

from backend.pipeline.wrong_way.contracts import Lane, Point, VehicleTrack
from backend.pipeline.wrong_way.lane_association import LaneAssociation


def create_lane(lane_id, x1, y1, x2, y2):
    return Lane(
        lane_id=lane_id,
        polygon=[
            Point(x1, y1),
            Point(x2, y1),
            Point(x2, y2),
            Point(x1, y2),
        ],
        direction_start=Point(x1, y2),
        direction_end=Point(x2, y1),
    )


def create_track(track_id, bbox):
    return VehicleTrack(
        track_id=track_id,
        vehicle_class="car",
        bbox=bbox,
    )


def test_vehicle_assigned_to_correct_lane():
    association = LaneAssociation()

    lane = create_lane("lane_1", 0, 0, 100, 100)
    track = create_track(1, (40, 20, 60, 80))

    result = association.find_lane(track, [lane])

    assert result.lane_id == "lane_1"
    assert track.lane_id == "lane_1"


def test_vehicle_outside_lane():
    association = LaneAssociation()

    lane = create_lane("lane_1", 0, 0, 100, 100)
    track = create_track(1, (150, 20, 170, 80))

    result = association.find_lane(track, [lane])

    assert result is None


def test_lane_transition_requires_confirmation():
    association = LaneAssociation(transition_frames=3)

    lane_1 = create_lane("lane_1", 0, 0, 100, 100)
    lane_2 = create_lane("lane_2", 100, 0, 200, 100)

    track = create_track(1, (40, 20, 60, 80))

    association.find_lane(track, [lane_1, lane_2])

    track.bbox = (120, 20, 140, 80)

    assert association.find_lane(track, [lane_1, lane_2]) is None
    assert track.lane_id == "lane_1"

    assert association.find_lane(track, [lane_1, lane_2]) is None
    assert track.lane_id == "lane_1"

    result = association.find_lane(track, [lane_1, lane_2])

    assert result.lane_id == "lane_2"
    assert track.lane_id == "lane_2"


def test_remove_track():
    association = LaneAssociation()

    lane = create_lane("lane_1", 0, 0, 100, 100)
    track = create_track(1, (40, 20, 60, 80))

    association.find_lane(track, [lane])
    association.remove(1)

    assert 1 not in association.assignments