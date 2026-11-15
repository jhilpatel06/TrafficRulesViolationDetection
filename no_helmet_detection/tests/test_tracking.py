from src.no_helmet_detection.geometry import Detection
from src.no_helmet_detection.tracking import ObjectTracker


def test_tracker_keeps_id_for_a_moving_motorcycle() -> None:
    tracker = ObjectTracker(max_gap=2)
    first = Detection((10, 10, 60, 55), "motorcycle", 0.9)
    next_frame = Detection((15, 12, 65, 57), "motorcycle", 0.88)

    first_track = tracker.update([first], frame_number=1)[0]
    next_track = tracker.update([next_frame], frame_number=2)[0]

    assert first_track.track_id == next_track.track_id


def test_tracker_assigns_a_new_id_after_a_long_disappearance() -> None:
    tracker = ObjectTracker(max_gap=1)
    first = Detection((10, 10, 60, 55), "motorcycle", 0.9)
    later = Detection((10, 10, 60, 55), "motorcycle", 0.9)

    first_track = tracker.update([first], frame_number=1)[0]
    later_track = tracker.update([later], frame_number=5)[0]

    assert later_track.track_id != first_track.track_id
