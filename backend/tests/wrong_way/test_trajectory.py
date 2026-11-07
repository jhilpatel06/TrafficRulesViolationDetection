# backend/tests/wrong_way/test_trajectory.py
from backend.pipeline.wrong_way.contracts import Point, VehicleTrack
from backend.pipeline.wrong_way.trajectory import TrajectoryManager


def create_track(track_id: int = 1) -> VehicleTrack:
    return VehicleTrack(
        track_id=track_id,
        vehicle_class="car",
        bbox=(10, 10, 50, 50),
    )


def test_trajectory_updates():
    manager = TrajectoryManager()
    track = create_track()

    manager.update(track, Point(20, 30))
    manager.update(track, Point(25, 35))

    assert len(track.trajectory) == 2
    assert track.trajectory[-1] == Point(25, 35)


def test_trajectory_respects_max_history():
    manager = TrajectoryManager(max_history=3)
    track = create_track()

    for i in range(5):
        manager.update(track, Point(i * 10, i * 10))

    assert len(track.trajectory) == 3
    assert track.trajectory[0] == Point(20, 20)


def test_ema_smoothing():
    manager = TrajectoryManager(ema_alpha=0.5)
    track = create_track()

    manager.update(track, Point(10, 10))
    manager.update(track, Point(20, 20))

    smoothed = track.smoothed_trajectory

    assert len(smoothed) == 2
    assert smoothed[0] == Point(10, 10)
    assert smoothed[1] == Point(15, 15)


def test_remove_track():
    manager = TrajectoryManager()
    track = create_track()

    manager.update(track, Point(10, 10))
    manager.remove(track.track_id)

    assert manager.get(track.track_id) == []
    assert manager.get_smoothed(track.track_id) == []