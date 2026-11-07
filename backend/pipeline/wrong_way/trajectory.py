# backend/pipeline/wrong_way/trajectory.py
#-----------------------------------------------------------------
# Maintains a separate trajectory buffer for each vehicle.
# Keeps at most 50 recent centroid positions.
# Automatically removes the oldest point when the buffer is full.
# Updates the corresponding VehicleTrack trajectory.

from collections import deque

from .contracts import Point, VehicleTrack


class TrajectoryManager:
    def __init__(
        self,
        max_history: int = 50,
        ema_alpha: float = 0.3,
    ):
        self.max_history = max_history
        self.ema_alpha = ema_alpha

        self.trajectories: dict[int, deque[Point]] = {}
        self.smoothed: dict[int, deque[Point]] = {}

    def update(self, track: VehicleTrack, centroid: Point) -> None:
        track_id = track.track_id

        if track_id not in self.trajectories:
            self.trajectories[track_id] = deque(maxlen=self.max_history)
            self.smoothed[track_id] = deque(maxlen=self.max_history)

        self.trajectories[track_id].append(centroid)
        previous = self.smoothed[track_id][-1] if self.smoothed[track_id] else centroid
        alpha = self.ema_alpha

        smoothed_point = Point(
            x=alpha * centroid.x + (1 - alpha) * previous.x,
            y=alpha * centroid.y + (1 - alpha) * previous.y,
        )

        self.smoothed[track_id].append(smoothed_point)
        track.trajectory = list(self.trajectories[track_id])
        track.smoothed_trajectory = list(self.smoothed[track_id])

    def remove(self, track_id: int) -> None:
        self.trajectories.pop(track_id, None)
        self.smoothed.pop(track_id, None)

    def get(self, track_id: int) -> list[Point]:
        return list(self.trajectories.get(track_id, []))

    def get_smoothed(self, track_id: int) -> list[Point]:
        return list(self.smoothed.get(track_id, []))